use futures::stream::FuturesUnordered;
use lazy_static::lazy_static;
use opencv::prelude::*;
use opencv::{
    calib3d::{find_homography, RANSAC},
    core::{
        no_array, norm2, KeyPoint, Point2f, Ptr, Scalar, ToInputArray, Vector, BORDER_CONSTANT,
        CV_8UC3, NORM_L2,
    },
    features2d::FlannBasedMatcher,
    flann::{IndexParams, SearchParams, FLANN_INDEX_KDTREE},
    imgcodecs::{imdecode, ImreadModes},
    imgproc::{warp_perspective, WARP_INVERSE_MAP},
    xfeatures2d::SIFT,
};
use readonly::{ReadOnlyMat, ReadOnlyVector};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, thread_local};
use tokio::{stream::StreamExt, sync::Semaphore};
use walkdir::{DirEntry, WalkDir};

mod readonly;

const MAX_CACHE_LOADED: usize = 200;
const MAX_IMAGES_LOADED: usize = 60;

lazy_static! {
    static ref CACHE_SEMAPHORE: Semaphore = Semaphore::new(MAX_CACHE_LOADED);
    static ref IMAGE_LOAD_SEMAPHORE: Semaphore = Semaphore::new(MAX_IMAGES_LOADED);
}

pub fn new_sift() -> Ptr<SIFT> {
    SIFT::create(0, 3, 0.04, 10., 1.6).unwrap()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct KP {
    pub angle: f32,
    pub class_id: i32,
    pub octave: i32,
    pub pt: (f32, f32),
    pub response: f32,
    pub size: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct KPDCache(pub String, pub Vec<KP>, pub Vec<Vec<f32>>);

fn decode_kpd(cache: KPDCache) -> (String, Vector<KeyPoint>, Mat) {
    (
        cache.0,
        Vector::from_iter(cache.1.into_iter().map(|kp| KeyPoint {
            pt: Point2f::new(kp.pt.0, kp.pt.1),
            size: kp.size,
            angle: kp.angle,
            response: kp.response,
            octave: kp.octave,
            class_id: kp.class_id,
        })),
        Mat::from_slice_2d(&cache.2).unwrap(),
    )
}

fn difference_between(
    src1: &dyn ToInputArray,
    src2: &dyn ToInputArray,
    rows: i32,
    cols: i32,
) -> f64 {
    let error = norm2(src1, src2, NORM_L2, &no_array().unwrap()).unwrap();
    let similarity = error / (rows * cols) as f64;
    similarity
}

thread_local! {
    static FLANN: FlannBasedMatcher = {
        // Create flann
        let mut index_params = IndexParams::default().unwrap();
        index_params.set_algorithm(FLANN_INDEX_KDTREE).unwrap();
        index_params.set_int("trees", 5).unwrap();
        let search_params = SearchParams::new(50, 0., true).unwrap();

        FlannBasedMatcher::new(&Ptr::new(index_params), &Ptr::new(search_params)).unwrap()
    }
}

async fn get_and_compare_cropped<'a>(
    file: DirEntry,
    cropped: Arc<ReadOnlyMat>,
    cropped_keypoints: Arc<ReadOnlyVector<KeyPoint>>,
    descriptors: Arc<ReadOnlyMat>,
) -> Option<(String, f64)> {
    let cache_permit = CACHE_SEMAPHORE.acquire().await;
    // Read file
    let f = tokio::fs::read(file.path()).await.ok()?;
    let kpdcache: KPDCache = bincode::deserialize(&f[..]).ok()?;
    let kpd = decode_kpd(kpdcache);
    // Use FLANN to find homography
    let matches = FLANN.with(|flann| {
        let mut matches = Vector::new();
        flann
            .knn_train_match(
                &*descriptors,
                &kpd.2,
                &mut matches,
                2,
                &no_array().unwrap(),
                false,
            )
            .unwrap();
        Some(matches)
    })?;
    let mut good = Vec::new();
    for matc in matches {
        let m = match matc.get(0) {
            Ok(x) => x,
            Err(_) => continue,
        };
        let n = match matc.get(1) {
            Ok(x) => x,
            Err(_) => continue,
        };
        if m.distance < 0.7 * n.distance {
            good.push(m);
        }
    }
    let good = good;
    if good.is_empty() {
        return None;
    }
    let uncropped_points: Vector<Point2f> = Vector::from_iter(
        good.iter()
            .map(|m| kpd.1.get(m.train_idx as usize).unwrap().pt),
    );
    let cropped_points: Vector<Point2f> = Vector::from_iter(
        good.iter()
            .map(|m| cropped_keypoints.get(m.query_idx as usize).unwrap().pt),
    );
    // Compare cropped and uncropped
    let m = find_homography(
        &cropped_points,
        &uncropped_points,
        &mut no_array().unwrap(),
        RANSAC,
        5.,
    )
    .ok()?;
    let permit = IMAGE_LOAD_SEMAPHORE.acquire().await;
    let uncropped_bytes: Vector<u8> =
        Vector::from_iter(tokio::fs::read(&kpd.0[..]).await.ok()?.into_iter());
    let uncropped_image = imdecode(&uncropped_bytes, ImreadModes::IMREAD_COLOR as i32).ok()?;
    // Unsafe because it exposes uninitialized memory, but we do not access it so there is no UB (🤞)
    let mut dst = unsafe { Mat::new_rows_cols(cropped.rows(), cropped.cols(), CV_8UC3).unwrap() };
    warp_perspective(
        &uncropped_image,
        &mut dst,
        &m,
        cropped.size().unwrap(),
        WARP_INVERSE_MAP as i32,
        BORDER_CONSTANT as i32,
        Scalar::default(),
    )
    .ok()?;
    assert_eq!(dst.typ().unwrap(), cropped.typ().unwrap());
    let diff = difference_between(&*cropped, &dst, cropped.rows(), cropped.cols());
    drop(cache_permit);
    drop(permit);
    Some((kpd.0, diff))
}

pub async fn get_uncropped(cropped: Mat) -> Option<(String, f64)> {
    let files: Vec<_> = WalkDir::new("./prod/kpd_cache")
        .min_depth(2)
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

    // Get keypoints and descriptors for cropped image
    let mut keypoints: Vector<KeyPoint> = Vector::new();
    let mut descriptors = Mat::default().unwrap();
    let mut sift = new_sift();
    sift.detect_and_compute(
        // cropped,
        &cropped,
        &Mat::default().unwrap(),
        &mut keypoints,
        &mut descriptors,
        false,
    )
    .unwrap();
    let cropped = Arc::new(ReadOnlyMat::new(cropped));
    let keypoints = Arc::new(ReadOnlyVector::new(keypoints));
    let descriptors = Arc::new(ReadOnlyMat::new(descriptors));

    // Spawn many tasks
    let tasks = FuturesUnordered::new();
    for file in files {
        let cropped_ref = cropped.clone();
        let keypoints_ref = keypoints.clone();
        let descriptors_ref = descriptors.clone();
        let handle = tokio::spawn(get_and_compare_cropped(
            file,
            cropped_ref,
            keypoints_ref,
            descriptors_ref,
        ));
        tasks.push(handle);
    }
    // Join all tasks
    let mut best_match = None;
    for o in tasks
        .filter_map(|r| r.ok())
        .collect::<Vec<Option<(String, f64)>>>()
        .await
    {
        if let Some((filename, diff)) = o {
            best_match = match best_match {
                Some((oldfilename, olddiff)) => {
                    if diff < olddiff {
                        Some((filename, diff))
                    } else {
                        Some((oldfilename, olddiff))
                    }
                }
                None => Some((filename, diff)),
            }
        }
    }
    best_match
}
