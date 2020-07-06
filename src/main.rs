use futures::stream::FuturesUnordered;
use opencv::prelude::*;
use opencv::{
    calib3d::{find_homography, RANSAC},
    core::{
        no_array, norm2, KeyPoint, Point2f, Ptr, Scalar, Vector, BORDER_CONSTANT, CV_8UC3, NORM_L2,
    },
    features2d::FlannBasedMatcher,
    flann::{IndexParams, SearchParams, FLANN_INDEX_KDTREE},
    imgcodecs::{imdecode, ImreadModes},
    imgproc::{warp_perspective, WARP_INVERSE_MAP},
    xfeatures2d::SIFT,
};
use serde::Deserialize;
use std::{thread_local, time::Instant};
use tokio::stream::StreamExt;
use walkdir::{DirEntry, WalkDir};

#[derive(Copy, Clone)]
struct MatPtr<'a>(&'a Mat);

impl<'a> MatPtr<'a> {
    unsafe fn new(mat: &'a Mat) -> Self {
        Self(mat)
    }
}

unsafe impl Send for MatPtr<'_> {}

#[derive(Copy, Clone)]
struct KeyPointsPtr<'a>(&'a Vector<KeyPoint>);

impl<'a> KeyPointsPtr<'a> {
    unsafe fn new(kp: &'a Vector<KeyPoint>) -> Self {
        Self(kp)
    }
}

unsafe impl Send for KeyPointsPtr<'_> {}

fn new_sift() -> Ptr<SIFT> {
    SIFT::create(0, 3, 0.04, 10., 1.6).unwrap()
}

#[derive(Deserialize, Debug)]
struct KP {
    angle: f32,
    class_id: i32,
    octave: i32,
    pt: (f32, f32),
    response: f32,
    size: f32,
}

#[derive(Deserialize, Debug)]
struct KPDCache(String, Vec<KP>, Vec<Vec<f32>>);

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

fn difference_between(src1: &Mat, src2: &Mat) -> f64 {
    let error = norm2(&src1, &src2, NORM_L2, &no_array().unwrap()).unwrap();
    let similarity = error / (src1.rows() * src1.cols()) as f64;
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
    cropped: MatPtr<'a>,
    cropped_keypoints: KeyPointsPtr<'a>,
    descriptors: MatPtr<'a>,
) -> Option<(String, f64)> {
    // Read file
    let f = tokio::fs::read_to_string(file.path()).await.ok()?;
    let kpdcache: KPDCache = serde_json::from_str(&f[..]).ok()?;
    let kpd = decode_kpd(kpdcache);
    let uncropped_bytes: Vector<u8> =
        Vector::from_iter(tokio::fs::read(&kpd.0[..]).await.ok()?.into_iter());
    // Use FLANN to find homography
    FLANN.with(|flann| {
        let mut matches = Vector::new();
        flann
            .knn_train_match(
                &descriptors.0,
                &kpd.2,
                &mut matches,
                2,
                &no_array().unwrap(),
                false,
            )
            .ok()?;

        let mut good = Vec::new();
        for matc in matches {
            // matc is always 2 elements long because of the k value passed to knn_train_match
            let m = unsafe { matc.get_unchecked(0) };
            let n = unsafe { matc.get_unchecked(1) };
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
                .map(|m| cropped_keypoints.0.get(m.query_idx as usize).unwrap().pt),
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
        let cropped = cropped.0;
        let uncropped_image = imdecode(&uncropped_bytes, ImreadModes::IMREAD_COLOR as i32).ok()?;
        let uncropped_image = uncropped_image;
        // Unsafe because it exposes uninitialized memory, but we do not access it so there is no UB (🤞)
        let mut dst =
            unsafe { Mat::new_rows_cols(cropped.rows(), cropped.cols(), CV_8UC3).unwrap() };
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
        let diff = difference_between(&cropped, &dst);
        Some((kpd.0, diff))
    })
}

async fn get_uncropped(cropped: Mat) -> Option<(String, f64)> {
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
    let keypoints = keypoints;
    let descriptors = descriptors;

    // This is extremely unsafe, but since Mat isn't Sync it's the best that we can do
    // Technically this is safe because croppedptr is only used while cropped is still in scope
    let croppedptr = unsafe { MatPtr::new((&cropped as *const Mat).as_ref().unwrap()) };
    let descriptorsptr = unsafe { MatPtr::new((&descriptors as *const Mat).as_ref().unwrap()) };
    let keypointsptr =
        unsafe { KeyPointsPtr::new((&keypoints as *const Vector<KeyPoint>).as_ref().unwrap()) };

    // Spawn many tasks
    let tasks = FuturesUnordered::new();
    for file in files {
        let handle = tokio::spawn(get_and_compare_cropped(
            file,
            croppedptr,
            keypointsptr,
            descriptorsptr,
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

#[tokio::main]
async fn main() {
    let start = Instant::now();
    let cropped = opencv::imgcodecs::imread("snivy.png", ImreadModes::IMREAD_COLOR as i32).unwrap();
    match get_uncropped(cropped).await {
        Some((filename, avg_diff)) => {
            let stop = Instant::now();
            let time = stop - start;
            println!(
                "Found closest file! {} Average difference was {}. Took {} seconds",
                filename,
                avg_diff,
                time.as_secs()
            );
        }
        None => {
            println!("Was not able to find a single file that fit the image well enough.");
        }
    }
}
