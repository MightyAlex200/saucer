use futures::stream::FuturesUnordered;
use futures::StreamExt;
use lazy_static::lazy_static;
use opencv::{
    core::{KeyPoint, Vector},
    imgcodecs::{imdecode, ImreadModes},
    prelude::{Feature2DTrait, Mat, MatTrait},
};
use saucer::{new_sift, KPDCache, KP};
use tokio::{io::AsyncWriteExt, sync::Semaphore};
use walkdir::{DirEntry, WalkDir};

lazy_static! {
    static ref WORKING_SEMAPHORE: Semaphore = Semaphore::new(5);
}

#[tokio::main]
async fn main() {
    println!("Building cache...");
    let files: Vec<_> = WalkDir::new("./prod/sources")
        .min_depth(2)
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();
    let file_count = files.len();
    let tasks = FuturesUnordered::new();
    for file in files {
        let handle = tokio::spawn(build_cache_for_file(file));
        tasks.push(handle);
    }
    let mut tasks = tasks.enumerate();
    while let Some((i, _)) = tasks.next().await {
        println!(
            "Processed file {} out of {} ({}%)",
            i,
            file_count,
            ((i + 1) as f32) / file_count as f32 * 100.
        );
    }
}

fn to_2d_vec(mat: Mat) -> Vec<Vec<f32>> {
    // TODO: this sucks
    let mut outer_vec = Vec::with_capacity(mat.cols() as usize);
    for col in 0..mat.cols() {
        let mut inner_vec = Vec::with_capacity(mat.rows() as usize);
        for row in 0..mat.rows() {
            // TODO: this might panic
            let val = mat.at_2d(row, col).unwrap();
            inner_vec.push(*val);
        }
        outer_vec.push(inner_vec);
    }
    outer_vec
}

fn into_kp(keypoint: KeyPoint) -> KP {
    KP {
        angle: keypoint.angle,
        class_id: keypoint.class_id,
        octave: keypoint.octave,
        pt: (keypoint.pt.x, keypoint.pt.y),
        response: keypoint.response,
        size: keypoint.size,
    }
}

async fn build_cache_for_file(file: DirEntry) {
    let name = file.file_name().to_str().unwrap().to_owned();
    let prefix = &name[0..2];
    let output_path = format!("./prod/kpd_cache/{}/{}.bc", prefix, name);
    if tokio::fs::metadata(&output_path).await.is_ok() {
        return;
    }
    let permit = WORKING_SEMAPHORE.acquire().await;
    // Load image, compute keypoints and descriptors, and save
    let bytes: Vector<u8> =
        Vector::from_iter(tokio::fs::read(file.path()).await.unwrap().into_iter());
    let filename = file.path().to_str().unwrap().to_owned();
    let cache;
    // Block to force the compiler to realize that we're not sending Mat types across threads
    // ... This kind of reminds me of Pony's `recover` blocks
    {
        let image = imdecode(&bytes, ImreadModes::IMREAD_COLOR as i32).unwrap();
        let mut keypoints: Vector<KeyPoint> = Vector::new();
        let mut descriptors = Mat::default().unwrap();
        let mut sift = new_sift();
        sift.detect_and_compute(
            &image,
            &Mat::default().unwrap(),
            &mut keypoints,
            &mut descriptors,
            false,
        )
        .unwrap();

        let kps: Vec<_> = keypoints
            .into_iter()
            .map(|keypoint| into_kp(keypoint))
            .collect();
        let serialized_descriptors = to_2d_vec(descriptors);
        cache = KPDCache(filename.clone(), kps, serialized_descriptors);
    }
    let serialized_bytes = bincode::serialize(&cache).unwrap();
    tokio::fs::create_dir(format!("./prod/kpd_cache/{}", prefix))
        .await
        .ok();
    let mut file = tokio::fs::File::create(output_path).await.unwrap();
    file.write_all(&serialized_bytes[..]).await.unwrap();
    drop(permit);
}
