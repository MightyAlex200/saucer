use opencv::imgcodecs::ImreadModes;
use saucer::get_uncropped;
use tokio::time::Instant;

#[tokio::main]
async fn main() {
    let start = Instant::now();
    let cropped =
        opencv::imgcodecs::imread("cropped.png", ImreadModes::IMREAD_COLOR as i32).unwrap();
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
