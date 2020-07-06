# Saucer
This is an application to find the original uncropped image in a local database from a cropped image.
It uses OpenCVs implementation of SIFT for feature detection, and FLANN for feature matching.

It's new implementation is written in Rust, but some python notebooks currently exist to set up the database cache.

## Installation of OpenCV and Compilation
SIFT is not included in OpenCV by default, because it is not a free algorithm. You must build your own version of OpenCV.
This document will not explain how to do that, but here are some cmake flags you will probably need:
```
-DOPENCV_ENABLE_NONFREE=ON -DOPENCV_EXTRA_MODULES_PATH=<path_to_contrib>/modules -DBUILD_opencv_legacy=OFF -DBUILD_opencv_world=ON
```

Once you've built OpenCV, you might need to set some environment variables if they are not already set. See https://github.com/twistedfall/opencv-rust for more information.

To build the Rust program, run this command:
```
cargo build --release
```

## Populating the database
Images are expected to be found in `./prod/sources/<ab>/<abcdef>.<ext>`, where `ab` is the first two characters of the filename, and `ext` is the file extension.
This arrangement helps relieve stress on some filesystems by not creating a single directory with far too many files.
There is no included software to do this, if you do not already have software that can do this you will need to make your own.

## Database cache
Feature detection is very computationally expensive, and doing it on every query is highly inefficient.
To get around this, the program uses a cache of json files to store the results of SIFT feature detection.
To build the cache, open `generate_prod_cache.ipynb` in a python notebook (e.g. Jupyter Lab) and run the cells in order.
Depending on how many images you have this may take some time.

## Usage
The program is not finished yet, so usage is not user friendly at the moment.
To use the Rust version, create a file named `cropped.png`, then run
```
cargo run --release
```
To search for the uncropped image in the database.
