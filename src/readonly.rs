//! Read-only wrappers for sharing OpenCV objects across threads

use opencv::{
    core::{Size, ToInputArray, Vector, VectorElement, VectorExtern, _InputArray},
    platform_types::size_t,
    prelude::{Mat, MatTrait, MatTraitManual},
};

pub struct ReadOnlyMat(Mat);

impl ReadOnlyMat {
    pub fn new(inner: Mat) -> Self {
        Self(inner)
    }

    pub fn rows(&self) -> i32 {
        self.0.rows()
    }

    pub fn cols(&self) -> i32 {
        self.0.cols()
    }

    pub fn size(&self) -> opencv::Result<Size> {
        self.0.size()
    }

    pub fn typ(&self) -> opencv::Result<i32> {
        self.0.typ()
    }

    pub fn into_inner(self) -> Mat {
        self.0
    }

    pub unsafe fn into_inner_ref(&self) -> &Mat {
        &self.0
    }

    pub unsafe fn into_inner_mut(&mut self) -> &mut Mat {
        &mut self.0
    }
}

impl ToInputArray for ReadOnlyMat {
    fn input_array(&self) -> opencv::Result<_InputArray> {
        self.0.input_array()
    }
}

// I can only hope this is safe ¯\_(ツ)_/¯
unsafe impl Send for ReadOnlyMat {}
unsafe impl Sync for ReadOnlyMat {}

// Read-only vector
pub struct ReadOnlyVector<T>(Vector<T>)
where
    T: VectorElement,
    Vector<T>: VectorExtern<T>;

impl<T: VectorElement> ReadOnlyVector<T>
where
    Vector<T>: VectorExtern<T>,
{
    pub fn new(inner: Vector<T>) -> Self {
        Self(inner)
    }

    pub fn get(&self, index: size_t) -> opencv::Result<T> {
        self.0.get(index)
    }

    pub unsafe fn get_unchecked(&self, index: size_t) -> T {
        self.0.get_unchecked(index)
    }

    pub fn into_inner(self) -> Vector<T> {
        self.0
    }

    pub unsafe fn into_inner_ref(&self) -> &Vector<T> {
        &self.0
    }

    pub unsafe fn into_inner_mut(&mut self) -> &mut Vector<T> {
        &mut self.0
    }
}

// TODO: this isn't safe unless T is Sync/Send, so more wrapper types are needed
unsafe impl<T: VectorElement> Send for ReadOnlyVector<T> where Vector<T>: VectorExtern<T> {}
unsafe impl<T: VectorElement> Sync for ReadOnlyVector<T> where Vector<T>: VectorExtern<T> {}
