/// A basic implementation of vectors, matrices and minimal math operations implemented as a learning exercise.
/// You are strongly advised to use an established linear algebra library such as `nalgebra` or `cgmath` in
/// preference to this amateur attempt.

#[derive(Clone, Copy, Debug)]
pub struct Vector3 {
    x: f32,
    y: f32,
    z: f32
}

impl Vector3 {
    /// Create a new 3-D vector from the three values passed in.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {x: x, y: y, z: z}
    }

    /// Multiply by a scalar.
    pub fn multiply_by(&self, s: f32) -> Self {
        Self {x: self.x * s, y: self.y * s, z: self.z * s}
    }

    /// Divide by a scalar.
    pub fn divide_by(&self, s: f32) -> Self {
        Self {x: self.x / s, y: self.y / s, z: self.z / s}
    }

    /// Add two `Vector3`s.
    pub fn add(&self, v: &Vector3) -> Self {
        Self {x: self.x + v.x, y: self.y + v.y, z: self.z + v.z}
    }

    /// Subtract the passed `Vector3` from this one.
    pub fn subtract(&self, v: &Vector3) -> Self {
        Self {x: self.x - v.x, y: self.y - v.y, z: self.z - v.z}
    }

    /// Calculate the dot product of this vector and the one passed as the parameter.
    pub fn dot(&self, v: &Vector3) -> f32 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}




pub fn tmp() {
    let a = Vector3::new(1.0, 2.0, 3.0);
    println!("a = {:#?}", a);

    let b = a.multiply_by(10.0);
    println!("a * 10.0 = {:#?}", b);

    let c = a.divide_by(10.0);
    println!("a / 10.0 = {:#?}", c);

    let d = Vector3::new(3.0, 2.0, 1.0);
    println!("d = {:#?}", d);

    let e = a.add(&d);
    println!("a + d = {:#?}", e);

    let f = a.subtract(&d);
    println!("a - d = {:#?}", f);

    let g = a.dot(&d);
    println!("a ⋅ d = {:#?}", g);

}
