/// A basic implementation of vectors, matrices and minimal math operations implemented as a learning exercise.
/// You are strongly advised to use an established linear algebra library such as `nalgebra` or `cgmath` in
/// preference to this amateur attempt.


#[derive(Clone, Copy, Debug)]
pub struct Matrix3x3 {
    // Internal representation is row first. Columns are identified by letters and rows by numbers (like spreadsheets).
    a1: f32,
    b1: f32,
    c1: f32,
    a2: f32,
    b2: f32,
    c2: f32,
    a3: f32,
    b3: f32,
    c3: f32
}

impl Matrix3x3 {
    /// Create a new 3x3 matrix vector from the nine values passed in. Columns are identified by letters and rows by
    /// numbers (like spreadsheets).
    pub fn new(
        a1: f32, b1: f32, c1: f32,
        a2: f32, b2: f32, c2: f32,
        a3: f32, b3: f32, c3: f32,
    ) -> Self {
        Self {
            a1: a1, b1: b1, c1: c1,
            a2: a2, b2: b2, c2: c2,
            a3: a3, b3: b3, c3: c3,
        }
    }
}


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

    /// Vector length.
    pub fn length(&self) -> f32 {
        f32::sqrt(self.x.powi(2) + self.y.powi(2) + self.z.powi(2))
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

    /// Normalizes the vector, i.e., divides x, y and z by its total length to give a vector that is the same
    /// direction, but exactly 1 unit in length.
    pub fn normalize(&mut self) {
        let length = self.length();
        if length > 0.0 {
            self.x = self.x / length;
            self.y = self.y / length;
            self.z = self.z / length;
        }
    }
}


// TODO Convert all the following into individual tests.
pub fn run_tests() {
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

    let mut h = a.clone();
    println!("Current length of a is {}", h.length());
    h.normalize();
    println!("Normalized a = {:#?}", h);
    println!("New length of a is {}", h.length());

    let m1 = Matrix3x3::new(1.0, 2.0, 4.0, 3.0, 6.0, 12.0, 4.0, 8.0, 16.0);
    println!("m1 = {:#?}", m1);
}
