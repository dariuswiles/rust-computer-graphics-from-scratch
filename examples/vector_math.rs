#[derive(Clone, Copy, Debug)]
pub struct Vector3 {
    data: [f32; 3],
}

impl Vector3 {
    pub fn new(v0: f32, v1: f32, v2: f32) -> Self {
        Self {data: [v0, v1, v2]}
    }

    /// Multiply by a scalar
    pub fn multiply_by(&self, s: f32) -> Self {
        Self {data: [self.data[0] * s, self.data[1] * s, self.data[2] * s]}
    }

    /// Divide by a scalar
    pub fn divide_by(&self, s: f32) -> Self {
        Self {data: [self.data[0] / s, self.data[1] / s, self.data[2] / s]}
    }

    /// Add two `Vector3`s
    pub fn add(&self, v: &Vector3) -> Self {
        Self {data: [self.data[0] + v.data[0], self.data[1] + v.data[1], self.data[2] + v.data[2]]}
    }

    /// Subtract the passed `Vector3` from this one
    pub fn subtract(&self, v: &Vector3) -> Self {
        Self {data: [self.data[0] - v.data[0], self.data[1] - v.data[1], self.data[2] - v.data[2]]}
    }

    pub fn dot(&self, v: &Vector3) -> f32 {
        self.data[0] * v.data[0] + self.data[1] * v.data[1] + self.data[2] * v.data[2]
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
