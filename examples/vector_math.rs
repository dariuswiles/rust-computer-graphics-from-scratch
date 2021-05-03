//! A basic implementation of vectors, matrices and minimal math operations implemented as a learning exercise.
//! You are strongly advised to use an established linear algebra library such as `nalgebra` or `cgmath` in
//! preference to this amateur attempt.

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
    /// Creates a new 3x3 matrix from the nine values passed in. Columns are identified by letters and rows by
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

    // TODO Define math operations as they are needed by the book.

}


#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector3 {
    x: f32,
    y: f32,
    z: f32
}

impl Vector3 {
    /// Creates a new 3D vector from the three values passed in.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {x: x, y: y, z: z}
    }

    /// Returns this vector's length.
    pub fn length(&self) -> f32 {
        f32::sqrt(self.x.powi(2) + self.y.powi(2) + self.z.powi(2))
    }

    /// Multiplies this `Vector3` by a scalar.
    pub fn multiply_by(&self, s: f32) -> Self {
        Self {x: self.x * s, y: self.y * s, z: self.z * s}
    }

    /// Divides this `Vector3` by a scalar.
    pub fn divide_by(&self, s: f32) -> Self {
        Self {x: self.x / s, y: self.y / s, z: self.z / s}
    }

    /// Adds the passed `Vector3` to this one.
    pub fn add(&self, v: &Vector3) -> Self {
        Self {x: self.x + v.x, y: self.y + v.y, z: self.z + v.z}
    }

    /// Subtracts the passed `Vector3` from this one, i.e., returns `self - v`.
    pub fn subtract(&self, v: &Vector3) -> Self {
        Self {x: self.x - v.x, y: self.y - v.y, z: self.z - v.z}
    }

    /// Calculates the dot product of this `Vector3` and the one passed as the parameter.
    pub fn dot(&self, v: &Vector3) -> f32 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    /// Calculates the cross product of this `Vector3` and the one passed as the parameter, i.e., `self × v`.
    pub fn cross(&self, v: &Vector3) -> Self {
        Self {
            x: self.y * v.z - self.z * v.y,
            y: self.z * v.x - self.x * v.z,
            z: self.x * v.y - self.y * v.x,
        }
    }

    /// Normalizes this `Vector3` and returns the normalized version as a new `Vector3`. Normalizing means dividing the
    /// `x`, `y` and `z` components by the `Vector3`'s length, resulting in a `Vector3` that is the same direction, but
    /// `exactly 1 unit in length. If the `Vector3` is zero length, an error is returned
    pub fn normalize(&self) -> Result<Self, ()> {
        let length = self.length();
        if length > 0.0 {
            Result::Ok (Self {
                x: self.x / length,
                y: self.y / length,
                z: self.z / length,
            })
        } else {
            Result::Err(())
        }
    }
}


pub fn run_tests() {
    let a = Vector3::new(0.0, 3.0, 4.0);
    let b = Vector3::new(3.1, 2.2, 1.3);

    let a_len = a.length();
    assert_eq!(a_len, 5.0);

    let a_mult_10 = a.multiply_by(10.0);
    assert_eq!(a_mult_10, Vector3::new(0.0, 30.0, 40.0));

    let a_div_10 = a.divide_by(10.0);
    assert_eq!(a_div_10, Vector3::new(0.0, 0.3, 0.4));

    let a_add_b = a.add(&b);
    assert_eq!(a_add_b, Vector3::new(3.1, 5.2, 5.3));

    let a_sub_b = a.subtract(&b);
    assert_eq!(a_sub_b, Vector3::new(-3.1, 0.79999995, 2.7));

    let a_dot_b = a.dot(&b);
    assert_eq!(a_dot_b, 11.8);

    let c1 = Vector3::new(2.0, 3.0, 4.0);
    let c2 = Vector3::new(5.0, 6.0, 7.0);
    let c1_cross_c2 = c1.cross(&c2);
    assert_eq!(c1_cross_c2, Vector3::new(-3.0, 6.0, -3.0));

    let d1 = Vector3::new(3.0, -3.0, 1.0);
    let d2 = Vector3::new(-12.0, 12.0, -4.0);
    let d1_cross_d2 = d1.cross(&d2);
    assert_eq!(d1_cross_d2, Vector3::new(0.0, 0.0, 0.0));

    let a_norm = a.normalize().unwrap();
    assert_eq!(a_norm, Vector3::new(0.0, 0.6, 0.8));


    println!("\nMatrix math\n");

    let m1 = Matrix3x3::new(1.0, 2.0, 4.0, 3.0, 6.0, 12.0, 4.0, 8.0, 16.0);
    println!("m1 = {:#?}", m1);
}

/// Run tests for the code in this file.
fn main() {
    run_tests();
    println!("Test run complete");
}
