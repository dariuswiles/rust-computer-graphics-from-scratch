//! A basic implementation of vectors, matrices and minimal math operations implemented as a
//! learning exercise. You are strongly advised to use an established linear algebra library such
//! as `nalgebra` or `cgmath` in preference to this amateur attempt.

use std::f64::consts::PI;


/// A 3 row and 3 column matrix. The individual fields are identified with letters for columns and
/// numbers for rows, similar to the system used by most spreadsheets. In other words, the fields
/// are named:
///   a1  b1  c1
///   a2  b2  c2
///   a3  b3  c3
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix3x3 {
    pub a1: f64,
    pub b1: f64,
    pub c1: f64,
    pub a2: f64,
    pub b2: f64,
    pub c2: f64,
    pub a3: f64,
    pub b3: f64,
    pub c3: f64
}

impl Matrix3x3 {
    /// Creates a new 3x3 matrix from the nine values passed in. Columns are identified by letters
    /// and rows by numbers (like spreadsheets).
    #[allow(dead_code)]
    pub fn new(
        a1: f64, b1: f64, c1: f64,
        a2: f64, b2: f64, c2: f64,
        a3: f64, b3: f64, c3: f64,
    ) -> Self {
        Self {
            a1: a1, b1: b1, c1: c1,
            a2: a2, b2: b2, c2: c2,
            a3: a3, b3: b3, c3: c3,
        }
    }

    /// Multiplies this `Matrix3x3` instance with the `Vector3` passed and returns the result as a
    /// new `Vector3` instance.
    #[allow(dead_code)]
    pub fn multiply_vector(&self, v: &Vector3) -> Vector3 {
        let row1 = &self.a1 * v.x + &self.b1 * v.y + &self.c1 * v.z;
        let row2 = &self.a2 * v.x + &self.b2 * v.y + &self.c2 * v.z;
        let row3 = &self.a3 * v.x + &self.b3 * v.y + &self.c3 * v.z;

        Vector3::new(row1, row2, row3)
    }
}



/// A 4 row and 4 column matrix. The individual fields are identified with letters for columns and
/// numbers for rows, similar to the system used by most spreadsheets. In other words, the fields
/// are named:
///   a1  b1  c1  d1
///   a2  b2  c2  d2
///   a3  b3  c3  d3
///   a4  b4  c4  d4
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix4x4 {
    pub a1: f64,
    pub b1: f64,
    pub c1: f64,
    pub d1: f64,
    pub a2: f64,
    pub b2: f64,
    pub c2: f64,
    pub d2: f64,
    pub a3: f64,
    pub b3: f64,
    pub c3: f64,
    pub d3: f64,
    pub a4: f64,
    pub b4: f64,
    pub c4: f64,
    pub d4: f64,
}

impl Matrix4x4 {
    /// Creates a new 4x4 matrix from the nine values passed in. Columns are identified by letters
    /// and rows by numbers (like spreadsheets).
    #[allow(dead_code)]
    pub fn new(
        a1: f64, b1: f64, c1: f64, d1: f64,
        a2: f64, b2: f64, c2: f64, d2: f64,
        a3: f64, b3: f64, c3: f64, d3: f64,
        a4: f64, b4: f64, c4: f64, d4: f64,
    ) -> Self {
        Self {
            a1: a1, b1: b1, c1: c1, d1: d1,
            a2: a2, b2: b2, c2: c2, d2: d2,
            a3: a3, b3: b3, c3: c3, d3: d3,
            a4: a4, b4: b4, c4: c4, d4: d4,
        }
    }


    /// Creates a new 4x4 matrix identity and returns it.
    #[allow(dead_code)]
    pub fn identity() -> Self {
        Self {
            a1: 1.0, b1: 0.0, c1: 0.0, d1: 0.0,
            a2: 0.0, b2: 1.0, c2: 0.0, d2: 0.0,
            a3: 0.0, b3: 0.0, c3: 1.0, d3: 0.0,
            a4: 0.0, b4: 0.0, c4: 0.0, d4: 1.0,
        }
    }


    /// Creates a new 4x4 transform matrix representing a rotation around the axis that runs from
    /// the origin along the Y-axis. The rotation parameter is in degrees.
    #[allow(dead_code)]
    pub fn new_oy_rotation_matrix(degrees: f64) -> Self {
        let (s, c) = f64::sin_cos(degrees*PI/180.0);

        Self {
            a1: c,   b1: 0.0, c1: -s,  d1: 0.0,
            a2: 0.0, b2: 1.0, c2: 0.0, d2: 0.0,
            a3: s,   b3: 0.0, c3: c,   d3: 0.0,
            a4: 0.0, b4: 0.0, c4: 0.0, d4: 1.0,
        }
    }


    /// Creates a new 4x4 transform matrix representing the 3D translation passed as a `Vector3`.
    #[allow(dead_code)]
    pub fn new_translation_matrix(translation: &Vector3) -> Matrix4x4 {
        Self {
            a1: 1.0, b1: 0.0, c1: 0.0, d1: translation.x,
            a2: 0.0, b2: 1.0, c2: 0.0, d2: translation.y,
            a3: 0.0, b3: 0.0, c3: 1.0, d3: translation.z,
            a4: 0.0, b4: 0.0, c4: 0.0, d4: 1.0,
        }
    }


   /// Creates a new 4x4 transform matrix representing the scaling passed as an `f64`.
    #[allow(dead_code)]
    pub fn new_scaling_matrix(sc: f64) -> Matrix4x4 {
        Self {
            a1: sc,  b1: 0.0, c1: 0.0, d1: 0.0,
            a2: 0.0, b2: sc,  c2: 0.0, d2: 0.0,
            a3: 0.0, b3: 0.0, c3: sc,  d3: 0.0,
            a4: 0.0, b4: 0.0, c4: 0.0, d4: 1.0,
        }
    }


    /// Multiplies this `Matrix4x4` instance with the `Vector4` passed and returns the result as a
    /// new `Vector4` instance.
    #[allow(dead_code)]
    pub fn multiply_vector(&self, v: &Vector4) -> Vector4 {
        Vector4::new(
            &self.a1 * v.x + &self.b1 * v.y + &self.c1 * v.z + &self.d1 * v.w,
            &self.a2 * v.x + &self.b2 * v.y + &self.c2 * v.z + &self.d2 * v.w,
            &self.a3 * v.x + &self.b3 * v.y + &self.c3 * v.z + &self.d3 * v.w,
            &self.a4 * v.x + &self.b4 * v.y + &self.c4 * v.z + &self.d4 * v.w,
        )
    }


    /// Transposes this `Matrix4x4` instance and returns the result as a new `Matrix4x4` instance.
    #[allow(dead_code)]
    pub fn transpose(&self) -> Self {
        Self {
            a1: self.a1, b1: self.a2, c1: self.a3, d1: self.a4,
            a2: self.b1, b2: self.b2, c2: self.b3, d2: self.b4,
            a3: self.c1, b3: self.c2, c3: self.c3, d3: self.c4,
            a4: self.d1, b4: self.d2, c4: self.d3, d4: self.d4,
        }
    }

    /// Multiplies this `Matrix4x4` instance with the `Matrix4x4` passed and returns the result as
    /// a new `Matrix4x4` instance.
    #[allow(dead_code)]
    pub fn multiply_matrix4x4(&self, m: &Matrix4x4) -> Matrix4x4 {
        Self {
            a1: &self.a1 * m.a1 + &self.b1 * m.a2 + &self.c1 * m.a3 + &self.d1 * m.a4,
            b1: &self.a1 * m.b1 + &self.b1 * m.b2 + &self.c1 * m.b3 + &self.d1 * m.b4,
            c1: &self.a1 * m.c1 + &self.b1 * m.c2 + &self.c1 * m.c3 + &self.d1 * m.c4,
            d1: &self.a1 * m.d1 + &self.b1 * m.d2 + &self.c1 * m.d3 + &self.d1 * m.d4,
            a2: &self.a2 * m.a1 + &self.b2 * m.a2 + &self.c2 * m.a3 + &self.d2 * m.a4,
            b2: &self.a2 * m.b1 + &self.b2 * m.b2 + &self.c2 * m.b3 + &self.d2 * m.b4,
            c2: &self.a2 * m.c1 + &self.b2 * m.c2 + &self.c2 * m.c3 + &self.d2 * m.c4,
            d2: &self.a2 * m.d1 + &self.b2 * m.d2 + &self.c2 * m.d3 + &self.d2 * m.d4,
            a3: &self.a3 * m.a1 + &self.b3 * m.a2 + &self.c3 * m.a3 + &self.d3 * m.a4,
            b3: &self.a3 * m.b1 + &self.b3 * m.b2 + &self.c3 * m.b3 + &self.d3 * m.b4,
            c3: &self.a3 * m.c1 + &self.b3 * m.c2 + &self.c3 * m.c3 + &self.d3 * m.c4,
            d3: &self.a3 * m.d1 + &self.b3 * m.d2 + &self.c3 * m.d3 + &self.d3 * m.d4,
            a4: &self.a4 * m.a1 + &self.b4 * m.a2 + &self.c4 * m.a3 + &self.d4 * m.a4,
            b4: &self.a4 * m.b1 + &self.b4 * m.b2 + &self.c4 * m.b3 + &self.d4 * m.b4,
            c4: &self.a4 * m.c1 + &self.b4 * m.c2 + &self.c4 * m.c3 + &self.d4 * m.c4,
            d4: &self.a4 * m.d1 + &self.b4 * m.d2 + &self.c4 * m.d3 + &self.d4 * m.d4,
        }
    }
}



/// A 3 element vector. The individual fields are named `x`, `y` and `z`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64
}

impl Vector3 {
    /// Creates a new 3D vector from the three values passed in.
    #[allow(dead_code)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {x: x, y: y, z: z}
    }

    /// Returns this vector's length.
    #[allow(dead_code)]
    pub fn length(&self) -> f64 {
        f64::sqrt(self.x.powi(2) + self.y.powi(2) + self.z.powi(2))
    }

    /// Multiplies this `Vector3` by a scalar.
    #[allow(dead_code)]
    pub fn multiply_by(&self, s: f64) -> Self {
        Self {x: self.x * s, y: self.y * s, z: self.z * s}
    }

    /// Divides this `Vector3` by a scalar.
    #[allow(dead_code)]
    pub fn divide_by(&self, s: f64) -> Self {
        Self {x: self.x / s, y: self.y / s, z: self.z / s}
    }

    /// Adds the passed `Vector3` to this one.
    #[allow(dead_code)]
    pub fn add(&self, v: &Vector3) -> Self {
        Self {x: self.x + v.x, y: self.y + v.y, z: self.z + v.z}
    }

    /// Subtracts the passed `Vector3` from this one, i.e., returns `self - v`.
    #[allow(dead_code)]
    pub fn subtract(&self, v: &Vector3) -> Self {
        Self {x: self.x - v.x, y: self.y - v.y, z: self.z - v.z}
    }

    /// Calculates the dot product of this `Vector3` and the one passed as the parameter.
    #[allow(dead_code)]
    pub fn dot(&self, v: &Vector3) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    /// Calculates the cross product of this `Vector3` and the one passed as the parameter, i.e.,
    /// `self × v`.
    #[allow(dead_code)]
    pub fn cross(&self, v: &Vector3) -> Self {
        Self {
            x: self.y * v.z - self.z * v.y,
            y: self.z * v.x - self.x * v.z,
            z: self.x * v.y - self.y * v.x,
        }
    }

    /// Normalizes this `Vector3` and returns the normalized version as a new `Vector3`.
    /// Normalizing means dividing the `x`, `y` and `z` components by the `Vector3`'s length,
    /// resulting in a `Vector3` that is the same direction, but `exactly 1 unit in length. If the
    /// `Vector3` is zero length, an error is returned.
    #[allow(dead_code)]
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


/// A 4 element vector. The individual fields are named `x`, `y`, `z` and `w`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vector4 {
    /// Creates a new 3D vector from the three values passed in.
    #[allow(dead_code)]
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self {x: x, y: y, z: z, w: w}
    }

    /// Returns this vector's length.
    #[allow(dead_code)]
    pub fn length(&self) -> f64 {
        f64::sqrt(self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + self.w.powi(2))
    }

    /// Multiplies this `Vector4` by a scalar.
    #[allow(dead_code)]
    pub fn multiply_by(&self, s: f64) -> Self {
        Self {x: self.x * s, y: self.y * s, z: self.z * s, w: self.w * s}
    }

    /// Divides this `Vector4` by a scalar.
    #[allow(dead_code)]
    pub fn divide_by(&self, s: f64) -> Self {
        Self {x: self.x / s, y: self.y / s, z: self.z / s, w: self.w / s}
    }

    /// Adds the passed `Vector4` to this one.
    #[allow(dead_code)]
    pub fn add(&self, v: &Vector4) -> Self {
        Self {x: self.x + v.x, y: self.y + v.y, z: self.z + v.z, w: self.w + v.w}
    }

    /// Subtracts the passed `Vector4` from this one, i.e., returns `self - v`.
    #[allow(dead_code)]
    pub fn subtract(&self, v: &Vector4) -> Self {
        Self {x: self.x - v.x, y: self.y - v.y, z: self.z - v.z, w: self.w - v.w}
    }

    /// Calculates the dot product of this `Vector4` and the one passed as the parameter.
    #[allow(dead_code)]
    pub fn dot(&self, v: &Vector4) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z + self.w * v.w
    }
}



#[allow(dead_code)]
fn run_tests() {
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
    assert_eq!(a_sub_b, Vector3::new(-3.1, 0.7999999999999998, 2.7));

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


    // Matrix math tests

    let m1 = Matrix3x3::new(1.0, 2.0,  4.0,
                            3.0, 6.0, 12.0,
                            4.0, 8.0, 16.0);
    let v1 = Vector3::new(3.1, 2.2, 1.3);
    let m1_multiply_v1 = m1.multiply_vector(&v1);
    assert_eq!(m1_multiply_v1, Vector3::new(12.7, 38.1, 50.8));




    let m40 = Matrix4x4::new(1.0, 1.25, 1.5, 1.75,
                             2.0, 2.25, 2.5, 2.75,
                             3.0, 3.25, 3.5, 3.75,
                             4.0, 4.25, 4.5, 4.75);
    let m40_transposed = m40.transpose();

    let expected_m40_transposed = Matrix4x4::new(1.0,  2.0,  3.0,  4.0,
                                                 1.25, 2.25, 3.25, 4.25,
                                                 1.5,  2.5,  3.5,  4.5,
                                                 1.75, 2.75, 3.75, 4.75);
    assert_eq!(m40_transposed, expected_m40_transposed);

    let m40_transposed_twice = m40_transposed.transpose();
    assert_eq!(m40_transposed_twice, m40);


    let v41 = Vector4::new(3.0, 2.2, 1.4, 0.6);
    let m40_multiply_v41 = m40.multiply_vector(&v41);
    assert_eq!(m40_multiply_v41, Vector4::new(8.9, 16.099999999999998, 23.299999999999997,
                                              30.500000000000004));


    let m40_multiply_m40 = m40.multiply_matrix4x4(&m40);
    let expected_m40_m40 = Matrix4x4::new(15.0, 16.375, 17.75, 19.125,
                                          25.0, 27.375, 29.75, 32.125,
                                          35.0, 38.375, 41.75, 45.125,
                                          45.0, 49.375, 53.75, 58.125);
    assert_eq!(m40_multiply_m40, expected_m40_m40);
}

/// Run tests for the code in this file.
#[allow(dead_code)]
fn main() {
    run_tests();
    println!("Test run complete");
}
