//! Implementation of pseudocode from chapter 8 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. I am not affiliated with Gabriel or his book in any way.
//!
//! This code displays filled triangles using interpolated shading.

use std::mem;
use std::iter::Iterator;
use std::vec::Vec;
use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
#[allow(dead_code)]
mod vector_math;

const CANVAS_WIDTH: usize = 600;
const CANVAS_HEIGHT: usize = 600;


/// Iterates over the range `i0` to `i1` inclusive, interpolating over a dependent range from `d0`
/// to `d1` inclusive.
///
/// Returns a vector of (f64, f64) pairs, where the former is the independent variable, and the
/// latter the dependent variable.
///
/// Note: This implementation differs from the book by returning dependent and independent
///       variables, rather than only the dependent variable. This is done because it reduces the
///       code size of the program as a whole.
///
/// # Examples
///
/// ```
/// let i = interpolate(10, 500.0, 14, 501.0);
/// assert_eq!(i, [(10, 500.0), (11, 500.25), (12, 500.5), (13, 500.75), (14, 501.0)]);
/// ```
fn interpolate(i0: f64, d0: f64, i1: f64, d1: f64) -> Vec<(f64, f64)> {
    let mut values = vec![];

    if i0 == i1 {
        values.push((i0, d0));
        return values;
    }

    let range: Vec<i32>;
    let a;

    if i0 < i1 {
        range = (i0.round() as i32 ..= i1.round() as i32).into_iter().collect();
        a = (d1 - d0) / (i1 - i0);
    } else {
        range = (i1.round() as i32 ..= i0.round() as i32).rev().into_iter().collect();
        a = (d1 - d0) / (i0 - i1);
    }

    let mut d = d0;

    for i in range {
        values.push((i as f64, d));
        d = d + a;
    }

    values
}


/// A 2D point with intensity value.
#[derive(Clone, Copy, Debug)]
struct Point {
    pub x: f64,
    pub y: f64,
    pub h: f64,  // Intensity
}

impl Point {
    #[allow(dead_code)]
    fn new(x: f64, y: f64, h: f64) -> Self {
        Self {x: x, y: y, h: h}
    }

    // Create a new `Point`, where `x` and `y` are given as `i32`s.
    fn from_ints(x: i32, y: i32, h: f64) -> Self {
        Self {x: x as f64, y: y as f64, h: h}
    }
}


/// Draws a filled triangle defined by the three points passed in, and the color passed. Each
/// `Point` object contains an intensity value `h`, that is interpolated between corners to give
/// a smooth transition between the corners.
fn draw_filled_triangle (canvas: &mut Canvas, p0: &Point, p1: &Point, p2: &Point, color: &Rgb) {
    let mut corner0 = p0;
    let mut corner1 = p1;
    let mut corner2 = p2;

    // Swap the three corners by height, such that: corner0 <= corner1 <= corner2.
    if corner1.y < corner0.y { mem::swap (&mut corner1, &mut corner0); }
    if corner2.y < corner0.y { mem::swap (&mut corner2, &mut corner0); }
    if corner2.y < corner1.y { mem::swap (&mut corner2, &mut corner1); }

    // Interpolate with the `y` coordinates as the independent variable because we want the value
    // `x` for each row (rather than looping over `x` to find `y`). The results are `vec`s of
    // `(f64, f64)`, representing `(y, x)` coordinates. Also interpolate intensity values.
    let x01 = interpolate(corner0.y, corner0.x, corner1.y, corner1.x);
    let h01 = interpolate(corner0.y, corner0.h, corner1.y, corner1.h);
    let x12 = interpolate(corner1.y, corner1.x, corner2.y, corner2.x);
    let h12 = interpolate(corner1.y, corner1.h, corner2.y, corner2.h);
    let x02 = interpolate(corner0.y, corner0.x, corner2.y, corner2.x);
    let h02 = interpolate(corner0.y, corner0.h, corner2.y, corner2.h);

    // Concatenate `x01` and `x12`, but remove the value at the end of `x01` as it is repeated as
    // the first value of `x12`
    let x012 = [&x01[..x01.len()-1], &x12[..]].concat();
    let h012 = [&h01[..h01.len()-1], &h12[..]].concat();

    let x_left;
    let x_right;
    let h_left;
    let h_right;
    let m = x02.len() / 2;

    // Look at the middle row of the triangle to determine whether `x02` or `x012` represents the
    // left side of the triangle.
    if x02[m].1 < x012[m].1 {   // Note that field `0` holds `x` coords, and `1` holds `y`.
        x_left = x02;
        h_left = h02;
        x_right = x012;
        h_right = h012;
    } else {
        x_left = x012;
        h_left = h012;
        x_right = x02;
        h_right = h02;
    }

    // For every line, draw a row between the left and right sides of the triangle.
    for y in corner0.y.round() as i32 .. corner2.y.round() as i32 {
        let x_start = x_left.get((y - corner0.y.round() as i32) as usize).unwrap().1.round() as i32;
        let x_end = x_right.get((y - corner0.y.round() as i32) as usize).unwrap().1.round() as i32;

        let h_segment = interpolate(x_start as f64,
                                    h_left.get((y - corner0.y.round() as i32) as usize)
                                        .unwrap().1,
                                    x_end as f64,
                                    h_right.get((y - corner0.y.round() as i32) as usize)
                                        .unwrap().1);
        for x in x_start .. x_end {
            canvas.put_pixel(x, y,
                             &color.multiply_by(h_segment.get((x - x_start) as usize).unwrap().1));
        }
    }
}


/// Creates a window and draws some test lines using rasterization techniques.
fn main() {

    let mut canvas = Canvas::new("Raster 04 (from chapter 8)", CANVAS_WIDTH, CANVAS_HEIGHT);

    let white = Rgb::from_ints(255,255,255);
    let green = Rgb::from_ints(0,255,0);

    let p0 = Point::from_ints(-200, -250, 0.3);
    let p1 = Point::from_ints(200, 50, 0.1);
    let p2 = Point::from_ints(20, 250, 1.0);


    canvas.clear_canvas(&white);
    draw_filled_triangle (&mut canvas, &p0, &p1, &p2, &green);
    canvas.display_until_exit();
}
