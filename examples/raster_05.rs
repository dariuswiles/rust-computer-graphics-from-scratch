//! Implementation of pseudocode from chapter 9 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. This code uses perspective projection techniques to draw a wireframe cube.
//!
//! I am not affiliated with Gabriel or his book in any way.

use std::iter::Iterator;
use std::vec::Vec;
use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
use crate::vector_math::{Vector3};
#[allow(dead_code)]
mod vector_math;

const CANVAS_WIDTH: usize = 600;
const CANVAS_HEIGHT: usize = 600;
const VIEWPORT_WIDTH: f64 = 1.0;
const VIEWPORT_HEIGHT: f64 = 1.0;
const DISTANCE_FROM_CAMERA_TO_VIEWPORT: f64 = 1.0;


/// Translates a point on the `viewport` in viewport coordinates, e.g., -0.5 to 0.5, to the
/// corresponding point on the `canvas` in canvas coordinates, e.g., 0 to 600. The result is left
/// as a pair of `f64` values because further math will be performed, so converting to `i32`s is
/// premature.
fn viewport_to_canvas(x: f64, y: f64) -> Point {
    Point::new(x * CANVAS_WIDTH as f64 / VIEWPORT_WIDTH,
               y * CANVAS_HEIGHT as f64 / VIEWPORT_HEIGHT)
}


/// Translates a point in 3D space to the corresponding point on the `viewport`.
fn project_vertex(v: &Vector3) -> Point {
    viewport_to_canvas(v.x * DISTANCE_FROM_CAMERA_TO_VIEWPORT / v.z,
                                 v.y * DISTANCE_FROM_CAMERA_TO_VIEWPORT / v.z)
}


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


/// A 2D point.
#[derive(Clone, Copy, Debug)]
struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    #[allow(dead_code)]
    fn new(x: f64, y: f64) -> Self {
        Self {x: x, y: y}
    }

    #[allow(dead_code)]
    fn from_ints(x: i32, y: i32) -> Self {
        Self {x: x as f64, y: y as f64}
    }
}


/// Draws a straight line between `p0.x`, `p0.y` and `p1.x`, `p1.y` in the given color (inclusive).
/// Coordinates outside the canvas coordinates do not result in an error and any part of the line
/// that is within the canvas will be drawn.
///
/// # Examples
/// ```
/// use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
/// #[allow(unused_imports)]
/// use crate::vector_math::*;
/// mod vector_math;
///
/// const CANVAS_WIDTH: usize = 600;
/// const CANVAS_HEIGHT: usize = 600;
///
/// let mut canvas = Canvas::new("Raster 01 (from chapter 6)", CANVAS_WIDTH, CANVAS_HEIGHT);
/// draw_line(&mut canvas, -400,0, 400,0, &Rgb::from_ints(255,0,255));
/// canvas.display_until_exit();
/// ```
fn draw_line(canvas: &mut Canvas, p0: &Point, p1: &Point, color: &Rgb) {
    let x_length = (p1.x - p0.x).abs();
    let y_length = (p1.y - p0.y).abs();

    if x_length > y_length {
        for p in interpolate(p0.x, p0.y, p1.x, p1.y) {
            canvas.put_pixel(p.0.round() as i32, p.1.round() as i32, &color);
        }
    } else {
        for p in interpolate(p0.y, p0.x, p1.y, p1.x) {
            canvas.put_pixel(p.1.round() as i32, p.0.round() as i32, &color);
        }
    }
}


/// Creates a window and draws a cube with perspective projection using rasterization techniques.
fn main() {

    let mut canvas = Canvas::new("Raster 05 (from chapter 9)", CANVAS_WIDTH, CANVAS_HEIGHT);

    let blue = Rgb::from_ints(0,0,255);
    let green = Rgb::from_ints(0,255,0);
    let red = Rgb::from_ints(255,0,0);
    let white = Rgb::from_ints(255,255,255);

    canvas.clear_canvas(&white);

    // The four points forming the front face of the cube.
    // Note - these points are taken from the book's JavaScript demo as the code in the book is
    //        wrong.
    let af = Vector3::new(-2.0, -0.5, 5.0);
    let bf = Vector3::new(-2.0,  0.5, 5.0);
    let cf = Vector3::new(-1.0,  0.5, 5.0);
    let df = Vector3::new(-1.0, -0.5, 5.0);

    // The four points forming the back face of the cube.
    // Note - these points are taken from the book's JavaScript demo as the code in the book is
    //        wrong.
    let ab = Vector3::new(-2.0, -0.5, 6.0);
    let bb = Vector3::new(-2.0,  0.5, 6.0);
    let cb = Vector3::new(-1.0,  0.5, 6.0);
    let db = Vector3::new(-1.0, -0.5, 6.0);


    // The front face
    draw_line(&mut canvas, &project_vertex(&af), &project_vertex(&bf), &blue);
    draw_line(&mut canvas, &project_vertex(&bf), &project_vertex(&cf), &blue);
    draw_line(&mut canvas, &project_vertex(&cf), &project_vertex(&df), &blue);
    draw_line(&mut canvas, &project_vertex(&df), &project_vertex(&af), &blue);

    // The back face
    draw_line(&mut canvas, &project_vertex(&ab), &project_vertex(&bb), &red);
    draw_line(&mut canvas, &project_vertex(&bb), &project_vertex(&cb), &red);
    draw_line(&mut canvas, &project_vertex(&cb), &project_vertex(&db), &red);
    draw_line(&mut canvas, &project_vertex(&db), &project_vertex(&ab), &red);

    // The front-to-back edges
    draw_line(&mut canvas, &project_vertex(&af), &project_vertex(&ab), &green);
    draw_line(&mut canvas, &project_vertex(&bf), &project_vertex(&bb), &green);
    draw_line(&mut canvas, &project_vertex(&cf), &project_vertex(&cb), &green);
    draw_line(&mut canvas, &project_vertex(&df), &project_vertex(&db), &green);

    canvas.display_until_exit();
}
