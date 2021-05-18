//! Implementation of pseudocode from chapter 7 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. This is code displays wireframe and filled triangles.
//!
//! I am not affiliated with Gabriel or his book in any way.

use std::vec::Vec;
use std::iter::Iterator;
use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
#[allow(unused_imports)]
use crate::vector_math::*;
mod vector_math;

const CANVAS_WIDTH: usize = 600;
const CANVAS_HEIGHT: usize = 600;


/// Iterates over a range of `i32`s, from `i0` to `i1` inclusive, interpolating over a dependent
/// range of `f64`s, from `d0` to `d1` inclusive.
///
/// Returns a vector of (i32, i64) tuples, where the former is the independent variable, and the
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
fn interpolate(i0: f64, d0: f64, i1: f64, d1: f64) -> Vec<Point> {
    let mut values = vec![];

    if i0 == i1 {
        values.push(Point::new(i0, d0));
        return values;
    }

    let range: Vec<i32>;
    let a;

    if i0 < i1 {
        range = (i0 as i32 ..= i1 as i32).into_iter().collect();
        a = (d1 - d0) / (i1 - i0);
    } else {
        range = (i1 as i32 ..= i0 as i32).rev().into_iter().collect();
        a = (d1 - d0) / (i0 - i1);
    }

    let mut d = d0;

    for i in range {
        values.push(Point::new(i as f64, d));
        d = d + a;
    }

    values
}


/// A 2D point.
struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    #[allow(dead_code)]
    fn new(x: f64, y: f64) -> Self {
        Self {x: x, y: y}
    }

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
            canvas.put_pixel(p.x as i32, p.y as i32, &color);
        }
    } else {
        for p in interpolate(p0.y, p0.x, p1.y, p1.x) {
            canvas.put_pixel(p.y as i32, p.x as i32, &color);
        }
    }
}


/// Draws a wireframe triangle defined by the three points passed.
fn draw_wireframe_triangle (canvas: &mut Canvas, p0: &Point, p1: &Point, p2: &Point, color: &Rgb) {
    draw_line(canvas, p0, p1, color);
    draw_line(canvas, p1, p2, color);
    draw_line(canvas, p2, p0, color);
}


/// Creates a window and draws some test lines using rasterization techniques.
fn main() {

    let mut canvas = Canvas::new("Raster 03 (from chapter 7)", CANVAS_WIDTH, CANVAS_HEIGHT);

    let white = Rgb::from_ints(255,255,255);
    let black = Rgb::from_ints(0,0,0);

    canvas.clear_canvas(&white);
    draw_wireframe_triangle (&mut canvas,
                             &Point::from_ints(-200, -250),
                             &Point::from_ints(200, 50),
                             &Point::from_ints(20, 250),
                             &black);

    canvas.display_until_exit();
}
