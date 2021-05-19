//! Implementation of pseudocode from chapter 7 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. This is code displays wireframe and filled triangles.
//!
//! I am not affiliated with Gabriel or his book in any way.

use std::mem;
use std::iter::Iterator;
use std::vec::Vec;
use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
#[allow(unused_imports)]
use crate::vector_math::*;
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


/// Draws a wireframe triangle defined by the three points passed in the color passed.
fn draw_wireframe_triangle (canvas: &mut Canvas, p0: &Point, p1: &Point, p2: &Point, color: &Rgb) {
    draw_line(canvas, p0, p1, color);
    draw_line(canvas, p1, p2, color);
    draw_line(canvas, p2, p0, color);
}


/// Draws a filled triangle defined by the three points passed in the color passed.
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
    // `(f64, f64)`, representing `(y, x)` coordinates.
    let x01 = interpolate(p0.y, p0.x, p1.y, p1.x);
    let x12 = interpolate(p1.y, p1.x, p2.y, p2.x);
    let x02 = interpolate(p0.y, p0.x, p2.y, p2.x);

    // Concatenate `x01` and `x12`, but remove the value at the end of `x01` as it is repeated as
    // the first value of `x12`
    let x012 = [&x01[..x01.len()-1], &x12[..]].concat();

    let x_left;
    let x_right;
    let m = x02.len() / 2;

    // Look at the middle row of the triangle to determine whether `x02` or `x012` represents the
    // left side of the triangle.
    if x02[m].1 < x012[m].1 {   // Note that field `0` holds `x` coords, and `1` holds `y`.
        x_left = x02;
        x_right = x012;
    } else {
        x_left = x012;
        x_right = x02;
    }

    // For every line, draw a row between the left and right sides of the triangle.
    for y in corner0.y.round() as i32 .. corner2.y.round() as i32 {
        let x_start = x_left.get((y - corner0.y.round() as i32) as usize).unwrap().1.round() as i32;
        let x_end = x_right.get((y - corner0.y.round() as i32) as usize).unwrap().1.round() as i32;

        for x in x_start .. x_end {
            canvas.put_pixel(x, y, color);
        }
    }
}


/// Creates a window and draws some test lines using rasterization techniques.
fn main() {

    let mut canvas = Canvas::new("Raster 03 (from chapter 7)", CANVAS_WIDTH, CANVAS_HEIGHT);

    let white = Rgb::from_ints(255,255,255);
    let black = Rgb::from_ints(0,0,0);
    let green = Rgb::from_ints(0,255,0);

    let p0 = Point::from_ints(-200, -250);
    let p1 = Point::from_ints(200, 50);
    let p2 = Point::from_ints(20, 250);


    canvas.clear_canvas(&white);
    draw_filled_triangle (&mut canvas, &p0, &p1, &p2, &green);
    draw_wireframe_triangle (&mut canvas, &p0, &p1, &p2, &black);
    canvas.display_until_exit();
}
