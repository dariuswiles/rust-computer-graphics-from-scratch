//! Implementation of pseudocode from chapter 6 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. This is code for the last section, where interpolation functionality is moved into a
//! separate function.
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
fn interpolate(i0: i32, d0: f64, i1: i32, d1: f64) -> Vec<(i32, f64)> {
    let mut values = vec![];

    if i0 == i1 {
        values.push((i0, d0));
        return values;
    }

    let range: Vec<i32>;
    let a;

    if i0 < i1 {
        range = (i0..=i1).into_iter().collect();
        a = (d1 - d0) / (i1 - i0) as f64;
    } else {
        range = (i1..=i0).rev().into_iter().collect();
        a = (d1 - d0) / (i0 - i1) as f64;
    }

    let mut d = d0;

    for i in range {
        values.push((i, d));
        d = d + a;
    }

    values
}


/// Draws a straight line between `x_0`, `y_0` and `x_1`, `y_1` in the given color (inclusive).
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
fn draw_line(canvas: &mut Canvas, x0: i32, y0: i32, x1: i32, y1: i32, color: &Rgb) {

    let x_length = (x1 - x0).abs();
    let y_length = (y1 - y0).abs();

    if x_length > y_length {
        for (x, y) in interpolate(x0, y0 as f64, x1, y1 as f64) {
            canvas.put_pixel(x, y as i32, &color);
        }
    } else {
        for (y, x) in interpolate(y0, x0 as f64, y1, x1 as f64) {
            canvas.put_pixel(x as i32, y, &color);
        }
    }
}


/// Creates a window and draws some test lines using rasterization techniques.
fn main() {

    let mut canvas = Canvas::new("Raster 02 (from chapter 6)", CANVAS_WIDTH, CANVAS_HEIGHT);

    // Test data
    draw_line(&mut canvas, -400,0, 400,0, &Rgb::from_ints(255,0,0));
    draw_line(&mut canvas, 0,400, 0,-400, &Rgb::from_ints(255,0,0));

    draw_line(&mut canvas, -90,10, 10,110, &Rgb::from_ints(0,255,0));
    draw_line(&mut canvas, 10,110, 110,10, &Rgb::from_ints(0,255,0));
    draw_line(&mut canvas, 110,10, 10,-90, &Rgb::from_ints(0,255,0));
    draw_line(&mut canvas, 10,-90, -90,10, &Rgb::from_ints(0,255,0));

    draw_line(&mut canvas, 10,-210, -190,-10, &Rgb::from_ints(130,130,255));
    draw_line(&mut canvas, -190,-10, 10,190, &Rgb::from_ints(130,130,255));
    draw_line(&mut canvas, 10,190, 210,-10, &Rgb::from_ints(130,130,255));
    draw_line(&mut canvas, 210,-10, 10,-210, &Rgb::from_ints(130,130,255));

    canvas.display_until_exit();
}
