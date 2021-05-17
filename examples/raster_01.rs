//! Implementation of pseudocode from chapter 6 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. This code implements the first sections of that chapter, where the line drawing is
//! performed in a monolithic function.
//!
//! I am not affiliated with Gabriel or his book in any way.

use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
#[allow(unused_imports)]
use crate::vector_math::*;
mod vector_math;

const CANVAS_WIDTH: usize = 600;
const CANVAS_HEIGHT: usize = 600;


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

fn draw_line(canvas: &mut Canvas, x_0: i32, y_0: i32, x_1: i32, y_1: i32, color: &Rgb) {
    let x_length = (x_1 - x_0).abs();
    let y_length = (y_1 - y_0).abs();

    if x_length > y_length {
        let y_start;
        let y_delta;

        if x_0 < x_1 {
            y_start = y_0 as f64;
            y_delta = (y_1 - y_0) as f64 / x_length as f64;
        } else {
            y_start = y_1 as f64;
            y_delta = (y_0 - y_1) as f64 / x_length as f64;
        }

        let mut y = y_start;

        for x in i32::min(x_0, x_1) .. i32::max(x_0, x_1) + 1 {
            canvas.put_pixel(x, y as i32, &color);
            y += y_delta;
        }
    } else {
        let x_start;
        let x_delta;

        if y_0 < y_1 {
            x_start = x_0 as f64;
            x_delta = (x_1 - x_0) as f64 / y_length as f64;
        } else {
            x_start = x_1 as f64;
            x_delta = (x_0 - x_1) as f64 / y_length as f64;
        }

        let mut x = x_start;

        for y in i32::min(y_0, y_1) .. i32::max(y_0, y_1) + 1 {
            canvas.put_pixel(x as i32, y, &color);
            x += x_delta;
        }
    }
}


/// Creates a window and a scene of entities to render. Loops over every pixel in the window canvas to determine the
/// correct color based on the scene's entities, and then displays the result in the window.
fn main() {
    let mut canvas = Canvas::new("Raster 01 (from chapter 6)", CANVAS_WIDTH, CANVAS_HEIGHT);

    // Test data
    draw_line(&mut canvas, -400,0, 400,0, &Rgb::from_ints(255,0,0));
    draw_line(&mut canvas, 0,400, 0,-400, &Rgb::from_ints(255,0,0));

    draw_line(&mut canvas, -90,10, 10,110, &Rgb::from_ints(0,255,0));
    draw_line(&mut canvas, 10,110, 110,10, &Rgb::from_ints(0,255,0));
    draw_line(&mut canvas, 110,10, 10,-90, &Rgb::from_ints(0,255,0));
    draw_line(&mut canvas, 10,-90, -90,10, &Rgb::from_ints(0,255,0));

    draw_line(&mut canvas, -10,-210, -190,-10, &Rgb::from_ints(0,0,255));
    draw_line(&mut canvas, -190,-10, -10,190, &Rgb::from_ints(0,0,255));
    draw_line(&mut canvas, -10,190, 190,-10, &Rgb::from_ints(0,0,255));
    draw_line(&mut canvas, 190,-10, -10,-210, &Rgb::from_ints(0,0,255));

    canvas.display_until_exit();
}
