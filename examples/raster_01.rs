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


/// A convenience structure to hold a point in 2D space.
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
        let y_start;
        let y_delta;

        if p0.x < p1.x {
            y_start = p0.y as f64;
            y_delta = (p1.y - p0.y) as f64 / x_length as f64;
        } else {
            y_start = p1.y as f64;
            y_delta = (p0.y - p1.y) as f64 / x_length as f64;
        }

        let mut y = y_start;

        for x in f64::min(p0.x, p1.x) as i32 .. f64::max(p0.x, p1.x) as i32 + 1 {
            canvas.put_pixel(x as i32, y as i32, &color);
            y += y_delta;
        }
    } else {
        let x_start;
        let x_delta;

        if p0.y < p1.y {
            x_start = p0.x as f64;
            x_delta = (p1.x - p0.x) as f64 / y_length as f64;
        } else {
            x_start = p1.x as f64;
            x_delta = (p0.x - p1.x) as f64 / y_length as f64;
        }

        let mut x = x_start;

        for y in f64::min(p0.y, p1.y) as i32 .. f64::max(p0.y, p1.y) as i32 + 1 {
            canvas.put_pixel(x as i32, y, &color);
            x += x_delta;
        }
    }
}


/// Creates a window and a scene of entities to render. Loops over every pixel in the window canvas to determine the
/// correct color based on the scene's entities, and then displays the result in the window.
fn main() {
    let mut canvas = Canvas::new("Raster 01 (from chapter 6)", CANVAS_WIDTH, CANVAS_HEIGHT);

    let red = Rgb::from_ints(255,0,0);
    let blue = Rgb::from_ints(255,0,0);
    let green = Rgb::from_ints(0,0,255);

    // Test data
    draw_line(&mut canvas, &Point::from_ints(-400, 0), &Point::from_ints(400, 0), &red);
    draw_line(&mut canvas, &Point::from_ints(0, 400), &Point::from_ints(0, -400), &red);

    draw_line(&mut canvas, &Point::from_ints(-90, 10), &Point::from_ints(10,110), &blue);
    draw_line(&mut canvas, &Point::from_ints(10, 110), &Point::from_ints(110,10), &blue);
    draw_line(&mut canvas, &Point::from_ints(110, 10), &Point::from_ints(10,-90), &blue);
    draw_line(&mut canvas, &Point::from_ints(10, -90), &Point::from_ints(-90,10), &blue);

    draw_line(&mut canvas, &Point::from_ints(-10, -210), &Point::from_ints(-190,-10), &green);
    draw_line(&mut canvas, &Point::from_ints(-190, -10), &Point::from_ints(-10,190), &green);
    draw_line(&mut canvas, &Point::from_ints(-10, 190), &Point::from_ints(190,-10), &green);
    draw_line(&mut canvas, &Point::from_ints(190, -10), &Point::from_ints(-10,-210), &green);

    canvas.display_until_exit();
}
