//! Implementation of pseudocode from chapter 2 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/) book. I am not
//! affiliated with Gabriel or his book in any way.

use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
#[allow(unused_imports)]
use crate::vector_math::*;
mod vector_math;

const CANVAS_WIDTH: usize = 1000;
const CANVAS_HEIGHT: usize = 600;
const VIEWPORT_WIDTH: f32 = 1.0;
const VIEWPORT_HEIGHT: f32 = 1.0;
const DISTANCE_FROM_CAMERA_TO_VIEWPORT: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Sphere {
    center: Vector3,
    radius: f32,
    color: Rgb,
}




/// Translates a point on the 2D canvas, expressed as `x` and `y` parameters, to a `Vector3` that goes from the
/// camera to that point.
fn canvas_to_viewport(x: f32, y: f32) -> Vector3 {
    Vector3::new(
        x * VIEWPORT_WIDTH / CANVAS_WIDTH as f32,
        y * VIEWPORT_HEIGHT / CANVAS_HEIGHT as f32,
        DISTANCE_FROM_CAMERA_TO_VIEWPORT
    )
}


fn trace_ray(origin: Vector3, direction: Vector3, t_min: f32, t_max: f32) -> Rgb {
    let closest_t = f32::INFINITY;
    let closest_sphere = Option::<usize>::None;

    // TODO main body
    Rgb {red: 0, green: 0, blue: 0}
}


fn main() {
    let mut canvas = Canvas::new("Raytracer 01 (from chapter 2)", CANVAS_WIDTH, CANVAS_HEIGHT);

    // TODO Unsure where to create the scene
    let scene = vec!(Sphere{
        center: Vector3::new(0.0, 0.0, 3.0),
        radius: 1.0,
        color: Rgb{red: 255, green: 0, blue: 0},
    });


    // Define the origin
    let origin = Vector3::new(0.0, 0.0, 0.0);

    let cw = CANVAS_WIDTH as i32;
    let ch = CANVAS_HEIGHT as i32;

    for x in -cw/2 .. cw/2 {
        for y in -ch/2 .. ch/2 {
            let direction = canvas_to_viewport(x as f32, y as f32);
            let color = trace_ray(origin, direction, 1.0, f32::INFINITY);

            canvas.put_pixel(x, y, &color);
        }
    }

    canvas.display_until_exit();
}


