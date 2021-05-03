//! Implementation of pseudocode from chapter 2 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/) book. I am not
//! affiliated with Gabriel or his book in any way.

use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
mod vector_math;

const WIDTH: usize = 1000;
const HEIGHT: usize = 600;
const DISTANCE_FROM_CAMERA_TO_VIEWPORT: f32 = 1.0;


//     let purple = Rgb {red: 255, green: 0, blue: 255};

/// Translates a point on the 2D canvas, expressed as `x` and `y` parameters, to a `Vector3` that goes from the
/// camera to that point.
fn canvas_to_viewport(x: f32, y: f32) -> Vector3 {
    Vector3::new(x * ??? / WIDTH, y * ??? / HEIGHT, DISTANCE_FROM_CAMERA_TO_VIEWPORT)
}



fn main() {
    let mut canvas = Canvas::new("Raytracer 01 (from chapter 2)", WIDTH, HEIGHT);

    // Define the origin
    let origin = Vector3::new(0, 0, 0);

    for x in -WIDTH/2 to WIDTH/2 {
        for x in -HEIGHT/2 to HEIGHT/2 {
            let direction = canvas_to_viewport(x, y);
            let color = trace_ray(origin, direction, 1, f32::INFINITY);

            canvas.put_pixel(x, y, &color);
        }
    }

    my_canvas.display_until_exit();
}


