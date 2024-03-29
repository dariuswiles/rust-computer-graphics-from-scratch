//! Implementation of pseudocode from chapter 2 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. I am not affiliated with Gabriel or his book in any way.
//!
//! This example is the first step in implementing a basic ray tracer. It renders three spheres
//! against a white background.

use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
use crate::vector_math::Vector3;
#[allow(dead_code)]
mod vector_math;

const CANVAS_WIDTH: usize = 600;
const CANVAS_HEIGHT: usize = 600;
const VIEWPORT_WIDTH: f64 = 1.0;
const VIEWPORT_HEIGHT: f64 = 1.0;
const DISTANCE_FROM_CAMERA_TO_VIEWPORT: f64 = 1.0;

#[derive(Clone, Debug)]
struct Scene {
    background_color: Rgb,
    entities: Vec<SceneEntity>,
}

#[derive(Clone, Copy, Debug)]
enum SceneEntity {
    Sphere(SphereEntity),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SphereEntity {
    center: Vector3,
    radius: f64,
    color: Rgb,
}

/// Translates a point on the 2D canvas, passed in the `x` and `y` parameters, to a `Vector3` that
/// goes from the camera to that point.
fn canvas_to_viewport(x: f64, y: f64) -> Vector3 {
    Vector3::new(
        x * VIEWPORT_WIDTH / CANVAS_WIDTH as f64,
        y * VIEWPORT_HEIGHT / CANVAS_HEIGHT as f64,
        DISTANCE_FROM_CAMERA_TO_VIEWPORT
    )
}


/// Returns the color of the closest sphere by extending the `direction` vector provided. Only
/// considers spheres within the range `t_min` and `t_max`, which are measured in world units. If
/// no sphere intersects `direction`, the `background_color` specified in the `scene` passed is
/// returned.
fn trace_ray(origin: &Vector3, direction: &Vector3, t_min: f64, t_max: f64, scene: &Scene) -> Rgb {
    let mut closest_t = f64::INFINITY;
    let mut closest_sphere = Option::<&SphereEntity>::None;

    for scene_ent in scene.entities.iter() {
        // "If" always true because `SceneEntity` only defines `Sphere` in this example.
        #[allow(irrefutable_let_patterns)]
        if let SceneEntity::Sphere(s) = scene_ent {
            let (t1, t2) = intersect_ray_sphere(origin, direction, s);
            if (t_min < t1) & (t1 < t_max) & (t1 < closest_t) {
                closest_t = t1;
                closest_sphere = Some(&s);
            }
            if (t_min < t2) & (t2 < t_max) & (t2 < closest_t) {
                closest_t = t2;
                closest_sphere = Some(&s);
            }
        }
    }

    if let Some(s) = closest_sphere {
        return s.color;
    } else {
        return scene.background_color;
    }
}

/// Determines if a line drawn from `origin` along `direction` intersects sphere `s`. If so, the
/// distances from the origin at which the line intersects the surface of the sphere are returned
/// as a tuple. If the line only intersects once, the same distance is returned twice in the tuple.
/// If the line does not intersect at all, a tuple with two elements set to positive infinity is
/// returned.
fn intersect_ray_sphere(origin: &Vector3, direction: &Vector3, s: &SphereEntity) -> (f64, f64) {
    let r = s.radius;
    let center_origin = origin.subtract(&s.center);

    let a = direction.dot(&direction);
    let b = 2.0 * center_origin.dot(&direction);
    let c = center_origin.dot(&center_origin) - r*r;

    let discriminant = b*b - 4.0*a*c;
    if discriminant < 0.0 {
        return (f64::INFINITY, f64::INFINITY);
    }

    let t1 = (-b + f64::sqrt(discriminant)) / (2.0*a);
    let t2 = (-b - f64::sqrt(discriminant)) / (2.0*a);
    return (t1, t2);
}


/// Creates a scene that includes viewport width and height (expressed in world space coordinates),
/// a default background color to be used if no other pixel value is set, and a vector of entity
/// objects to display.
fn create_scene() -> Scene {
    Scene {
        background_color: Rgb::from_ints(255, 255, 255),     // White
        entities: vec!(
            SceneEntity::Sphere(SphereEntity{
                center: Vector3::new(0.0, -1.0, 3.0),
                radius: 1.0,
                color: Rgb::from_ints(255, 0, 0),    // Red
            }),
            SceneEntity::Sphere(SphereEntity{
                center: Vector3::new(2.0, 0.0, 4.0),
                radius: 1.0,
                color: Rgb::from_ints(0, 0, 255),    // Blue
            }),
            SceneEntity::Sphere(SphereEntity{
                center: Vector3::new(-2.0, 0.0, 4.0),
                radius: 1.0,
                color: Rgb::from_ints(0, 255, 0),    // Green
            }),
        ),
    }
}


/// Creates a window and a scene of entities to render. Loops over every pixel in the window canvas
/// to determine the correct color based on the scene's entities, and then displays the result in
/// the window.
fn main() {
    let mut canvas = Canvas::new("Raytracer 01 (from chapter 2)", CANVAS_WIDTH, CANVAS_HEIGHT);

    let scene = create_scene();

    // Define the origin
    let origin = Vector3::new(0.0, 0.0, 0.0);

    let cw = CANVAS_WIDTH as i32;
    let ch = CANVAS_HEIGHT as i32;

    for x in -cw/2 .. cw/2 {
        for y in -ch/2 .. ch/2 {
            let direction = canvas_to_viewport(x as f64, y as f64);
            let color = trace_ray(&origin, &direction, 1.0, f64::INFINITY, &scene);

            canvas.put_pixel(x, y, &color.clamp());
        }
    }

    canvas.display_until_exit();
}
