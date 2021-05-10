//! Extends the implementation based on the second part of chapter 4 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/) book with the
//! ability to set an arbitrary position and direction for the camera, as discussed in the first part of chapter 5.
//!
//! I am not affiliated with Gabriel or his book in any way.

use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
#[allow(unused_imports)]
use crate::vector_math::*;
mod vector_math;

const CANVAS_WIDTH: usize = 600;
const CANVAS_HEIGHT: usize = 600;
const VIEWPORT_WIDTH: f64 = 1.0;
const VIEWPORT_HEIGHT: f64 = 1.0;
const DISTANCE_FROM_CAMERA_TO_VIEWPORT: f64 = 1.0;
const RECURSION_LIMIT: i32 = 3;
const TRACE_EPSILON: f64 = f64::EPSILON*1000000.0;

#[derive(Clone, Debug)]
struct Scene {
    viewport_width: f64,
    viewport_height: f64,
    background_color: Rgb,
    entities: Vec<SceneEntity>,
}

#[derive(Clone, Copy, Debug)]
enum SceneEntity {
    Light(LightType),
    Sphere(SphereEntity),
}

#[derive(Clone, Copy, Debug)]
enum LightType {
    Ambient(AmbientLightEntity),
    Directional(DirectionalLightEntity),
    Point(PointLightEntity),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AmbientLightEntity {
    intensity: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DirectionalLightEntity {
    intensity: f64,
    direction: Vector3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PointLightEntity {
    intensity: f64,
    position: Vector3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SphereEntity {
    center: Vector3,
    radius: f64,
    color: Rgb,
    specular: i32,
    reflective: f64,
}


/// Translates a point on the 2D canvas, passed in the `x` and `y` parameters, to a `Vector3` that goes from the
/// camera to that point.
fn canvas_to_viewport(x: f64, y: f64) -> Vector3 {
    Vector3::new(
        x * VIEWPORT_WIDTH / CANVAS_WIDTH as f64,
        y * VIEWPORT_HEIGHT / CANVAS_HEIGHT as f64,
        DISTANCE_FROM_CAMERA_TO_VIEWPORT
    )
}


/// Reflects a `ray` by bouncing it off of an assumed flat surface defined by the `normal` passed.
fn reflect_ray(ray: &Vector3, normal: &Vector3) -> Vector3 {
    let normal_dot_ray = normal.dot(&ray);
    return normal.multiply_by(2.0 * normal_dot_ray)
                 .subtract(&ray)
}


/// Given the `position` and `normal` of a surface, loops through all the lights in the scene and determines the
/// total intensity of them at `position`. All lights are white for simplicity, so only a single intensity value is
/// returned (rather individual RGB values).
fn compute_lighting(
    position: &Vector3,
    normal: &Vector3,
    towards_view: &Vector3,
    specular: i32,
    scene: &Scene) -> f64 {

    let mut i = 0.0;

    for scene_ent in scene.entities.iter() {

        if let SceneEntity::Light(light_type) = scene_ent {
            if let LightType::Ambient(am_light) = light_type {
                i += am_light.intensity;
            } else {
                let mut light_vector = Vector3::new(0.0, 0.0, 0.0);
                let mut light_intensity = 0.0;
                let mut t_max = 0.0;

                if let LightType::Point(point_light) = light_type {
                    light_vector = point_light.position.subtract(position);
                    light_intensity = point_light.intensity;
                    t_max = 1.0;
                } else if let LightType::Directional(dir_light) = light_type {
                    light_vector = dir_light.direction;
                    light_intensity = dir_light.intensity;
                    t_max = f64::INFINITY;
                }

                // Shadow check - do not add light from this source if there is a sphere intersecting it
                let (shadow_sphere, _shadow_t) = closest_intersection(&position, &light_vector, 0.001, t_max, &scene);
                if shadow_sphere != Option::None{
                    continue;
                }

                // Diffuse lighting calculation
                let normal_dot_light = normal.dot(&light_vector);

                // `normal_dot_light` is negative if the light is coming from behind the surface, and we ignore such
                // cases as they would incorrectly reduce light intensity.
                if normal_dot_light > 0.0 {
                    i += light_intensity * normal_dot_light/(normal.length() * light_vector.length())
                }


                // Specular lighting calculation
                if specular != -1 {

                    let reflection = reflect_ray(&light_vector, &normal);

                    let reflection_dot_towards_view = reflection.dot(towards_view);
                    if reflection_dot_towards_view > 0.0 {
                        i += light_intensity * f64::powi(reflection_dot_towards_view /
                                (reflection.length() * towards_view.length()), specular);
                    }
                }
            }
        }
    }

    return i;
}


/// Returns the handle of the closest sphere and the distance of its closest intersection from the `origin` provided.
/// The intersection is determined by extending the `direction` vector provided from this `origin`. Only intersections
/// within the range `t_min` and `t_max`, which are measured in world units, are considered. If no sphere intersects,
/// the sphere handle returned is `Option::None`, and the distance of closest intersection is `f64::INFINITY`.
fn closest_intersection<'a>(origin: &Vector3,
                        direction: &Vector3,
                        t_min: f64,
                        t_max: f64,
                        scene: &'a Scene)
                        -> (Option<&'a SphereEntity>, f64) {
    let mut closest_t = f64::INFINITY;
    let mut closest_sphere = Option::<&SphereEntity>::None;

    for scene_ent in scene.entities.iter() {
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

    return (closest_sphere, closest_t);
}

/// Returns the color of the closest sphere by extending the `direction` vector provided. Only considers spheres within
/// the range `t_min` and `t_max`, which are measured in world units. If no sphere intersects `direction`, the
/// `background_color` specified in the `scene` passed is returned.
fn trace_ray(origin: &Vector3,
             direction: &Vector3,
             t_min: f64,
             t_max: f64,
             recursion_depth: i32,
             scene: &Scene,)
             -> Rgb {
    let (closest_sphere, closest_t) = closest_intersection(&origin, &direction, t_min, t_max, &scene);

    if let Some(s) = closest_sphere {
        let position = origin.add(&direction.multiply_by(closest_t));  // Compute intersection
        let normal = position.subtract(&s.center);   // Compute normal of sphere at this intersection
        let normal_norm = normal.normalize().unwrap();  // Panics if light is in same location as sphere surface
        let intensity = compute_lighting(&position, &normal_norm, &direction.multiply_by(-1.0), s.specular, &scene);

        let local_color = s.color.multiply_by(intensity);

        // If we reach the recursion limit, or the sphere is not reflective, return the local color
        if (s.reflective <= 0.0) | (recursion_depth <= 0) {
            return local_color;
        }

        // Compute the reflected color
        let reflected_ray = reflect_ray(&direction.multiply_by(-1.0), &normal_norm);

        let reflected_color = trace_ray(&position, &reflected_ray, TRACE_EPSILON, f64::INFINITY, recursion_depth - 1, &scene);

        return local_color.multiply_by(1.0 - s.reflective).add(&(&reflected_color).multiply_by(s.reflective));


    } else {
        return scene.background_color;
    }
}


/// Determines if a line drawn from `origin` along `direction` intersects sphere `s`. If so, the distances from the
/// origin at which the line intersects the surface of the sphere are returned as a tuple. If the line only intersects
/// once, the same distance is returned twice in the tuple. If the line does not intersect at all, a tuple with two
/// elements set to positive infinity is returned.
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


/// Creates a scene that includes viewport width and height (expressed in world space coordinates), a default
/// background color to be used if no other pixel value is set, and a vector of entity objects to display.
fn create_scene() -> Scene {
    Scene {
        viewport_width: VIEWPORT_WIDTH,
        viewport_height: VIEWPORT_HEIGHT,
        background_color: Rgb::from_ints(0, 0, 0),     // Black
//         background_color: Rgb::from_ints(red: 255, green: 255, blue: 255),     // White
        entities: vec!(
            SceneEntity::Sphere(SphereEntity{
                center: Vector3::new(0.0, -1.0, 3.0),
                radius: 1.0,
                color: Rgb::from_ints(255, 0, 0),    // Red
                specular: 500,  // Shiny
                reflective: 0.2,   // A bit reflective
            }),
            SceneEntity::Sphere(SphereEntity{
                center: Vector3::new(2.0, 0.0, 4.0),
                radius: 1.0,
                color: Rgb::from_ints(0, 0, 255),    // Blue
                specular: 500,  // Shiny
                reflective: 0.3,   // A bit more reflective
            }),
            SceneEntity::Sphere(SphereEntity{
                center: Vector3::new(-2.0, 0.0, 4.0),
                radius: 1.0,
                color: Rgb::from_ints(0, 255, 0),    // Green
                specular: 10,  // Somewhat shiny
                reflective: 0.4, // Even more reflective
            }),
            SceneEntity::Sphere(SphereEntity{
                center: Vector3::new(0.0, -5001.0, 0.0),
                radius: 5000.0,
                color: Rgb::from_ints(255, 255, 0),  // Yellow
                specular: 1000,  // Very shiny
                reflective: 0.5, // Half reflective
            }),
            SceneEntity::Light(LightType::Ambient(AmbientLightEntity{
                intensity: 0.2,
            })),
            SceneEntity::Light(LightType::Point(PointLightEntity{
                intensity: 0.6,
                position: Vector3::new(2.0, 1.0, 0.0),
            })),
            SceneEntity::Light(LightType::Directional(DirectionalLightEntity{
                intensity: 0.2,
                direction: Vector3::new(1.0, 4.0, 4.0),
            })),
        ),
    }
}


/// Creates a window and a scene of entities to render. Loops over every pixel in the window canvas to determine the
/// correct color based on the scene's entities, and then displays the result in the window.
fn main() {

    let mut canvas = Canvas::new("Chapter 5 Arbitrary camera position", CANVAS_WIDTH, CANVAS_HEIGHT);

    let scene = create_scene();

    // Define the position and orientation of the camera
    let camera_position = Vector3::new(0.0, 0.0, 0.0);

    let camera_rotation = Matrix3x3::new(
                            0.7071, 0.0, -0.7071,
                            0.0,    1.0,  0.0,
                            0.7071, 0.0,  0.7071);


    let cw = CANVAS_WIDTH as i32;
    let ch = CANVAS_HEIGHT as i32;

    for x in -cw/2 .. cw/2 {
        for y in -ch/2 .. ch/2 {

// TODO Implement matrix multiplication so following calculation can be performed.
//             let direction = camera_rotation./* matrix multiply */(&canvas_to_viewport(x as f64, y as f64));
//             let color = trace_ray(&camera_position, &direction, 1.0, f64::INFINITY, RECURSION_LIMIT, &scene);

//             canvas.put_pixel(x, y, &color.clamp());
        }
    }

    canvas.display_until_exit();
}



#[cfg(test)]
fn test_reflect_ray_1() {
    let v1 = Vector3::new(3.0, 7.0, 11.0);
    let v2 = Vector3::new(13.0, 1.0, 5.0);
    let rr_result = reflect_ray(&v1, &v2);
    assert!(rr_result == Vector3::new(2623.0, 195.0, 999.0));
}

#[cfg(test)]
fn test_reflect_ray_2() {
    let v3 = Vector3::new(3.3, 7.7, 11.1);
    let v4 = Vector3::new(13.3, 1.1, 5.5);
    let rr_result = reflect_ray(&v3, &v4);
    assert!(rr_result == Vector3::new(3013.406, 241.80202, 1236.41));
}
