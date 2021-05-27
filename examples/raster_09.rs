//! Implementation of pseudocode from chapter 11 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. This code implements clipping.
//!
//! I am not affiliated with Gabriel or his book in any way.

use std::f64::consts::SQRT_2;
use std::iter::Iterator;
use std::vec::Vec;
use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
#[allow(unused_imports)]
use crate::vector_math::*;
mod vector_math;

const CANVAS_WIDTH: usize = 600;
const CANVAS_HEIGHT: usize = 600;
const VIEWPORT_WIDTH: f64 = 1.0;
const VIEWPORT_HEIGHT: f64 = 1.0;
const DISTANCE_FROM_CAMERA_TO_VIEWPORT: f64 = 1.0;


/// General struct to hold vertex and triangle data for any model shape.
struct Model<'a> {
    vertices: &'a [Vector3],
    triangles: &'a [Triangle],
    bounds_center: &'a Vector3,
    bounds_radius: f64,
}


/// An instance of a particular model shape. Includes the `model` to use and its position,
/// orientation and scale. A `transform` matrix is generated when the object is created that
/// performs the translation, orientation and scaling, so these computations only need to be
/// performed once per instance. A `ModelInstance` object should not be changed after creation
/// because the `transform` would no longer be correct. This restriction includes the `position`
/// field, preventing instances being moved.
struct ModelInstance<'a> {
    model: &'a Model<'a>,
    #[allow(dead_code)]
    position: Vector3,
    #[allow(dead_code)]
    orientation: Matrix4x4,
    #[allow(dead_code)]
    scale: f64,
    transform: Matrix4x4,
}

impl<'a> ModelInstance<'a> {
    /// Creates and returns a new `ModelInstance` object, and automatically generates and stores a
    /// `transform` matrix that performs the combination of the translation, orientation and
    /// scaling operations.
    fn new(model: &'a Model, position: Vector3, orientation: Matrix4x4, scale: f64) -> Self {
        let transform =
            Matrix4x4::new_translation_matrix(&position)
            .multiply_matrix4x4(
                &orientation.multiply_matrix4x4(
                    &Matrix4x4::new_scaling_matrix(scale)));

        Self {
            model: model,
            position: position,
            orientation: orientation,
            scale: scale,
            transform: transform,
        }
    }
}


/// A camera, consisting of a position in 3D space, and an orientation. The latter is expressed in
/// homogenous coordinates, i.e., a 4x4 matrix.
struct Camera {
    position: Vector3,
    orientation: Matrix4x4,
    clipping_planes: [Plane; 5]
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


/// A triangle, consisting of 3 vertex indices defining the position of its corners, and a color.
#[derive(Clone, Copy, Debug)]
struct Triangle {
    pub vertices: (usize, usize, usize),
    pub color: Rgb,
}

impl Triangle {
    #[allow(dead_code)]
    fn new(vertices: (usize, usize, usize), color: Rgb) -> Self {
        Self {vertices: vertices, color: color}
    }
}


/// A 2D plane, defined as a normal to the plane, and a distance from the origin. The normal should
/// be a unit vector, i.e., its length should be 1.
#[derive(Clone, Copy, Debug)]
struct Plane {
    pub normal: Vector3,
    pub distance: f64,
}


/// Translates a point on the `viewport` in viewport coordinates, e.g., -0.5 to 0.5, to the
/// corresponding point on the `canvas` in canvas coordinates, e.g., 0 to 600. The result is left
/// as a pair of `f64` values because further math will be performed, so converting to `i32`s is
/// premature.
fn viewport_to_canvas(x: f64, y: f64) -> Point {
    Point::new(x * CANVAS_WIDTH as f64 / VIEWPORT_WIDTH,
               y * CANVAS_HEIGHT as f64 / VIEWPORT_HEIGHT)
}


/// Translates a point in 3D space to the corresponding point on the `viewport`.
fn project_vertex(v: &Vector4) -> Point {
    viewport_to_canvas(v.x * DISTANCE_FROM_CAMERA_TO_VIEWPORT / v.z,
                       v.y * DISTANCE_FROM_CAMERA_TO_VIEWPORT / v.z)
}


/// Iterates over the range `i0` to `i1` inclusive, interpolating over a dependent range from `d0`
/// to `d1` inclusive. `i0` must be lower than or equal to `i1`, or the vector returned will be
/// empty.
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

    if i0 > i1 {
        return values;
    }

    let range = (i0.round() as i32 ..= i1.round() as i32).into_iter().collect::<Vec<_>>();
    let delta = (d1 - d0) / (i1 - i0);
    let mut d = d0;

    for i in range {
        values.push((i as f64, d));
        d = d + delta;
    }

    values
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
        // The line is more horizontal than vertical. Determine which point is more to the left,
        // so the line is drawn from left to right.
        let (left, right);
        if p0.x < p1.x {
            left = p0;
            right = p1;
        } else {
            left = p1;
            right = p0;
        }

        for p in interpolate(left.x, left.y, right.x, right.y) {
            canvas.put_pixel(p.0.round() as i32, p.1.round() as i32, &color);
        }
    } else {
        // The line is more vertical than horizontal. Determine which point is lower, so the line
        // is drawn from bottom to top.
        let (bottom, top);
        if p0.y < p1.y {
            bottom = p0;
            top = p1;
        } else {
            bottom = p1;
            top = p0;
        }
        for p in interpolate(bottom.y, bottom.x, top.y, top.x) {
            canvas.put_pixel(p.1.round() as i32, p.0.round() as i32, &color);
        }
    }
}


/// Renders the `ModelInstance` passed by iterating through the list of triangles and vertices
/// that it contains, using the `transform` provided to transform each vertex into camera space,
/// then calling `render_triangle` to draw the triangle on the 2D canvas.
fn render_instance(canvas: &mut Canvas, mi: &ModelInstance, transform: &Matrix4x4) {
    let mut projected = vec![];

    // The `vertices` are defined in the 3D world coordinates, so project each vertex onto the
    // viewport, resulting in a vector of 2D viewport `Point`s.
    for v in mi.model.vertices {
        let vertex_h = Vector4::new(v.x, v.y, v.z, 1.0);

        projected.push(project_vertex(&transform.multiply_vector(&vertex_h)));
    }

    // Render each triangle by passing coordinates of each corner as a 2D `Point` on the viewport.
    for t in mi.model.triangles {
        render_triangle(canvas, &t, &projected);
    }
}


// xxx
fn transform_and_clip(clipping_planes: &[Plane; 5], model: &Model, transform: &Matrix4x4)
    -> Option<Model> {

    // Apply `transform` to the center position of the bounding sphere and see if the instance is
    // completely outside any of the clipping planes. If so, return immediately.
    let transformed_center = transform.multiply_matrix4x4(model.bounds_center);

    for cp in clipping_planes {
        let distance = cp.normal.dot(&transformed_center) + cp.distance;
        if distance < -model.bounds_radius {
            return None;
        }
    }

    let mut vertices = Vec::new();

    // Apply modelview transform to each vertex in the model instance.
    for v in model.vertices {
        vertices.push(transform.multiply_matrix4x4(Vector4::new(&v.x, &v.y, &v.z, 0.0)));
    }

    // Loop through every clipping plane clipping all the model vertices each time. The output of
    // one interaction is used as input to the next to handle cases where a triangle intersects
    // multiply clipping planes.
    let mut triangles = model.triangles.clone();

    for cp in clipping_planes {
        let mut new_triangles = Vec::new();
        for t in triangles {
            clip_triangle(&t, cp, &mut new_triangles, &vertices);
        }

        triangles = new_triangles;
    }

    Some(Model::new {
        vertices: &vertices,
        triangles: &triangles,
        bounds_center: transformed_center,
        bounds_radius: model.bounds_radius,
    })
}










/// Renders every model instance in the `instances` array passed. The `Camera` object passed is
/// used to create a world view to camera view transform by reversing its transpose and rotation.
/// This is combined with the transform included with each model instance to create a transform
/// that converts from instance space to camera space. This transform is generated once per
/// instance and passed as input to the `render_instance` function.
fn render_scene(canvas: &mut Canvas, camera: &Camera, instances: &[ModelInstance]) {
    let camera_matrix =
        camera.orientation
            .transpose()
            .multiply_matrix4x4(&Matrix4x4::new_translation_matrix(
                &camera.position.multiply_by(-1.0),
            ));

    for mi in instances {
        let transform = camera_matrix.multiply_matrix4x4(&mi.transform);
        let clipped = transform_and_clip(&camera.clipping_planes, &mi.model, &transform);
        if 1 == 1 /* TODO fix this logic */ {
            render_instance(canvas, mi, &transform);
        }
    }
}


/// Render a wireframe triangle on the canvas, using the viewport coordinates of its corners.
fn render_triangle(canvas: &mut Canvas, triangle: &Triangle, projected: &Vec<Point>) {
    draw_wireframe_triangle(canvas,
                            &projected.get(triangle.vertices.0).unwrap(),
                            &projected.get(triangle.vertices.1).unwrap(),
                            &projected.get(triangle.vertices.2).unwrap(),
                            &triangle.color
    );
}


/// Draws a wireframe triangle defined by the three points passed in the color passed.
fn draw_wireframe_triangle (canvas: &mut Canvas, p0: &Point, p1: &Point, p2: &Point, color: &Rgb) {
    draw_line(canvas, p0, p1, color);
    draw_line(canvas, p1, p2, color);
    draw_line(canvas, p2, p0, color);
}


/// Returns the distance between a given `plane` and `vertex`.
fn signed_distance(plane: &Plane, vertex: &Vector3) -> f64 {
    vertex.x * plane.normal.x +
    vertex.y * plane.normal.y +
    vertex.z * plane.normal.z +
    plane.distance
}



fn intersection(zzz1: &Point, zzz2: &Point, plane: &Plane) {
    unimplemented!();
}





///
fn clip_triangle_against_plane(triangles: &Vec<Triangle>, plane: &Plane) -> Vec<Triangle> {
    let mut clipped_triangles = Vec::new();

    for t in triangles {
        clipped_triangles.append(&mut clip_triangle(&t, plane));
    }

    clipped_triangles
}


/// Returns a `Vec` containing the triangles needed to render the visible part of the `triangle`
/// passed. If `triangle` is completely within the clipping volume, it is returned as the only
/// element. If it is completely outside, the returned `Vec` is empty. If it is partially within,
/// new triangles that draw just the visible portion of `triangle` are returned.
fn clip_triangle(triangle: &Triangle, plane: &Plane) -> Vec<Triangle> {
    let d0 = signed_distance(plane, triangle.vertices.0);
    let d1 = signed_distance(plane, triangle.vertices.1);
    let d2 = signed_distance(plane, triangle.vertices.2);

    let mut positive = Vec::new();
    let mut negative = Vec::new();
    if d0 > 0.0 { positive.push(&triangle.vertices.0); }
        else { negative.push(&triangle.vertices.0); }
    if d1 > 0.0 { positive.push(&triangle.vertices.1); }
        else { negative.push(&triangle.vertices.1); }
    if d2 > 0.0 { positive.push(&triangle.vertices.2); }
        else { negative.push(&triangle.vertices.2); }

    match positive.len() {
        3 => return vec![*triangle],
        2 => {
                let a = positive.pop().unwrap();
                let b = positive.pop().unwrap();
                let c = negative.pop().unwrap();

                // TODO It may be better to pass first 2 parms as 1st.subtract(2nd)
                let a_prime = intersection(&a, &c, plane);  // TODO pseudcode awaiting implementation of `intersection`
                let b_prime = intersection(&b, &c, plane);  // TODO pseudcode awaiting implementation of `intersection`

                return vec![
                    Triangle::new((a, b, a_prime), Rgb::from_ints(0,0,0)),  // TODO interpolate color
                    Triangle::new((a_prime, b, b_prime), Rgb::from_ints(0,0,0)),    // TODO interpolate color
                ];
            },
        1 => {
                let a = positive.pop().unwrap();
                let b = negative.pop().unwrap();
                let c = negative.pop().unwrap();

                // TODO see above comments
                let b_prime = intersection(&a, &b, plane);  // TODO pseudcode awaiting implementation of `intersection`
                let c_prime = intersection(&a, &c, plane);  // TODO pseudcode awaiting implementation of `intersection`

                return vec![Triangle::new((a, b_prime, c_prime), Rgb::from_ints(0,0,0))];    // TODO interpolate color
            },
        0 => return vec![],
        _ => panic!("The number of triangle vertices within a clipping plane was not within 0-3")
    }
}




/// Creates a window, creates a scene containing two cubes and draws them with perspective
/// projection using rasterization techniques.
fn main() {

    let mut canvas = Canvas::new("Raster 09 (from chapter 11)", CANVAS_WIDTH, CANVAS_HEIGHT);

    let red = Rgb::from_ints(255,0,0);
    let green = Rgb::from_ints(0,255,0);
    let blue = Rgb::from_ints(0,0,255);
    let yellow = Rgb::from_ints(255,255,0);
    let purple = Rgb::from_ints(255,0,255);
    let cyan = Rgb::from_ints(0,255,255);
    let white = Rgb::from_ints(255,255,255);

    canvas.clear_canvas(&white);

    // Define vertices for the 8 corners of the cube to be rendered.
    let vertices = [
        Vector3::new( 1.0,  1.0,  1.0),  // Vertex 0
        Vector3::new(-1.0,  1.0,  1.0),  // Vertex 1
        Vector3::new(-1.0, -1.0,  1.0),  // Vertex 2
        Vector3::new( 1.0, -1.0,  1.0),  // Vertex 3
        Vector3::new( 1.0,  1.0, -1.0),  // Vertex 4
        Vector3::new(-1.0,  1.0, -1.0),  // Vertex 5
        Vector3::new(-1.0, -1.0, -1.0),  // Vertex 6
        Vector3::new( 1.0, -1.0, -1.0),  // Vertex 7
    ];

    // Define triangles with 3-tuples of indexes into the previously defined vertices.
    let triangles = [
        Triangle::new((0, 1, 2), red),
        Triangle::new((0, 2, 3), red),
        Triangle::new((4, 0, 3), green),
        Triangle::new((4, 3, 7), green),
        Triangle::new((5, 4, 7), blue),
        Triangle::new((5, 7, 6), blue),
        Triangle::new((1, 5, 6), yellow),
        Triangle::new((1, 6, 2), yellow),
        Triangle::new((4, 5, 1), purple),
        Triangle::new((4, 1, 0), purple),
        Triangle::new((2, 6, 7), cyan),
        Triangle::new((2, 7, 3), cyan),
    ];

    let cube = Model {
                    vertices: &vertices,
                    triangles: &triangles,
                    bounds_center: &Vector3::new(0.0, 0.0, 0.0),
                    bounds_radius: f64::sqrt(3.0),
                    };

    let instances = [ModelInstance::new(
                            &cube,
                            Vector3::new(-1.5, 0.0, 7.0),
                            Matrix4x4::identity(),
                            0.75,
                        ),
                     ModelInstance::new(
                            &cube,
                            Vector3::new(1.25, 2.5, 7.5),
                            Matrix4x4::new_oy_rotation_matrix(195.0),
                            1.0,
                        ),
                     ModelInstance::new(
                            &cube,
                            Vector3::new(-2.8, 1.0, 2.1),
                            Matrix4x4::new_oy_rotation_matrix(195.0),
                            0.15,
                        ),
                    ];

    let camera = Camera {
                    position: Vector3::new(-3.0, 1.0, 2.0),
                    orientation: Matrix4x4::new_oy_rotation_matrix(-30.0),
                    clipping_planes: [
                        Plane { normal: Vector3::new(0.0, 0.0, 1.0), distance: -1.0 },  // Near
                        Plane { normal: Vector3::new(SQRT_2, 0.0, SQRT_2), distance: 0.0 }, // Lft
                        Plane { normal: Vector3::new(-SQRT_2, 0.0, SQRT_2), distance: 0.0 }, // Rgt
                        Plane { normal: Vector3::new(0.0, -SQRT_2, SQRT_2), distance: 0.0 }, // Top
                        Plane { normal: Vector3::new(0.0, SQRT_2, SQRT_2), distance: 0.0 }, // Bttm
                    ],
                };

    render_scene(&mut canvas, &camera, &instances);

    canvas.display_until_exit();
}
