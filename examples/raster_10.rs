//! Implementation of pseudocode from chapter 12 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. This code implements a simple depth buffer, and renders objects with filled sides (rather
//! than the wireframes used in prior examples).
//!
//! I am not affiliated with Gabriel or his book in any way.

use std::iter::Iterator;
use std::mem;
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
struct Model {
    vertices: Vec<Vector4>,
    triangles: Vec<Triangle>,
    bounds_center: Vector4,
    bounds_radius: f64,
}


/// An instance of a particular model shape. Includes the `model` to use and its position,
/// orientation and scale. A `transform` matrix is generated when the object is created that
/// performs the translation, orientation and scaling, so these computations only need to be
/// performed once per instance. A `ModelInstance` object should not be changed after creation
/// because the `transform` would no longer be correct. This restriction includes the `position`
/// field, preventing instances being moved.
struct ModelInstance<'a> {
    model: &'a Model,
    #[allow(dead_code)]
    position: Vector4,
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
    fn new(model: &'a Model, position: Vector4, orientation: Matrix4x4, scale: f64) -> Self {
        let transform =
            Matrix4x4::new_translation_matrix_from_vec4(&position)
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
    position: Vector4,
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
        Self { x: x, y: y }
    }

    #[allow(dead_code)]
    fn from_ints(x: i32, y: i32) -> Self {
        Self {
            x: x as f64,
            y: y as f64
        }
    }
}


/// A 2D point with depth information
#[derive(Clone, Copy, Debug)]
struct PointWithDepth {
    pub x: f64,
    pub y: f64,
    pub depth: f64,
}

impl PointWithDepth {
    #[allow(dead_code)]
    fn new(x: f64, y: f64, depth: f64) -> Self {
        Self { x: x, y: y, depth: depth }
    }
}


/// A triangle, consisting of 3 vertex indices defining the position of its corners, and a color.
#[derive(Clone, Copy, Debug)]
struct Triangle {
    pub indexes: [usize; 3],
    pub color: Rgb,
}

impl Triangle {
    #[allow(dead_code)]
    fn new(vertices: [usize; 3], color: Rgb) -> Self {
        Self {
            indexes: vertices,
            color: color
        }
    }
}


/// A 2D plane, defined as a normal to the plane, and a distance from the origin. The normal should
/// be a unit vector, i.e., its length should be 1.
#[derive(Clone, Copy, Debug)]
struct Plane {
    pub normal: Vector4,
    pub distance: f64,
}


/// Translates a point on the `viewport` in viewport coordinates, e.g., -0.5 to 0.5, to the
/// corresponding point on the `canvas` in canvas coordinates, e.g., 0 to 600. The result is left
/// as a pair of `f64` values because further math will be performed, so converting to `i32`s is
/// premature.
fn viewport_to_canvas(x: f64, y: f64) -> Point {
    Point::new(
        x * CANVAS_WIDTH as f64 / VIEWPORT_WIDTH,
        y * CANVAS_HEIGHT as f64 / VIEWPORT_HEIGHT
    )
}


/// Translates a point in 3D space to the corresponding 2D point on the `viewport`. Also returns
/// `z` information for use with the depth buffer.
fn project_vertex(v: &Vector4) -> PointWithDepth {
    let viewport_coords = viewport_to_canvas(
        v.x * DISTANCE_FROM_CAMERA_TO_VIEWPORT / v.z,
        v.y * DISTANCE_FROM_CAMERA_TO_VIEWPORT / v.z
    );

    PointWithDepth::new(viewport_coords.x, viewport_coords.y, v.z)
}


/// A buffer to store the inverse depth, i.e., 1/z, at each pixel position of the canvas. Its width
/// and height are set to `CANVAS_WIDTH` and `CANVAS_HEIGHT`.
struct DepthBuffer {
    buffer: [[f64; CANVAS_WIDTH]; CANVAS_HEIGHT],
}

impl DepthBuffer {
    /// Creates a new DepthBuffer and initializes the depth of all cells to 0.0, which represents
    /// the farthest distance in the 1/z depth system.
    fn new() -> DepthBuffer {
        Self { buffer: [[0.0; CANVAS_WIDTH]; CANVAS_HEIGHT] }
    }

    /// Compares the stored depth of the `x`, `y` cell to the `depth` value passed (in 1/z format).
    /// If the value passed is less than or equal to the stored value (indicating that it is
    /// farther than the stored value), no changes are made and `false` is returned to indicate
    /// that the pixel is behind one that has already been drawn, so no action is required.
    /// Otherwise, the stored depth is updated with the value passed and `true` is returned.
    ///
    /// `x` and `y` must be defined in canvas coordinates, e.g., the range of `x` is
    /// -CANVAS_WIDTH/2..CANVAS_WIDTH/2. If `x` or `y` are outside their respective ranges, no
    /// changes are made and `false` is returned.
    fn check_set_nearer_pixel(&mut self, x: i32, y: i32, depth: f64) -> bool {
        let screen_x = CANVAS_WIDTH as i32/2 + x;
        let screen_y = CANVAS_HEIGHT as i32/2 - y - 1;

        if (screen_x < 0) | (screen_x >= CANVAS_WIDTH as i32) | (screen_y < 0) |
            (screen_y >= CANVAS_HEIGHT as i32) {
            return false;
        }

        if depth > self.buffer[screen_x as usize][screen_y as usize] {
            self.buffer[screen_x as usize][screen_y as usize] = depth;

            return true;
        }

        return false;
    }
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

    let range = (i0.round() as i32 ..= i1.round() as i32)
        .into_iter()
        .collect::<Vec<_>>();
    let delta = (d1 - d0) / (i1 - i0);
    let mut d = d0;

    for i in range {
        values.push((i as f64, d));
        d = d + delta;
    }

    values
}


// Interpolates over `y0`, `y1` and `y2`, which represent height coordinates of the three corners
// of a triangle. It must be the case that: `y0` <= `y1` <= `y2`, i.e., `y0` is the lowest corner
// and `y2` the highest. Associated dependent values `d0`, `d1` and `d2` are linearly interpolated
// between the heights of the three corners.
//
// A pair of vectors is returned: the first for the side of the triangle extending from `y0` to
// `y2` directly, and the other for the two sides that go from `y0` to `y2` via `y1`. Each vector
// maps every value of `y` to the interpolated value of the dependent variable `d`.
fn edge_interpolate(y0: f64, d0: f64, y1: f64, d1: f64, y2: f64, d2: f64)
    -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {

    let d01 = interpolate(y0, d0, y1, d1);
    let d12 = interpolate(y1, d1, y2, d2);
    let d02 = interpolate(y0, d0, y2, d2);

    // Concatenate `x01` and `x12`, but remove the value at the end of `x01` as it is repeated as
    // the first value of `x12`
    let d012 = [&d01[..d01.len()-1], &d12[..]].concat();

    (d02, d012)
}


/// Returns a non-normalized normal of the three points passed as arguments.
fn compute_triangle_normal(v0: &Vector4, v1: &Vector4, v2: &Vector4)
    -> Vector4 {

    // It is impossible to calculate the cross product on 4-element vectors, so reduce inputs to
    // 3-element vectors.
    let v0_new = Vector3::from_vector4(v0);
    let v1_new = Vector3::from_vector4(v1);
    let v2_new = Vector3::from_vector4(v2);

    let a = v1_new.subtract(&v0_new);
    let b = v2_new.subtract(&v0_new);
    return Vector4::from_vector3(&a.cross(&b), 0.0)
}


/// Renders a filled triangle on the canvas. `projected` contains the triangle's corners in
/// projected coordinates, i.e., 2D coordinates where `x` is in the range -width/2..width/2, and
/// likewise for height. The projected coordinates contain depth information for use with
/// `depth_buffer`.
fn render_triangle(
    canvas: &mut Canvas,
    depth_buffer: &mut DepthBuffer,
    triangle: &Triangle,
    vertices: &Vec<Vector4>,
    projected: &Vec<PointWithDepth>
) {
    let v0 = &vertices.get(triangle.indexes[0]).unwrap();
    let v1 = &vertices.get(triangle.indexes[1]).unwrap();
    let v2 = &vertices.get(triangle.indexes[2]).unwrap();

    // Compute the triangle's normal using the projected vertices. The vertices in the models are
    // ordered *clockwise*, so the *left*-hand rule is used to calculate the normal from the
    // cross product.
    let normal = compute_triangle_normal(&v0, &v1, &v2);

    // Backface culling.
    //
    // Calculate the center point of the triangle by averaging its three corners, and negate as a
    // shortcut for 'camera origin - center point'. This gives the vector from the center point to
    // the camera origin. Dot this with the triangle's normal to determine whether the triangle is
    // facing toward the camera, and return immediately if it isn't.
    let center = v0.add(&v1).add(&v2).multiply_by(-1.0/3.0);
    if center.dot(&normal) <= 0.0 {
        return;
    }

    let p0 = &projected.get(triangle.indexes[0]).unwrap();
    let p1 = &projected.get(triangle.indexes[1]).unwrap();
    let p2 = &projected.get(triangle.indexes[2]).unwrap();

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
    let (x02, x012) = edge_interpolate(
        corner0.y, corner0.x,
        corner1.y, corner1.x,
        corner2.y, corner2.x);

    // As above, but interpolate depth values.
    let (z02, z012) = edge_interpolate(
        corner0.y, 1.0/corner0.depth,
        corner1.y, 1.0/corner1.depth,
        corner2.y, 1.0/corner2.depth);

    let x_left;
    let x_right;
    let z_left;
    let z_right;
    let m = x02.len() / 2;

    // Look at the middle row of the triangle to determine whether `x02` or `x012` represents the
    // left side of the triangle.
    if x02[m].1 < x012[m].1 {   // Note that field `0` holds `x` coords, and `1` holds `y`.
        x_left = x02;
        x_right = x012;
        z_left = z02;
        z_right = z012;
    } else {
        x_left = x012;
        x_right = x02;
        z_left = z012;
        z_right = z02;
    }

    // For every canvas line, draw a row between the left and right sides of the triangle.
    for y in corner0.y.round() as i32 .. corner2.y.round() as i32 {
        let x_start = x_left.get((y - corner0.y.round() as i32) as usize).unwrap().1.round() as i32;
        let x_end = x_right.get((y - corner0.y.round() as i32) as usize).unwrap().1.round() as i32;
        let z_start = z_left.get((y - corner0.y.round() as i32) as usize).unwrap().1;
        let z_end = z_right.get((y - corner0.y.round() as i32) as usize).unwrap().1;

        // Compute depth information for every pixel we are about to draw so we can tell whether
        // they are in front of or behind any existing pixel that has been written at the same
        // location.
        let depth_info = interpolate(
            x_start as f64, z_start as f64,
            x_end as f64, z_end as f64,
        );

        for x in x_start .. x_end {
            if depth_buffer.check_set_nearer_pixel(x, y, depth_info[(x - x_start) as usize].1) {
                canvas.put_pixel(x, y, &triangle.color);
            }
        }
    }
}


/// Returns the distance between a given normalized `plane` and `vertex`. A positive value means
/// `vertex` is located on the side of the plane that `plane`'s normal points.
///
/// `plane.normal` must be a unit vector.
///
/// # Examples
/// ```
/// let plane = Plane { normal: Vector4::new(0.0, 0.0, 1.0, 0.0), distance: -1.0 };
/// let point = Vector4::new(0.0, 0.0, 3.0, 1.0);
///
/// assert_eq!(signed_distance(&plane, &point), 2.0);
/// ```
fn signed_distance(plane: &Plane, vertex: &Vector4) -> f64 {
    vertex.dot(&plane.normal) + plane.distance
}


/// Returns the point where line `v0` to `v1` intersects `plane`.
///
/// # Panics
///
/// Will panic with a divide by zero if the line `v0` to `v1` is parallel to `plane`.
fn intersection(v0: &Vector4, v1: &Vector4, plane: &Plane) -> Vector4 {
    let t = (-plane.distance - &plane.normal.dot(v0)) /
            (&plane.normal.dot(&v1.subtract(v0)));

    v0.add(&v1.subtract(v0).multiply_by(t))
}


/// Determines the the triangles needed to render the visible part of `triangle`. If `triangle` is
/// partially within the clipping volume, one or two new triangles that draw just the visible
/// portion of `triangle` are created.
///
/// The triangle (or triangles) necessary to render `triangle` are added to `triangles`, and any
/// additional vertexes required are added to `vertexes`. If `triangle` is completely within the
/// clipping volume, the only change is to add `triangle` to `triangles`. If `triangle` is
/// completely outside the clipping volume, nothing is added to `triangles` as no rendering is
/// required.
fn clip_triangle(
    triangle: Triangle,
    plane: &Plane,
    triangles: &mut Vec<Triangle>,
    vertexes: &mut Vec<Vector4>
) {
    let v0_idx = triangle.indexes[0];
    let v1_idx = triangle.indexes[1];
    let v2_idx = triangle.indexes[2];

    let v0 = vertexes.get(v0_idx).unwrap().clone();
    let v1 = vertexes.get(v1_idx).unwrap().clone();
    let v2 = vertexes.get(v2_idx).unwrap().clone();

    let d0 = signed_distance(plane, &v0);
    let d1 = signed_distance(plane, &v1);
    let d2 = signed_distance(plane, &v2);

    let mut inside_vertex_count = 0;
    let mut v0_is_inside = false;
    let mut v1_is_inside = false;
    let mut v2_is_inside = false;
    if d0 > 0.0 { inside_vertex_count += 1; v0_is_inside = true; }
    if d1 > 0.0 { inside_vertex_count += 1; v1_is_inside = true; }
    if d2 > 0.0 { inside_vertex_count += 1; v2_is_inside = true; }

    match inside_vertex_count {
        3 => triangles.push(triangle),
        2 => {
            // Two vertexes are inside the clipping volume. Discarding the part of the triangle
            // outside the clipping volume leaves an irregular quadilateral which is created as
            // two new triangles. The vertexes of one triangle are formed from the two vertexes
            // inside the clipping volume and one of the points of intersection between a triangle
            // edge that intersects the clipping plane. The other triangle is formed from one of
            // the vertexes inside the clipping volume and both points of intersection between the
            // triangle edges that intersect the clipping plane.

            if !v0_is_inside {
                let intersect10 = intersection(&v1, &v0, plane);
                let intersect20 = intersection(&v2, &v0, plane);

                vertexes.push(intersect10);
                let intersect10_idx = vertexes.len() - 1;
                vertexes.push(intersect20);
                let intersect20_idx = vertexes.len() - 1;

                triangles.push(Triangle::new(
                    [v1_idx, v2_idx, intersect20_idx],
                    triangle.color
                ));

                triangles.push(Triangle::new(
                    [v1_idx, intersect20_idx, intersect10_idx],
                    triangle.color
                ));
            } else if !v1_is_inside {
                let intersect01 = intersection(&v0, &v1, plane);
                let intersect21 = intersection(&v2, &v1, plane);

                vertexes.push(intersect01);
                let intersect01_idx = vertexes.len() - 1;
                vertexes.push(intersect21);
                let intersect21_idx = vertexes.len() - 1;

                triangles.push(Triangle::new(
                    [v2_idx, v0_idx, intersect01_idx],
                    triangle.color
                ));

                triangles.push(Triangle::new(
                    [v2_idx, intersect01_idx, intersect21_idx],
                    triangle.color
                ));
            } else {
                let intersect02 = intersection(&v0, &v2, plane);
                let intersect12 = intersection(&v1, &v2, plane);

                vertexes.push(intersect02);
                let intersect02_idx = vertexes.len() - 1;
                vertexes.push(intersect12);
                let intersect12_idx = vertexes.len() - 1;

                triangles.push(Triangle::new(
                    [v0_idx, v1_idx, intersect12_idx],
                    triangle.color
                ));

                triangles.push(Triangle::new(
                    [v0_idx, intersect12_idx, intersect02_idx],
                    triangle.color
                ));
            }
        }
        1 => {
            // One vertex is inside the clipping volume. A new triangle is created with this vertex
            // and two new vertexes at the positions the two sides of the triangle intersect the
            // clipping plane.

            let (mut new0_idx, mut new1_idx, mut new2_idx) = (usize::MAX, usize::MAX, usize::MAX);
            let inside_vertex;

            if v0_is_inside {
                inside_vertex = v0.clone();
                new0_idx = v0_idx;
            } else if v1_is_inside {
                inside_vertex = v1.clone();
                new1_idx = v1_idx;
            } else {
                inside_vertex = v2.clone();
                new2_idx = v2_idx;
            }

            if !v0_is_inside {
                let intersect = intersection(&inside_vertex, &v0, plane);
                vertexes.push(intersect);
                new0_idx = vertexes.len() - 1;
            }

            if !v1_is_inside {
                let intersect = intersection(&inside_vertex, &v1, plane);
                vertexes.push(intersect);
                new1_idx = vertexes.len() - 1;
            }

            if !v2_is_inside {
                let intersect = intersection(&inside_vertex, &v2, plane);
                vertexes.push(intersect);
                new2_idx = vertexes.len() - 1;
            }

            triangles.push(Triangle::new(
                [new0_idx, new1_idx, new2_idx],
                triangle.color
            ));
        }
        0 => {}
        _ => panic!("Internal error: unexpected number of triangle vertexes in clipping volume"),
    }
}


// Clips `model` against the five `clipping_planes`, modifying the lists of vertices and triangles
// so that only objects within the clip volume are included. Triangles that are only partially
// within the volume have their vertices redefined to the boundary of the clipping volume, and may
// be split into two triangles if needed.
//
// If `model` is completely outside the clipping volume, returns `None`. Otherwise, returns a new
// `Model` containing the modified vertices and triangles.
fn transform_and_clip(clipping_planes: &[Plane; 5], model: &Model, transform: &Matrix4x4)
    -> Option<Model> {

    // Apply `transform` to the center position of the bounding sphere and see if the instance is
    // completely outside any of the clipping planes. If so, discard the whole `model' by returning
    // `None` immediately.
    let transformed_center = transform.multiply_vector(&model.bounds_center);

    for cp in clipping_planes {
        let distance = &cp.normal.dot(&transformed_center) + cp.distance;
        if distance < -model.bounds_radius {
            return None;
        }
    }

    let mut modified_vertices = Vec::new();

    // Apply modelview transform to each vertex in the model instance.
    for v in &model.vertices {
        modified_vertices.push(transform.multiply_vector(&v));
    }

    // Loop through every clipping plane clipping all the model vertices for each. The output of
    // one interaction is used as input to the next to handle cases where a triangle intersects
    // multiple clipping planes.
    let mut triangles = model.triangles.clone();

    for cp in clipping_planes {
        let mut new_triangles = Vec::new();
        for t in triangles {
            clip_triangle(t, &cp, &mut new_triangles, &mut modified_vertices);
        }

        triangles = new_triangles;
    }

    Some(Model {
        vertices: modified_vertices,
        triangles: triangles.to_vec(),
        bounds_center: transformed_center,
        bounds_radius: model.bounds_radius,
    })
}


/// Renders the `Model` passed by iterating through the list of triangles and vertices
/// that it contains, using the `transform` provided to transform each vertex into camera space,
/// then calling `render_triangle` to draw the triangle on the 2D canvas.
fn render_instance(canvas: &mut Canvas, depth_buffer: &mut DepthBuffer, model: &Model) {
    let mut projected = vec![];

    // The `vertices` are defined in the 3D world coordinates, so project each vertex onto the
    // viewport, resulting in a vector of 2D viewport `Point`s.
    for v in &model.vertices {
        projected.push(project_vertex(&v));
    }

    // Render each triangle by passing coordinates of each corner as a 2D `Point` on the viewport.
    for t in &model.triangles {
        render_triangle(canvas, depth_buffer, &t, &model.vertices, &projected);
    }
}


/// Renders every model instance in the `instances` array passed. The `Camera` object passed is
/// used to create a world view to camera view transform by reversing its transpose and rotation.
/// This is combined with the transform included with each model instance to create a transform
/// that converts from instance space to camera space. This transform is generated once per
/// instance and passed as input to the `render_instance` function.
fn render_scene(
    canvas: &mut Canvas,
    depth_buffer: &mut DepthBuffer,
    camera: &Camera, instances: &[ModelInstance]
) {
    let camera_matrix = camera.orientation
        .transpose()
        .multiply_matrix4x4(&Matrix4x4::new_translation_matrix_from_vec4(
            &camera.position.multiply_by(-1.0),
        ));

    for mi in instances {
        let transform = camera_matrix.multiply_matrix4x4(&mi.transform);
        let clipped_model = transform_and_clip(&camera.clipping_planes, &mi.model, &transform);
        if let Some(cm) = clipped_model {
            render_instance(canvas, depth_buffer, &cm);
        }
    }
}


/// Creates a window, creates a scene containing two cubes and draws them with perspective
/// projection using rasterization techniques.
fn main() {
    let mut canvas = Canvas::new("Raster 10 (from chapter 12)", CANVAS_WIDTH, CANVAS_HEIGHT);
    let mut depth_buffer = DepthBuffer::new();


    let red = Rgb::from_ints(255,0,0);
    let green = Rgb::from_ints(0,255,0);
    let blue = Rgb::from_ints(0,0,255);
    let yellow = Rgb::from_ints(255,255,0);
    let purple = Rgb::from_ints(255,0,255);
    let cyan = Rgb::from_ints(0,255,255);
    let white = Rgb::from_ints(255,255,255);

    canvas.clear_canvas(&white);

    // Define vertices for the 8 corners of the cube to be rendered.
    let vertices = vec![
        Vector4::new( 1.0,  1.0,  1.0, 1.0),  // Vertex 0
        Vector4::new(-1.0,  1.0,  1.0, 1.0),  // Vertex 1
        Vector4::new(-1.0, -1.0,  1.0, 1.0),  // Vertex 2
        Vector4::new( 1.0, -1.0,  1.0, 1.0),  // Vertex 3
        Vector4::new( 1.0,  1.0, -1.0, 1.0),  // Vertex 4
        Vector4::new(-1.0,  1.0, -1.0, 1.0),  // Vertex 5
        Vector4::new(-1.0, -1.0, -1.0, 1.0),  // Vertex 6
        Vector4::new( 1.0, -1.0, -1.0, 1.0),  // Vertex 7
    ];

    // Define triangles with 3-tuples of indexes into the previously defined vertices.
    let triangles = vec![
        Triangle::new([0, 1, 2], red),
        Triangle::new([0, 2, 3], red),
        Triangle::new([4, 0, 3], green),
        Triangle::new([4, 3, 7], green),
        Triangle::new([5, 4, 7], blue),
        Triangle::new([5, 7, 6], blue),
        Triangle::new([1, 5, 6], yellow),
        Triangle::new([1, 6, 2], yellow),
        Triangle::new([4, 5, 1], purple),
        Triangle::new([4, 1, 0], purple),
        Triangle::new([2, 6, 7], cyan),
        Triangle::new([2, 7, 3], cyan),
    ];

    let cube = Model {
                    vertices: vertices,
                    triangles: triangles,
                    bounds_center: Vector4::new(0.0, 0.0, 0.0, 1.0),
                    bounds_radius: f64::sqrt(3.0),
                    };


    let instances = [ModelInstance::new(
                            &cube,
                            Vector4::new(-1.5, 0.0, 7.0, 1.0),
                            Matrix4x4::identity(),
                            0.75,
                        ),
                     ModelInstance::new(
                            &cube,
                            Vector4::new(1.25, 2.5, 7.5, 1.0),
//                             Vector4::new(3.3, 2.5, 7.5, 1.0),  // Use to test clipping
                            Matrix4x4::new_oy_rotation_matrix(195.0),
                            1.0,
                        ),
                    ];

    let s_z = 1.0 / f64::sqrt(5.0);
    let s_xy = 2.0 / f64::sqrt(5.0);
    let camera = Camera {
                    position: Vector4::new(-3.0, 1.0, 2.0, 1.0),
                    orientation: Matrix4x4::new_oy_rotation_matrix(-30.0),
                    clipping_planes: [
                        Plane { normal: Vector4::new(0.0, 0.0, 1.0, 0.0), distance: -1.0 }, // Near
                        Plane { normal: Vector4::new(s_xy, 0.0, s_z, 0.0), distance: 0.0 }, // Left
                        Plane { normal: Vector4::new(-s_xy, 0.0, s_z, 0.0), distance: 0.0 }, // Rgt
                        Plane { normal: Vector4::new(0.0, -s_xy, s_z, 0.0), distance: 0.0 }, // Top
                        Plane { normal: Vector4::new(0.0, s_xy, s_z, 0.0), distance: 0.0 }, // Btm
                    ],
                };

    render_scene(&mut canvas, &mut depth_buffer, &camera, &instances);

    canvas.display_until_exit();
}
