//! Implementation of pseudocode from chapter 13 of Gabriel Gambetta's
//! [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. This code implements flat, Gouraud and Phong shading algorithms in conjunction with
//! diffuse and/or specular lighting, and normals can be generated automatically from model
//! vertexes or taken from normals provided as part of the models. Options are chosen using command
//! line options.
//!
//! I am not affiliated with Gabriel or his book in any way.

use std::env;
use std::f64::consts::PI;
use std::iter::Iterator;
use std::path::Path;
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


/// User-selectable lighting choices.
#[derive(Copy, Clone, Debug)]
enum Lighting {
    Diffuse,
    Specular,
    Both,
}

/// User-selectable shading choices.
#[derive(Copy, Clone, Debug)]
enum Shading {
    Flat,
    Gouraud,
    Phong,
}

/// User-selectable choices in the source of normals used for lighting.
#[derive(Copy, Clone, Debug)]
enum Normals {
    Computed,
    Model,
}

/// A structure to hold all user-selectable option choices.
#[derive(Copy, Clone, Debug)]
struct UserChoices {
    lighting: Lighting,
    shading: Shading,
    normals: Normals,
}


/// General struct to hold vertex and triangle data for any model shape.
struct Model {
    vertexes: Vec<Vector4>,
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
    position: Vector4,
    orientation: Matrix4x4,
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
    pub normals: [Vector4; 3]
}

impl Triangle {
    #[allow(dead_code)]
    fn new(indexes: [usize; 3], color: Rgb, normals: [Vector4; 3]) -> Self {
        Self {
            indexes: indexes,
            color: color,
            normals: normals,
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

#[derive(Clone, Copy, Debug)]
enum Light {
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
    vector: Vector4,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PointLightEntity {
    intensity: f64,
    position: Vector4,
}


/// Returns a `String` of information listing the command line options. The first line is a "usage"
/// line containing the name used to invoke this command, as passed in `cmd_name`.
fn help_information(cmd_name: &str) -> String {
    format!(
r#"Usage: {} [lighting] [shading] [normals]
Where lighting is one of: diffuse specular both
and shading is one of: flat gouraud phong
and normals is one of: computed model

The defaults are: both phong model
"#, cmd_name)
}


/// Returns a `Result` containing `UserChoices` with lighting, shading and normals options either
/// chosen by the user or not specified and at their default value. If the "help" option is chosen,
/// or any arguments are unrecognized, a help message listing usage information is shown and an
/// `Err` result is returned.
fn parse_command_line_arguments() -> Result<UserChoices, ()>{
    let mut args: Vec<String> = env::args().collect();

    let command_path = args.remove(0);
    let cmd_path = Path::new(&command_path);
    let cmd_name = &cmd_path.file_stem().unwrap().to_str().unwrap();

    let mut arg_help = false;
    let mut arg_unrecognized = false;
    let mut arg_lighting = Lighting::Both;
    let mut arg_shading = Shading::Phong;
    let mut arg_normals = Normals::Model;

    for arg in args {
        match arg.to_lowercase().as_str() {
            "help" | "-h" => arg_help = true,
            "diffuse" => arg_lighting = Lighting::Diffuse,
            "specular" => arg_lighting = Lighting::Specular,
            "both" => arg_lighting = Lighting::Both,
            "flat" => arg_shading = Shading::Flat,
            "gouraud" => arg_shading = Shading::Gouraud,
            "phong" => arg_shading = Shading::Phong,
            "computed" => arg_normals = Normals::Computed,
            "model" => arg_normals = Normals::Model,
            _ => arg_unrecognized = true,
        }
    }

    if arg_unrecognized { println!("{}: Unrecognized argument\n", &cmd_name); }

    if arg_help | arg_unrecognized {
        println!("{}", help_information(&cmd_name));
        return Result::Err(());
    }

    Result::Ok(UserChoices { lighting: arg_lighting, shading: arg_shading, normals: arg_normals } )
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


// Converts a `Point` in 2D canvas coordinates to a pair of `f64`s representing the equivalent
// 2D viewport coordinates. This conversion is the inverse of `viewport_to_canvas`.
fn canvas_to_viewport (p2d: Point) -> (f64, f64) {
    ((p2d.x * VIEWPORT_WIDTH / CANVAS_WIDTH as f64),
     (p2d.y * VIEWPORT_HEIGHT / CANVAS_HEIGHT as f64))
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


/// Translates a point in 3D space to the corresponding 2D point on the `viewport`. Also returns
/// `z` information for use with the depth buffer.
fn unproject_vertex(x: f64, y: f64, inv_z: f64) -> PointWithDepth { // TODOMaybe take: PointWithDepth
    let orig_z = 1.0 / inv_z;
    let ux = x * orig_z / DISTANCE_FROM_CAMERA_TO_VIEWPORT;
    let uy = y * orig_z /  DISTANCE_FROM_CAMERA_TO_VIEWPORT;
    let viewport_coords = canvas_to_viewport(Point { x: ux, y: uy });

    PointWithDepth::new(viewport_coords.0, viewport_coords.1, orig_z)
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


/// Generates and returns a `Model` approximating a sphere in the `color` passed. The sphere is
/// created with a triangle mesh dividing of the number of horizontal and vertical divisions
/// defined by `divs`.
fn generate_sphere(divs: i32, color: &Rgb) -> Model {
    let mut vertexes = Vec::new();
    let mut triangles = Vec::new();


    let delta_angle = 2.0 * PI / divs as f64;

    for d in 0..divs+1 {
        let y = (2.0 / divs as f64) * (d as f64 - divs as f64 / 2.0);
        let radius = f64::sqrt(1.0 - y*y);

        for i in 0..divs {
            let vertex = Vector4::new(
                radius*f64::cos(i as f64 * delta_angle),
                y,
                radius*f64::sin(i as f64 * delta_angle),
                1.0,
            );
            vertexes.push(vertex);
        }
    }

    // Generate triangles.
    for d in 0..divs {
        for i in 0..divs-1 {
            let i0 = (d*divs + i) as usize;

            triangles.push(Triangle::new(
                [i0, i0+(divs as usize)+1, i0+1],
                *color,
                [vertexes.get(i0).unwrap().clone(),
                    vertexes.get(i0+(divs as usize)+1).unwrap().clone(),
                    vertexes.get(i0+1).unwrap().clone()
                ],
            ));
            triangles.push(Triangle::new(
                [i0, i0+(divs as usize), i0+(divs as usize)+1],
                *color,
                [vertexes.get(i0).unwrap().clone(),
                    vertexes.get(i0+(divs as usize)).unwrap().clone(),
                    vertexes.get(i0+(divs as usize)+1).unwrap().clone()
                ],
            ));
        }
    }

    Model {
        vertexes: vertexes,
        triangles: triangles,
        bounds_center: Vector4::new(0.0, 0.0, 0.0, 1.0),
        bounds_radius: f64::sqrt(1.0),
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

    // TODO DE-BUG
    if (y0 > y1) | (y1 > y2) {
        panic!(format!("edge_interpolate called with y0={}, d0={}\t\ty1={}, d1={}\t\ty2={}, d2={}", y0, d0, y1, d1, y2, d2));
    }

    let d01 = interpolate(y0, d0, y1, d1);
    let d12 = interpolate(y1, d1, y2, d2);
    let d02 = interpolate(y0, d0, y2, d2);

//     println!("\td01 = {:?}", d01);
//     println!("\td12 = {:?}", d12);
//     println!("\td02 = {:?}", d02);

    // Concatenate `x01` and `x12`, but remove the value at the end of `x01` as it is repeated as
    // the first value of `x12`
    let d012 = [&d01[..d01.len()-1], &d12[..]].concat();

//     println!("\tReturning d02 = {:?}", d02); // de-BUG
//     println!("\tReturning d012 = {:?}\n", d012); // de-BUG

    (d02, d012)
}


/// The function is called with `vertex_indexes`, an array of 3 indexes into `projected`, a vector
/// of points. The function returns an array indicating the order in which the 3 elements of
/// `vertex_indexes` should be referenced in order for the y coordinates of the points in
/// `projected` to be sorted from smallest value to largest, i.e., from bottom to top. In other
/// words, the array return references elements in `indexes` which in turn reference points in
/// `projected`.
fn sorted_vertex_indexes(vertex_indexes: &[usize; 3], projected: &Vec<PointWithDepth>)
    -> [usize; 3]
{

    let mut mapped_indexes = (0..3)
        .map(|n| (n, projected.get(vertex_indexes[n]).unwrap()))
        .collect::<Vec<(_, _)>>();

    mapped_indexes.sort_by(|a, b| a.1.y.partial_cmp(&b.1.y).unwrap());

    [   mapped_indexes.get(0).unwrap().0,
        mapped_indexes.get(1).unwrap().0,
        mapped_indexes.get(2).unwrap().0
    ]
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


/// TODO documentation
fn compute_illumination(vertex: &Vector4, normal: &Vector4, camera: &Camera, lights: &[Light],
    user_choices: &UserChoices) -> f64
{
    let mut illumination = 0.0;

    for light in lights {
        let vl;
        let light_intensity;

        match light {
            Light::Ambient(am_light) => {
                illumination += am_light.intensity;
                continue;
            },
            Light::Directional(dir_light) => {
                let camera_matrix = camera.orientation.transpose();
                vl = camera_matrix.multiply_vector(&dir_light.vector);
                light_intensity = dir_light.intensity;
            },
            Light::Point(point_light) => {
                let camera_matrix = camera.orientation.transpose().multiply_matrix4x4(
                    &Matrix4x4::new_translation_matrix_from_vec4(&camera.position
                        .multiply_by(-1.0))
                );

                let transformed_light = &camera_matrix.multiply_vector(&point_light.position);  // TODO I believe position is correct, but differs from JS on GitHub.

                // Calculate a vector from the light to the vertex we are calculating illumination
                // for.
                vl = transformed_light.add(&vertex.multiply_by(-1.0));
                light_intensity = point_light.intensity;
            }
        }

        match user_choices.lighting {
            Lighting::Diffuse | Lighting::Both => {
                let cos_alpha = vl.dot(&normal) / vl.magnitude() * normal.magnitude();

                if cos_alpha > 0.0 {
                    illumination += cos_alpha * light_intensity;
                }
            },
            _ => ()
        }

        // Specular lighting calculation
        match user_choices.lighting {
            Lighting::Specular | Lighting::Both => {
                let reflected = normal.multiply_by(2.0 * vl.dot(&normal)).add(&vl.multiply_by(-1.0));
                let view = camera.position.add(&vertex.multiply_by(-1.0));

                let cos_beta = reflected.dot(&view) / reflected.magnitude() * view.magnitude();
                if cos_beta > 0.0 {
                    let specular = 50;

                    illumination += cos_beta.powi(specular) * light_intensity;
                }
            },
            _ => ()
        }
    }

    illumination
}


/// Renders a filled triangle on the canvas. `projected` contains the triangle's corners in
/// projected coordinates, i.e., 2D coordinates where `x` is in the range -width/2..width/2, and
/// likewise for height. The projected coordinates contain depth information for use with
/// `depth_buffer`.
fn render_triangle(
    triangle: &Triangle,
    vertexes: &Vec<Vector4>,
    projected: &Vec<PointWithDepth>,
    camera: &Camera,
    lights: &[Light],
    orientation: &Matrix4x4,
    canvas: &mut Canvas,
    depth_buffer: &mut DepthBuffer,
    user_choices: &UserChoices,
) {

    // Create three variables indexing into `triangle.indexes` that reference the points in
    // `projected` from lowest to highest y-coordinates. `i0` will indicate the entry in the
    // `triangle_indexes` array that references the `projected` point of the triangle's corners
    // that has the smallest y coordinate.
    // TODO Don't think I can assign to `indexes` directly, as JS does (at least not in Rust 2018). Worth a try though.
    let [i0, i1, i2] = sorted_vertex_indexes(&triangle.indexes, projected);
    let indexes = [i0, i1, i2];

//     println!("In render_triangle, indexes = {:?}", indexes);

    let v0 = &vertexes.get(triangle.indexes[i0]).unwrap();
    let v1 = &vertexes.get(triangle.indexes[i1]).unwrap();
    let v2 = &vertexes.get(triangle.indexes[i2]).unwrap();

    // Compute the triangle's normal using the projected vertexes. The vertexes in the models are
    // ordered *clockwise*, so the *left*-hand rule is used to calculate the normal from the
    // cross product.
    let normal = compute_triangle_normal(
        vertexes.get(triangle.indexes[0]).unwrap(),
        vertexes.get(triangle.indexes[1]).unwrap(),
        vertexes.get(triangle.indexes[2]).unwrap(),
    );

    // Backface culling.
    //
    // Calculate the center point of the triangle by averaging its three corners, and negate as a
    // shortcut for 'camera origin - center point'. This gives the vector from the center point to
    // the camera origin. Dot this with the triangle's normal to determine whether the triangle is
    // facing toward the camera, and return immediately if it isn't.
    let center = vertexes.get(triangle.indexes[0]).unwrap()
                    .add(vertexes.get(triangle.indexes[1]).unwrap())
                    .add(vertexes.get(triangle.indexes[2]).unwrap())
                    .multiply_by(-1.0/3.0);
    if center.dot(&normal) <= 0.0 {
        return;
    }

    let p0 = &projected.get(triangle.indexes[i0]).unwrap();
    let p1 = &projected.get(triangle.indexes[i1]).unwrap();
    let p2 = &projected.get(triangle.indexes[i2]).unwrap();

    let corner0 = p0;
    let corner1 = p1;
    let corner2 = p2;

    // DEBUG
    if (p0.y > p1.y) | (p1.y > p2.y) {
        panic!(format!("In render_triangle with p0.y, p1.y, p2.y = {}, {}, {}", p0.y, p1.y, p2.y));
    }

    // Interpolate with the `y` coordinates as the independent variable because we want the value
    // `x` for each row (rather than looping over `x` to find `y`). The results are `vec`s of
    // `(f64, f64)`, representing `(y, x)` coordinates.
    let (x02, x012) = edge_interpolate(
        corner0.y, corner0.x,
        corner1.y, corner1.x,
        corner2.y, corner2.x);

    // As above, but interpolate depth values.
    let (iz02, iz012) = edge_interpolate(
        corner0.y, 1.0/corner0.depth,
        corner1.y, 1.0/corner1.depth,
        corner2.y, 1.0/corner2.depth);


    // Depending on the user's choice, set normals to either those defined with the model, or ones
    // we compute from the vertexes.
    let normal0;
    let normal1;
    let normal2;

    match user_choices.normals {
        Normals::Model => {
            let transform = camera.orientation.transpose().multiply_matrix4x4(&orientation);

            normal0 = transform.multiply_vector(&triangle.normals[i0]);
            normal1 = transform.multiply_vector(&triangle.normals[i1]);
            normal2 = transform.multiply_vector(&triangle.normals[i2]);
        },
        Normals::Computed => {
            normal0 = normal.clone();
            normal1 = normal.clone();
            normal2 = normal.clone();
        },
    }

    // TODO Implement the below. Difficulty is passing different variables to code later in this fn. The different shading techniques need to pass different variables. :(
    match user_choices.shading {
        Shading::Flat => {
            let _ = &lights;
            // Flat shading: compute lighting for the entire triangle.
//     var center = Vertex((v0.x + v1.x + v2.x)/3.0, (v0.y + v1.y + v2.y)/3.0, (v0.z + v1.z + v2.z)/3.0);
//     var intensity = ComputeIllumination(center, normal0, camera, lights);



        },
        Shading::Gouraud => {
            // Gouraud shading: compute lighting at the vertexes, and interpolate.
//     var i0 = ComputeIllumination(v0, normal0, camera, lights);
//     var i1 = ComputeIllumination(v1, normal1, camera, lights);
//     var i2 = ComputeIllumination(v2, normal2, camera, lights);

        },
        Shading::Phong => {
            // Phong shading: interpolate normal vectors.
    /*var [nx02, nx012] = EdgeInterpolate(p0.y, normal0.x, p1.y, normal1.x, p2.y, normal2.x);
    var [ny02, ny012] = EdgeInterpolate(p0.y, normal0.y, p1.y, normal1.y, p2.y, normal2.y);
    var [nz02, nz012] = EdgeInterpolate(p0.y, normal0.z, p1.y, normal1.z, p2.y, normal2.z);
    */

        },
    }

    let x_left;
    let x_right;
    let iz_left;
    let iz_right;
    let m = x02.len() / 2;

    // Look at the middle row of the triangle to determine whether `x02` or `x012` represents the
    // left side of the triangle.
    if x02[m].1 < x012[m].1 {   // Note that field `0` holds `x` coords, and `1` holds `y`.
        x_left = x02;
        x_right = x012;
        iz_left = iz02;
        iz_right = iz012;
    } else {
        x_left = x012;
        x_right = x02;
        iz_left = iz012;
        iz_right = iz02;
    }

    // For every canvas line, draw a row between the left and right sides of the triangle.
    for y in corner0.y.round() as i32 .. corner2.y.round() as i32 {
        let x_start = x_left.get((y - corner0.y.round() as i32) as usize).unwrap().1.round() as i32;
        let x_end = x_right.get((y - corner0.y.round() as i32) as usize).unwrap().1.round() as i32;
        let iz_start = iz_left.get((y - corner0.y.round() as i32) as usize).unwrap().1;
        let iz_end = iz_right.get((y - corner0.y.round() as i32) as usize).unwrap().1;

        // Compute depth information for every pixel we are about to draw so we can tell whether
        // they are in front of or behind any existing pixel that has been written at the same
        // location.
        let depth_info = interpolate(
            x_start as f64, iz_start as f64,
            x_end as f64, iz_end as f64,
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

    let v0 = vertexes.get(v0_idx).unwrap();
    let v1 = vertexes.get(v1_idx).unwrap();
    let v2 = vertexes.get(v2_idx).unwrap();

    let d0 = signed_distance(plane, &v0);
    let d1 = signed_distance(plane, &v1);
    let d2 = signed_distance(plane, &v2);

    let mut positive = Vec::new();
    let mut negative = Vec::new();
    if d0 > 0.0 { positive.push(v0_idx); } else { negative.push(v0_idx); }
    if d1 > 0.0 { positive.push(v1_idx); } else { negative.push(v1_idx); }
    if d2 > 0.0 { positive.push(v2_idx); } else { negative.push(v2_idx); }

    match positive.len() {
        3 => triangles.push(triangle),
        2 => {
            let a_idx = positive.pop().unwrap();
            let b_idx = positive.pop().unwrap();
            let c_idx = negative.pop().unwrap();

            let a = vertexes.get(a_idx).unwrap();
            let b = vertexes.get(b_idx).unwrap();
            let c = vertexes.get(c_idx).unwrap();

            let a_prime = intersection(&a, &c, plane);
            let b_prime = intersection(&b, &c, plane);

            vertexes.push(a_prime);
            let a_prime_idx = vertexes.len() - 1;
            vertexes.push(b_prime);
            let b_prime_idx = vertexes.len() - 1;

            triangles.push(Triangle::new(
                [a_idx, b_idx, a_prime_idx],
                triangle.color,
                [ // TODO These are the normals for the corners. W/o book, I need to determine what this should be. Maybe interpolate normals from corners?
                    Vector4::new(0.0, 1.0, 0.0, 1.0),
                    Vector4::new(0.0, 1.0, 0.0, 1.0),
                    Vector4::new(0.0, 1.0, 0.0, 1.0),
                ]
            ));
            triangles.push(Triangle::new(
                [a_prime_idx, b_idx, b_prime_idx],
                triangle.color,
                [ // TODO These are the normals for the corners. W/o book, I need to determine what this should be. Maybe interpolate normals from corners?
                    Vector4::new(0.0, 1.0, 0.0, 1.0),
                    Vector4::new(0.0, 1.0, 0.0, 1.0),
                    Vector4::new(0.0, 1.0, 0.0, 1.0),
                ]
            ));
        }
        1 => {
            let a_idx = positive.pop().unwrap();
            let b_idx = negative.pop().unwrap();
            let c_idx = negative.pop().unwrap();

            let a = vertexes.get(a_idx).unwrap();
            let b = vertexes.get(b_idx).unwrap();
            let c = vertexes.get(c_idx).unwrap();

            let b_prime = intersection(&a, &b, plane);
            let c_prime = intersection(&a, &c, plane);

            vertexes.push(b_prime);
            let b_prime_idx = vertexes.len() - 1;
            vertexes.push(c_prime);
            let c_prime_idx = vertexes.len() - 1;

            triangles.push(Triangle::new(
                [a_idx, b_prime_idx, c_prime_idx],
                triangle.color,
                [ // TODO These are the normals for the corners. W/o book, I need to determine what this should be. Maybe interpolate normals from corners?
                    Vector4::new(0.0, 1.0, 0.0, 1.0),
                    Vector4::new(0.0, 1.0, 0.0, 1.0),
                    Vector4::new(0.0, 1.0, 0.0, 1.0),
                ]
            ));
        }
        0 => {}
        _ => panic!("Internal error: unexpected number of triangle vertexes in clipping volume"),
    }
}


// Clips `model` against the five `clipping_planes`, modifying the lists of vertexes and triangles
// so that only objects within the clip volume are included. Triangles that are only partially
// within the volume have their vertexes redefined to the boundary of the clipping volume, and may
// be split into two triangles if needed.
//
// If `model` is completely outside the clipping volume, returns `None`. Otherwise, returns a new
// `Model` containing the modified vertexes and triangles.
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

    let mut modified_vertexes = Vec::new();

    // Apply modelview transform to each vertex in the model instance.
    for v in &model.vertexes {
        modified_vertexes.push(transform.multiply_vector(&v));
    }

    // Loop through every clipping plane clipping all the model vertexes for each. The output of
    // one interaction is used as input to the next to handle cases where a triangle intersects
    // multiple clipping planes.
    let mut triangles = model.triangles.clone();

    for cp in clipping_planes {
        let mut new_triangles = Vec::new();
        for t in triangles {
            clip_triangle(t, &cp, &mut new_triangles, &mut modified_vertexes);
        }

        triangles = new_triangles;
    }

    Some(Model {
        vertexes: modified_vertexes,
        triangles: triangles.to_vec(),
        bounds_center: transformed_center,
        bounds_radius: model.bounds_radius,
    })
}


/// Renders the `Model` passed by iterating through the list of triangles and vertexes
/// that it contains, using the `transform` provided to transform each vertex into camera space,
/// then calling `render_triangle` to draw the triangle on the 2D canvas.
fn render_model(model: &Model, camera: &Camera, lights: &[Light], orientation: &Matrix4x4,
    canvas: &mut Canvas, depth_buffer: &mut DepthBuffer, user_choices: &UserChoices)
{
    let mut projected = vec![];

    // The `vertexes` are defined in the 3D world coordinates, so project each vertex onto the
    // viewport, resulting in a vector of 2D viewport `Point`s.
    for v in &model.vertexes {
        projected.push(project_vertex(&v));
    }

    // Render each triangle by passing coordinates of each corner as a 2D `Point` on the viewport.
    for t in &model.triangles {
        render_triangle(&t, &model.vertexes, &projected, camera, lights, orientation,
        canvas, depth_buffer, user_choices);
    }
}


/// Renders every model instance in the `instances` array passed. The `Camera` object passed is
/// used to create a world view to camera view transform by reversing its transpose and rotation.
/// This is combined with the transform included with each model instance to create a transform
/// that converts from instance space to camera space. This transform is generated once per
/// instance and passed as input to the `render_model` function.
fn render_scene(
    camera: &Camera,
    instances: &[ModelInstance],
    lights: &[Light],
    canvas: &mut Canvas,
    depth_buffer: &mut DepthBuffer,
    user_choices: &UserChoices,
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
            render_model(&cm, &camera, lights, &mi.orientation, canvas, depth_buffer,
                user_choices);
        }
    }
}


/// Creates a window, creates a scene containing two cubes and a sphere and draws them with
/// perspective projection using rasterization techniques.
fn main() {
    let user_choices;

    if let Result::Ok(choices) = parse_command_line_arguments() {
        user_choices = choices;
        println!("Arguments read successfully");
        println!("\tLighting = {:#?}", user_choices.lighting);
        println!("\tShading = {:#?}", user_choices.shading);
        println!("\tNormals = {:#?}", user_choices.normals);
    } else {
        return;
    }

    let mut canvas = Canvas::new("Raster 11 (from chapter 13)", CANVAS_WIDTH, CANVAS_HEIGHT);
    let mut depth_buffer = DepthBuffer::new();


    let red = Rgb::from_ints(255,0,0);
    let green = Rgb::from_ints(0,255,0);
    let blue = Rgb::from_ints(0,0,255);
    let yellow = Rgb::from_ints(255,255,0);
    let purple = Rgb::from_ints(255,0,255);
    let cyan = Rgb::from_ints(0,255,255);
    let white = Rgb::from_ints(255,255,255);

    canvas.clear_canvas(&white);

    // Define vertexes for the 8 corners of the cube to be rendered.
    let vertexes = vec![
        Vector4::new( 1.0,  1.0,  1.0, 1.0),  // Vertex 0
        Vector4::new(-1.0,  1.0,  1.0, 1.0),  // Vertex 1
        Vector4::new(-1.0, -1.0,  1.0, 1.0),  // Vertex 2
        Vector4::new( 1.0, -1.0,  1.0, 1.0),  // Vertex 3
        Vector4::new( 1.0,  1.0, -1.0, 1.0),  // Vertex 4
        Vector4::new(-1.0,  1.0, -1.0, 1.0),  // Vertex 5
        Vector4::new(-1.0, -1.0, -1.0, 1.0),  // Vertex 6
        Vector4::new( 1.0, -1.0, -1.0, 1.0),  // Vertex 7
    ];

    // Define triangles with an array of 3 indexes referencing the `vertexes` array.
    let triangles = vec![
        Triangle::new([0, 1, 2], red, [
            Vector4::new(0.0, 0.0, 1.0, 1.0),
            Vector4::new(0.0, 0.0, 1.0, 1.0),
            Vector4::new(0.0, 0.0, 1.0, 1.0),
        ]),
        Triangle::new([0, 2, 3], red, [
            Vector4::new(0.0, 0.0, 1.0, 1.0),
            Vector4::new(0.0, 0.0, 1.0, 1.0),
            Vector4::new(0.0, 0.0, 1.0, 1.0),
        ]),
        Triangle::new([4, 0, 3], green, [
            Vector4::new(1.0, 0.0, 0.0, 1.0),
            Vector4::new(1.0, 0.0, 0.0, 1.0),
            Vector4::new(1.0, 0.0, 0.0, 1.0),
        ]),
        Triangle::new([4, 3, 7], green, [
            Vector4::new(1.0, 0.0, 0.0, 1.0),
            Vector4::new(1.0, 0.0, 0.0, 1.0),
            Vector4::new(1.0, 0.0, 0.0, 1.0),
        ]),
        Triangle::new([5, 4, 7], blue, [
            Vector4::new(0.0, 0.0, -1.0, 1.0),
            Vector4::new(0.0, 0.0, -1.0, 1.0),
            Vector4::new(0.0, 0.0, -1.0, 1.0),
        ]),
        Triangle::new([5, 7, 6], blue, [
            Vector4::new(0.0, 0.0, -1.0, 1.0),
            Vector4::new(0.0, 0.0, -1.0, 1.0),
            Vector4::new(0.0, 0.0, -1.0, 1.0),
        ]),
        Triangle::new([1, 5, 6], yellow, [
            Vector4::new(-1.0, 0.0, 0.0, 1.0),
            Vector4::new(-1.0, 0.0, 0.0, 1.0),
            Vector4::new(-1.0, 0.0, 0.0, 1.0),
        ]),
        Triangle::new([1, 6, 2], yellow, [
            Vector4::new(-1.0, 0.0, 0.0, 1.0),
            Vector4::new(-1.0, 0.0, 0.0, 1.0),
            Vector4::new(-1.0, 0.0, 0.0, 1.0),
        ]),
        Triangle::new([4, 5, 1], purple, [
            Vector4::new(0.0, 1.0, 0.0, 1.0),
            Vector4::new(0.0, 1.0, 0.0, 1.0),
            Vector4::new(0.0, 1.0, 0.0, 1.0),
        ]),
        Triangle::new([4, 1, 0], purple, [
            Vector4::new(0.0, 1.0, 0.0, 1.0),
            Vector4::new(0.0, 1.0, 0.0, 1.0),
            Vector4::new(0.0, 1.0, 0.0, 1.0),
        ]),
        Triangle::new([2, 6, 7], cyan, [
            Vector4::new(0.0, -1.0, 0.0, 1.0),
            Vector4::new(0.0, -1.0, 0.0, 1.0),
            Vector4::new(0.0, -1.0, 0.0, 1.0),
        ]),
        Triangle::new([2, 7, 3], cyan, [
            Vector4::new(0.0, -1.0, 0.0, 1.0),
            Vector4::new(0.0, -1.0, 0.0, 1.0),
            Vector4::new(0.0, -1.0, 0.0, 1.0),
        ]),
    ];

    let cube = Model {
                    vertexes: vertexes,
                    triangles: triangles,
                    bounds_center: Vector4::new(0.0, 0.0, 0.0, 1.0),
                    bounds_radius: f64::sqrt(3.0),
                };

    let sphere = generate_sphere(15, &green);

    let instances = [
        ModelInstance::new(
            &cube,
            Vector4::new(-1.5, 0.0, 7.0, 1.0),
            Matrix4x4::identity(),
            0.75,
        ),
        ModelInstance::new(
            &cube,
            Vector4::new(1.25, 2.5, 7.5, 1.0),
            Matrix4x4::new_oy_rotation_matrix(195.0),
            1.0,
        ),
        ModelInstance::new(
            &sphere,
            Vector4::new(1.75, -0.5, 7.0, 1.0),
            Matrix4x4::identity(),
            1.5,
        ),
    ];

    let s_z = 1.0 / f64::sqrt(5.0);
    let s_xy = 2.0 / f64::sqrt(5.0);
    let camera = Camera {
        position: Vector4::new(-3.0, 1.0, 2.0, 1.0),
        orientation: Matrix4x4::new_oy_rotation_matrix(-30.0),
        clipping_planes: [
            Plane { normal: Vector4::new(0.0, 0.0, 1.0, 0.0), distance: -1.0 },  // Near
            Plane { normal: Vector4::new(s_xy, 0.0, s_z, 0.0), distance: 0.0 },  // Left
            Plane { normal: Vector4::new(-s_xy, 0.0, s_z, 0.0), distance: 0.0 },  // Right
            Plane { normal: Vector4::new(0.0, -s_xy, s_z, 0.0), distance: 0.0 },  // Top
            Plane { normal: Vector4::new(0.0, s_xy, s_z, 0.0), distance: 0.0 },  // Bottom
        ],
    };

    let lights = [
        Light::Ambient(AmbientLightEntity { intensity: 0.2, } ),
        Light::Point(PointLightEntity {
            intensity: 0.6,
            position: Vector4::new(2.0, 1.0, 0.0, 1.0),
        } ),
        Light::Directional(DirectionalLightEntity {
            intensity: 0.2,
            vector: Vector4::new(1.0, 4.0, 4.0, 0.0),
        } ),
    ];

    render_scene(&camera, &instances, &lights, &mut canvas, &mut depth_buffer, &user_choices);

    canvas.display_until_exit();
}
