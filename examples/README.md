# Examples

Most examples in this directory are Rust implementations of the JavaScript demos on
[Gabriel Gambetta's website](https://gabrielgambetta.com/computer-graphics-from-scratch/) for his
book **Computer Graphics from Scratch**. I am not affiliated with Gabriel or his book in any way.

I wrote these examples while learning Rust, so the quality of the code isn't great from a Rust
perspective. I was also learning computer graphics, so it isn't great from that perspective
either! However, I'm expecting most people will implement their own versions of Gabriel's demos
rather than refer to my code.


## Summary of Examples

The following table lists the examples in this directory in the order they are presented in the
Computer Graphics from Scratch book. Except for the switch from ray tracing to rasterizing, each
example builds on the code of the previous example, except were noted.


| Example Name | Book Section | Summary |
| --- | --- | --- |
| raytracer_01.rs | Chapter 2 | The first step in implementing a basic ray tracer. Render three spheres against a white background. |
| raytracer_02.rs | Chapter 3 | Add a much larger sphere to act as a surface, and render with diffuse lighting consisting of ambient, point and directional light sources. |
| raytracer_03.rs | Chapter 3 | Add specular lighting. |
| raytracer_04.rs | Chapter 4 | Add shadows. |
| raytracer_05.rs | Chapter 4 | Add reflections. |
| raytracer_06.rs | Chapter 5 | Enable arbitrary camera positioning. |
| raytracer_bump.rs | Chapter 5 | Rudimentary attempt at bump mapping. It uses an algorithm to produce a textured pattern on the smaller spheres. |
| raytracer_transparency.rs | Chapter 5 | **Incomplete** attempt to extend raytracer_06 to render transparent objects. |
| raytracer_triangle.rs | Chapter 5 | Extend raytracer_06 to render triangles as well as spheres. |
| raytracer_triangle_threading.rs | Chapter 5 | Enhance raytracer_triangle to use multi-threading. |
| raster_01.rs | Chapter 6 | The first step in implementing rasterization. Draw lines using interpolation. |
| raster_02.rs | Chapter 6 | Separate interpolation code into a separate function so it can be reused. |
| raster_03.rs | Chapter 7 | Display wireframe and filled triangles. |
| raster_04.rs | Chapter 8 | Display filled triangles using interpolated shading. |
| raster_05.rs | Chapter 9 | Draw a wireframe cube using perspective projection. |
| raster_06.rs | Chapter 10 | Implement triangle drawing and render a wireframe cube constructed from triangles. |
| raster_07.rs | Chapter 10 | Make code in previous example more generic and render two wireframe cubes. |
| raster_08.rs | Chapter 10 | Switch to using homogeneous coordinates. |
| raster_09.rs | Chapter 11 | Implement clipping. |
| raster_10.rs | Chapter 12 | Implement a simple depth buffer and render objects with filled sides. |
| raster_11.rs | Chapter 13 | Implement flat, Gouraud and Phong shading algorithms; and diffuse and/or specular lighting. Normals can be generated automatically from model vertexes or taken from the models.* |
| raster_12.rs | Chapter 14 | Implement textures.* |
| raster_12_bilinear.rs | Chapter 14 | Add bilinear filters to raster_12.* |
| raster_12_bilinear_mipmap.rs | Chapter 14 | Add bilinear filters and mipmapping to raster_12.* |
| raster_12_bilinear_normal.rs | Chapter 15 | Add bump mapping based on normal adjustment to raster_12.* |

\* Use command-line options to select different behaviors. Use `cargo run --example example_name help` to view the options available for a particular example.


All examples rely on **vector_math** to define vector and matrix objects and associated math
operations.


## Licenses and Contribution

See [this repo's main README](../README.md) for license and contribution information.
