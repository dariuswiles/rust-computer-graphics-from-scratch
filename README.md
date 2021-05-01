This repository implements the examples from Gabriel Gambetta's [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/) book, in Rust. I am not affiliated with Gabriel or his book in any way, but I recommend it as a good way to understand the underlying mechanics of computer graphics by creating your own raytracing and rasterizer renderers purely in software.

The book uses pseudocode for its examples, but assumes a hypothetical `PutPixel` function is available that displays a pixel of a given color at given screen coordinates. This repository implements this functionality as a Rust library, and takes care of basic window creation and event handling. You may find it useful if you wish to implement the examples in the book without first having to create something similar yourself. The repository's `examples` directory contains my implementations of the examples from the book, but this is probably of less interest to you as you likely want to create your own.

TODO Explain the underlying minifb library.

TODO Provide a short example of using the library, including what a Rust project needs to add to its Cargo.toml.
