This repository implements the examples from Gabriel Gambetta's [Computer Graphics from
Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/) book, in Rust. I am not
affiliated with Gabriel or his book in any way.

The book uses pseudocode for its examples, but assumes a hypothetical `PutPixel` function is
available that displays a pixel of a given color at given screen coordinates. This repository
implements this functionality as a Rust library, and takes care of basic window creation and event
handling. You may find it useful if you wish to implement the examples in the book without first
having to create something similar yourself. The repository's `examples` directory contains my
implementations of the examples from the book, but this is probably of less interest to you as you
likely want to create your own.

Here's an example that creates a window with a fixed canvas 600 pixels wide by 600 pixels high,
draws a tiny red "H" in the center and displays the results until the user presses escape or closes
the window.

```rust
use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};

const WIDTH: usize = 600;
const HEIGHT: usize = 600;

fn main() {
    let mut my_canvas = Canvas::new("A tiny red 'H'", WIDTH, HEIGHT);

    let red = Rgb::from_ints(255, 0, 0);

    my_canvas.put_pixel(-1, 1, &red);
    my_canvas.put_pixel(-1, 0, &red);
    my_canvas.put_pixel(-1, -1, &red);
    my_canvas.put_pixel(0, 0, &red);
    my_canvas.put_pixel(1, 1, &red);
    my_canvas.put_pixel(1, 0, &red);
    my_canvas.put_pixel(1, -1, &red);

    my_canvas.display_until_exit();
}
```


## License

The source code is licensed under the BSD Zero Clause License. See [LICENSE](LICENSE) or
https://opensource.org/licenses/0BSD.

The _*crate-texture.jpg*_ image is taken from
[Gabriel Gambetta's GitHub repository](https://github.com/ggambetta/computer-graphics-from-scratch) and he has released
it under the _*Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license*_ (BY-NC-SA). See
[LICENSE](LICENSE) or
https://creativecommons.org/licenses/by/4.0/


## Contribution

You agree that any contribution you submit for inclusion will be licensed under the BSD Zero Clause License.
