//! A crate providing basic functionality to display a window with an image constructed by setting pixel values one at a
//! time. It is intended purely for learning purposes, and in particular for people implementing the example exercises
//! in Gabriel Gambetta's [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. Note that I am not affiliated with Gabriel or his book in any way.

pub mod canvas {
    use minifb::{Key, ScaleMode, Window, WindowOptions};

    /// A basic wrapper around the `minifb` crate that creates a window with a black background. Individual pixels
    /// can be set one at a time using `put_pixel`, but changes are not shown until `display_until_exit` is called.
    /// `display_until_exit` does not return until the user exits the program, so no further changes can be made.
    /// This basic functionality is meant purely as a learning aid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};
    /// # fn main() {
    /// // Display four white pixels in the center of the window
    ///
    /// let mut my_canvas = Canvas::new("Title for my 800x600 window", 800, 600);
    ///
    /// my_canvas.put_pixel(0, 0, &Rgb::from_ints(255, 255, 255));
    /// my_canvas.put_pixel(0, 1, &Rgb::from_ints(255, 255, 255));
    /// my_canvas.put_pixel(1, 0, &Rgb::from_ints(255, 255, 255));
    /// my_canvas.put_pixel(1, 1, &Rgb::from_ints(255, 255, 255));
    ///
    /// my_canvas.display_until_exit();
    /// # }
    /// ```
    #[derive(Debug)]
    pub struct Canvas {
        window: Window,
        buffer: Vec<u32>,
        width: usize,
        height: usize,
    }

    impl Canvas {
        /// Creates a new window with a title of `name` and with the usable area (i.e., excluding
        /// window title bar and decorations), of the given `width` and `height`. The window is set
        /// as non-resizable to avoid the complexity of changing the usable canvas area in response
        /// to user actions.
        pub fn new (name: &str, width: usize, height: usize) -> Self {
            let mut window = Window::new(
                name,
                width,
                height,
                WindowOptions {
                    resize: false,
                    scale_mode: ScaleMode::UpperLeft,
                    ..WindowOptions::default()
                },
            )
            .expect("Window creation failed");

            // Limit frame rate to a maximum of 50 frames per second to reduce CPU usage
            window.limit_update_rate(Some(std::time::Duration::from_micros(20_000)));

            let mut buffer: Vec<u32> = Vec::with_capacity(width * height);
            buffer.resize (width * height, 0);
            Canvas {window, buffer, width, height}
        }


        /// Clears the canvas with `color`.
        pub fn clear_canvas (&mut self, color: &Rgb) {
            let col: u32 = (color.red as u32) * 65536 +
                           (color.green as u32) * 256 +
                           (color.blue as u32);

            for i in 0 .. self.buffer.len() {
                self.buffer[i] = col;
            }
        }


        /// Sets a single pixel on the canvas at the given `x`, `y` coordinates. The center of the canvas is the origin,
        /// i.e., where `x = 0, y = 0`.
        /// `x` is the horizontal component ranging from `-width/2` at the furthest left to `width/2 - 1` at the
        /// furthest right.
        /// `y` is the vertical component, ranging from `-height/2` at the bottom to `height/2 - 1` at the top.
        /// If either `x` or `y` is outside these ranges, the function returns without setting a pixel.
        /// The pixel `color` is an [`Rgb` struct](struct.Rgb.html) defining red, green and blue components.
        /// Changes will only become visible when [`display_until_exit`](#method.display_until_exit) is called.
        pub fn put_pixel (&mut self, x: i32, y: i32, color: &Rgb) {
            let (width, height) = (self.width as i32, self.height as i32);

            let screen_x = width/2 + x;
            let screen_y = height/2 - y - 1;

            if (screen_x < 0) | (screen_x >= width) | (screen_y < 0) | (screen_y >= height) {
                return;
            }

            let pixel_pos_in_buffer = (screen_x + width * screen_y) as usize;

            self.buffer[pixel_pos_in_buffer] =
                (color.red as u32) * 65536 +
                (color.green as u32) * 256 +
                (color.blue as u32);
        }


        ///  Updates the window with all pixels set using `put_pixel` and displays this by looping continuously until
        ///  the window closes or the user presses the escape key. This function does not return until either event
        ///  occurs, which means this function can only be called once in a program.
        pub fn display_until_exit(&mut self) {
            // The unwrap causes the code to exit if the update fails
            while self.window.is_open() && !self.window.is_key_down(Key::Escape) {
                self.window
                    .update_with_buffer(&self.buffer, self.width, self.height)
                    .unwrap();
            }
        }
    }





    /// An RGB color expressed as red, green and blue components stored as floats and with no limits on their ranges.
    /// This is useful during calculations when values may be outside the normal RGB 0-255 range. Use `clamp` to
    ///
    /// # Examples
    ///
    /// ```
    /// # use rust_computer_graphics_from_scratch::canvas::{Rgb};
    /// # fn main() {
    /// let purple = Rgb {red: 255.0, green: -0.5, blue: 260.606};
    /// let clamped = purple.clamp();
    /// assert!(clamped.green == 0.0);
    /// assert!(clamped.blue == 255.0);
    /// # }
    /// ```
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Rgb {
        pub red: f64,
        pub green: f64,
        pub blue: f64,
    }

    impl Rgb {
        /// Individually multiplies the `red`, `green` and `blue` components by the value passed, and returns a new
        /// `Rgb` instance with the result.
        pub fn multiply_by(&self, m: f64) -> Rgb {
            Self {
                red: self.red * m,
                green: self.green * m,
                blue: self.blue * m,
            }
        }

        /// Individually adds the `red`, `green` and `blue` components with the corresponding components of the `Rgb`
        /// instance passed, , and returns a new `Rgb` instance with the result.
        pub fn add(&self, a: &Rgb) -> Rgb {
            Self {
                red: self.red + a.red,
                green: self.green + a.green,
                blue: self.blue + a.blue,
            }
        }

        /// Individually clamps the `red`, `green` and `blue` components to the range 0-255. Negative values become 0
        /// and values greater than 255 become 255.
        pub fn clamp(&self) -> Rgb {
            Rgb {
                red:   f64::min(255.0, f64::max(0.0, self.red)),
                green: f64::min(255.0, f64::max(0.0, self.green)),
                blue:  f64::min(255.0, f64::max(0.0, self.blue)),
            }
        }

        /// Creates a new `Rgb` instance from the `red`, `green` and `blue` values passed.
        pub fn from_ints(red: i16, green: i16, blue: i16) -> Rgb {
            Rgb {red: red as f64, green: green as f64, blue: blue as f64}
        }
    }
}
