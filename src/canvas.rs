//! A crate providing basic functionality to display a window with an image constructed by setting pixel values one at a
//! time. It is intended purely for learning purposes, and in particular for people implementing the example exercises
//! in Gabriel Gambetta's [Computer Graphics from Scratch](https://gabrielgambetta.com/computer-graphics-from-scratch/)
//! book. Note that I am not affiliated with Gabriel or his book in any way.

use minifb::{Key, ScaleMode, Window, WindowOptions};

/// A basic wrapper around the `minifb` crate that creates a window with a black background. Individual pixels
/// can be set one at a time using `put_pixel`, but changes are not shown until `display_until_exit` is called.
/// `display_until_exit` does not return until the user exits the program, so no further changes can be made.
/// This basic functionality is meant purely as a learning aid.
///
/// # Examples
///
/// ```
/// // Display four white pixels in the center of the window
///
/// let mut my_canvas = Canvas::new("Title for my 800x600 window", 800, 600);
///
/// my_canvas.put_pixel(0, 0, Rgb(red: 255, green: 255, blue: 255));
/// my_canvas.put_pixel(0, 1, Rgb(red: 255, green: 255, blue: 255));
/// my_canvas.put_pixel(1, 0, Rgb(red: 255, green: 255, blue: 255));
/// my_canvas.put_pixel(1, 1, Rgb(red: 255, green: 255, blue: 255));
///
/// my_canvas.display_until_exit();
/// ```
#[derive(Debug)]
pub struct Canvas {
    window: Window,
    buffer: Vec<u32>,
    width: usize,
    height: usize,
}

impl Canvas {
    /// Creates a new window with a title of `name` and with the usable area (i.e., excluding window title bar and
    /// decorations), of the given `width` and `height`. The window is set as non-resizable to avoid the complexity
    /// of changing the usable canvas area in response to user actions.
    //
    // TODO Should any of the parameters be optional, and have default values if not given?
    // TODO Should name be the last parameter, as it is the most likely to be optional?
    // TODO Window resize is prevented to simplify code. However, would be good to add this bearing in mind:
    //          1. The current window size can be read with `window.get_size().0` and `.1`.
    //          2. The buffer can be resized with `buffer.resize(new_size.0 * new_size.1, 0);`
    //          3. The new size is passed during update with
    //                  `window.update_with_buffer(&buffer, new_size.0, new_size.1) ....`
    //          TODO The update fn takes the width & height of the buffer, not the window. What is minifb's
    //               behavior when the two don't match? (Only relevant if change to `resize: true`.)
    //               The API docs say the buffer needs to be at least as big as the window, but provide no further
    //               info.
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


/// An RGB color expressed as red, green and blue components ranging in value from 0 to 255.
///
/// # Examples
///
/// ```
///     let purple = Rgb {red: 255, green: 0, blue: 255};
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Rgb {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}
