//! Canvas module documentation goes here

pub mod canvas {
    use minifb::{Key, ScaleMode, Window, WindowOptions};

    /// Canvas *struct* documentation goes here
    #[derive(Debug)]
    pub struct Canvas {
        window: Window,
        buffer: Vec<u32>,
        width: usize,
        height: usize,
    }

    impl Canvas {

        /// Documentation for `new`
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

            // Limit frame rate to a maximum of 50 frames per second
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
        /// Changes will only become visible the next time [`display`](#method.display) is called.
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


        ///  Update the window with the current canvas, making all `put_pixel` updates made since the last call to
        ///  `display` visible. Also checks if the user has requested the window to close or is pressing the escape key,
        ///  and exits the program if so.
        pub fn display(&mut self) {

            // The unwrap causes the code to exit if the update fails
            while self.window.is_open() && !self.window.is_key_down(Key::Escape) {
                self.window
                    .update_with_buffer(&self.buffer, self.width, self.height)
                    .unwrap();
            }
        }
    }

    /// TODO Rgb docs
    #[derive(Clone, Copy, Debug)]
    pub struct Rgb {
        pub red: u8,
        pub green: u8,
        pub blue: u8,
    }
}
