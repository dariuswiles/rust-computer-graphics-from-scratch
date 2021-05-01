//! Canvas module documentation goes here

pub mod canvas {
    #[allow(unused_imports)]
    use minifb::{Key, ScaleMode, Window, WindowOptions};

    /// Canvas *struct* documentation goes here

    // TODO Remove pub from fields (but not struct)
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
        //               behavior when the two don't match?
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

            // Limit frame rate to a maximum of about 60 frames per second
            window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

            let mut buffer: Vec<u32> = Vec::with_capacity(width * height);
            buffer.resize (width * height, 0);
            Canvas {window, buffer, width, height}
        }

        pub fn put_pixel (&mut self, x: usize, y: usize, rgb: &Rgb) {
            // TODO Validate input. Book requires this function to be very forgiving.
            self.buffer[x + self.width * y] = (rgb.red as u32) * 65536 + (rgb.green as u32) * 256 + (rgb.blue as u32);
        }

        pub fn display(&mut self) {

            // The unwrap causes the code to exit if the update fails
            while self.window.is_open() && !self.window.is_key_down(Key::Escape) {
                self.window
                    .update_with_buffer(&self.buffer, self.width, self.height)
                    .unwrap();
            }
        }
    }

    // TODO Remove need for all fields to be public?
    pub struct Rgb {
        pub red: u8,
        pub green: u8,
        pub blue: u8,
    }
}

