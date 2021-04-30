use minifb::{Key, ScaleMode, Window, WindowOptions};

const WIDTH: usize = 1200;
const HEIGHT: usize = 800;

fn main() {

    let mut window = Window::new(
        "minifb test - Press ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::UpperLeft,
            ..WindowOptions::default()
        },
    )
    .expect("Unable to create window");

    // Limit frame rate to a maximum of about 60 frames per second
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    let mut buffer: Vec<u32> = Vec::with_capacity(WIDTH * HEIGHT);

    let mut size = (0, 0);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let new_size = (window.get_size().0, window.get_size().1);
        if new_size != size {
            size = new_size;
            buffer.resize(size.0 * size.1, 0);
        }

        buffer[10000] = 255*65536 + 255 * 256 + 255;
        buffer[10001] = 255*65536 + 255 * 256 + 255;
        buffer[10002] = 0*65536 + 255 * 256 + 255;
        buffer[10003] = 0*65536 + 255 * 256 + 255;
        buffer[10004] = 255*65536 + 0 * 256 + 255;
        buffer[10005] = 255*65536 + 0 * 256 + 255;

//         for i in buffer.iter_mut() {
//             *i = 255*65536 + 180 * 256 + 90;
//         }

        // The unwrap causes the code to exit if the update fails
        window
            .update_with_buffer(&buffer, new_size.0, new_size.1)
            .unwrap();
    }
}
