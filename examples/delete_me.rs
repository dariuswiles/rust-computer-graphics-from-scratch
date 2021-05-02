use rust_computer_graphics_from_scratch::canvas::{Canvas, Rgb};

const WIDTH: usize = 1000;
const HEIGHT: usize = 600;

fn main() {

    let mut my_canvas = Canvas::new("Window title goes here", WIDTH, HEIGHT);

    let white = Rgb {red: 255, green: 255, blue: 255};
    let red = Rgb {red: 255, green: 0, blue: 0};
    let green = Rgb {red: 0, green: 255, blue: 0};
    let purple = Rgb {red: 255, green: 0, blue: 255};

    my_canvas.put_pixel(100, 10, &white);
    my_canvas.put_pixel(101, 10, &white);
    my_canvas.put_pixel(100, 11, &white);
    my_canvas.put_pixel(101, 11, &white);
    my_canvas.put_pixel(100, 12, &white);
    my_canvas.put_pixel(101, 12, &white);
    my_canvas.put_pixel(100, 20, &red);
    my_canvas.put_pixel(101, 20, &red);
    my_canvas.put_pixel(100, 21, &red);
    my_canvas.put_pixel(101, 21, &red);
    my_canvas.put_pixel(100, 22, &red);
    my_canvas.put_pixel(101, 22, &red);

    my_canvas.put_pixel(120, 10, &green);
    my_canvas.put_pixel(121, 10, &green);
    my_canvas.put_pixel(120, 11, &green);
    my_canvas.put_pixel(121, 11, &green);
    my_canvas.put_pixel(120, 12, &green);
    my_canvas.put_pixel(121, 12, &green);
    my_canvas.put_pixel(120, 20, &purple);
    my_canvas.put_pixel(121, 20, &purple);
    my_canvas.put_pixel(120, 21, &purple);
    my_canvas.put_pixel(121, 21, &purple);
    my_canvas.put_pixel(120, 22, &purple);
    my_canvas.put_pixel(121, 22, &purple);

    my_canvas.display();
}
