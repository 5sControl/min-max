import cv2
import numpy as np

convert_to_tuple_of_ints = lambda x: tuple(list(map(int, x)))


def put_text_using_cv2(src, text, bl_position, bgr_color, thickness=1, shadow_shift=1, **kwargs) -> None:
    for shift, color in [(shadow_shift, (0, 0, 0)), (0, bgr_color)]:
        cv2.putText(src,
                    f"{text}",
                    org=(bl_position[0] - shift, bl_position[1] - shift),
                    color=color,
                    thickness=thickness,
                    fontFace=kwargs.get("fontFace", cv2.FONT_HERSHEY_SIMPLEX),
                    lineType=kwargs.get("lineType", cv2.LINE_AA),
                    fontScale=kwargs.get("fontScale", 1 / 2))


def put_rectangle_using_cv2(src, ltrb_rectangle, bgr_color, thickness=2, shadow_shift=1, **kwargs) -> None:
    l, t, r, b = ltrb_rectangle
    for shift, color in [(shadow_shift, (0, 0, 0)), (0, bgr_color)]:
        cv2.rectangle(src,
                      pt1=(l - shift, t - shift),
                      pt2=(r - shift, b - shift),
                      color=color,
                      thickness=thickness,
                      lineType=kwargs.get("lineType", cv2.LINE_AA))


def put_rectangle(src, rectangle, color, **kwargs):
    """Draws a simple, thick, or filled up-right rectangle.

    Important! This function is allowed to perform rounding of coordinates and color intensities.
    Args:
      src: ndarray image of shape (H, W, n_channels).
      rectangle: ndarray with rectangles in left-, top-, right-, bottom- format.
      color: ndarray with color or colors in BGR format.
      kwargs: optianaly, thikness, shadow_shift, lineType, fontFace or fontFace
    Returns:
         None
    """
    ltrb_rectangles = np.array(rectangle).reshape(-1, 4)
    bgr_colors = np.array(color).reshape(-1, 3)

    for i, rectangle in enumerate(ltrb_rectangles):
        bgr_color = bgr_colors[i % len(bgr_colors)]
        put_rectangle_using_cv2(src, convert_to_tuple_of_ints(rectangle), convert_to_tuple_of_ints(bgr_color), **kwargs)


def put_text(src, text, position, color, **kwargs):
    "Important! This function is allowed to perform rounding of coordinates and color intensities."
    put_text_using_cv2(src, text, convert_to_tuple_of_ints(position), convert_to_tuple_of_ints(color), **kwargs)


def put_rectangle_with_left_aligned_title(src, text, rectangle, color, padding=(0, 5), **kwargs) -> None:
    assert padding[0] >= 0 and padding[1] >= 0, 'The `padding` argument can only contain positive values.'
    l, t, _, _ = rectangle
    put_rectangle(src, rectangle, color, **kwargs)
    put_text(src, text, (l - padding[0], t - padding[1]), color, **kwargs)


def put_legend(src,
               short_descriptions,
               colors,
               position,
               marker_size=(36, 16),
               margin=(2, 2, 2, 2),
               padding=(11, 7),
               marker_frame_width=2,
               **kwargs):
    """ Draw simple legend for each of the provided colors with provided descriptions.

    Args:
        src: ndarray of shape `(H, W, n_channes)`, where n_channes is ether 1, 3 or 4.
        short_descriptions: list of strings
        colors: list of colors in `BGR` format of the same length as short_descriptions argument.
        position: left boddom position of the whole legend primitive.
        marker_size: size of the color marker.
        margin: margin of the legend primitive in `(l, t, r, b)` format.
        padding: inner padding between markers and descriptions.
        marker_frame_width: size of the marker frame.
        kwargs: put_rectangle and put_text kwargs.

    Return:
        None
    """
    assert len(short_descriptions) == len(colors), "Legend is suppose to contain description for each color."
    position = (position[0],
                position[1] - margin[1] - margin[3] - (marker_size[1] + padding[1]) * len(colors))
    for i, (description, color) in enumerate(zip(short_descriptions, colors)):
        cursor_position = (position[0] + margin[0],
                           position[1] + margin[1] + (marker_size[1] + padding[1]) * i)

        put_rectangle(src,
                      rectangle=(marker_frame_width + cursor_position[0],
                                 marker_frame_width + cursor_position[1],
                                 -marker_frame_width + cursor_position[0] + marker_size[0],
                                 -marker_frame_width + cursor_position[1] + marker_size[1]),
                      color=color,
                      thickness=-1,
                      **kwargs)

        put_text(src,
                 description,
                 position=(cursor_position[0] + padding[0] + marker_size[0],
                           cursor_position[1] + marker_size[1] - marker_frame_width),
                 color=color,
                 **kwargs)


def create_tiled_text_placeholder(text, size=(256, 256), color=(239,242,155), text_color=(161,225,252)):
    max_shift = 20
    side = max(size) + max_shift

    letter_size = 15
    n_repetitions = 30
    text += " " * len(text)
    n_rows = side // letter_size + 1

    place_holder = np.ones((side, side, 1), dtype=np.uint8) * np.array(color, dtype=np.uint8)
    for i in range(n_rows):
        put_text(place_holder,
                 "  " * i + text * n_repetitions,
                 (-n_rows * letter_size, letter_size * (1 + i)),
                 color=text_color,
                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                 shadow_shift=1,
                 thickness=1,
                 fontScale=1 / 2)

    l, t = np.random.randint(0, max_shift - 1), np.random.randint(0, max_shift - 1)
    return place_holder[t:t + size[1], l:l + size[0]]


if __name__ == "__main__":
    part_colors = np.array([[255, 0, 0],
                            [255, 85, 0],
                            [255, 170, 0],
                            [255, 0, 85],
                            [255, 0, 170],
                            [0, 255, 0],
                            [85, 255, 0],
                            [170, 255, 0],
                            [0, 255, 85],
                            [0, 255, 170],
                            [0, 0, 255],
                            [85, 0, 255],
                            [170, 0, 255],
                            [0, 85, 255],
                            [0, 170, 255],
                            [255, 255, 0],
                            [255, 255, 85],
                            [255, 255, 170],
                            [255, 0, 255],
                            [255, 85, 255],
                            [255, 170, 255],
                            [0, 255, 255],
                            [85, 255, 255],
                            [170, 255, 255],

                            [216, 95, 141],
                            [255, 204, 223],
                            [252, 225, 161],
                            [155, 242, 239],
                            [77, 67, 165],
                            [251, 58, 39],
                            [255, 142, 0],
                            [255, 228, 0],
                            [77, 150, 255],
                            [117, 185, 247],
                            [242, 125, 3],
                            [11, 83, 148],
                            [41, 107, 6]
                            ], dtype=np.uint8)[::-1, ::-1]

    im_example = cv2.imread("../data/vencer-sarthe-2015-15.jpg")
    im_example = cv2.resize(im_example, (416, 416))

    put_rectangle(im_example, (10, 10, 200, 200), tuple(list(map(int, part_colors[11]))), thickness=3)

    put_legend(im_example,
               ["dest", "rest", "test", "Abulabudabdab"],
               part_colors[:4],
               (20, im_example.shape[0] - 20))
    cv2.imshow("Example", im_example)
    cv2.waitKey()
    cv2.destroyAllWindows()
