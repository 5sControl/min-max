import numpy as np
import cv2

MIN_DETECTION_SIDE = 32

shape2size = lambda shape: np.array([shape[1], shape[0]])


def fill_by_content(canvas, image):
    target_size, object_size = shape2size(canvas.shape), shape2size(image.shape)

    object_size = tuple((object_size * (target_size / object_size).min()).astype(int))
    image = cv2.resize(image, object_size)

    shift = (target_size - object_size) // 2
    l, t, r, b = np.clip(np.stack((shift, object_size + shift), axis=0),
                         a_min=np.array([0, 0]),
                         a_max=target_size).reshape(4).astype(int)
    canvas[t:b, l:r] = image[:b - t, :r - l]
    return canvas


def letterbox(image, size=(640, 640), color=(114, 114, 114)):
    place_holder = np.ones(shape=(*size, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    x = fill_by_content(place_holder, image)
    return x


def split_and_letterbox(image, ltrb_boxes, size=(640, 640)):
    original_size, target_size = shape2size(image.shape), tuple(size)

    ltrb_boxes = np.minimum(ltrb_boxes.reshape(-1, 2, 2), original_size)
    patches = []
    for l, t, r, b in ltrb_boxes.reshape(-1, 4):
        if min(b - t, r - l) < MIN_DETECTION_SIDE:
            print(f"Warning! One of the specified regions are empty or too small. "
                  f"Image size: {original_size}, region: ({l}, {t}, {r}, {b})")
            patch = np.zeros(shape=(3, *target_size), dtype=np.uint8)
        else:
            patch = image[t:b, l:r]
            patch = letterbox(patch, target_size)
        patches.append(patch)
    batch = np.stack(patches, axis=0)
    return batch


# proper way is to use class with __call__ method
def get_letterbox_map(object_size, target_size):
    size = (object_size * (target_size / object_size).min()).astype(int)
    shift = (target_size - size) // 2
    lt_rb = np.clip(np.stack((shift, size + shift), axis=0),
                    a_min=np.array([0, 0]),
                    a_max=target_size).reshape(4).astype(int)
    shift = lt_rb[0]
    return lambda x: x / object_size * size + shift


def get_inverse_letterbox_map(region, target_size):
    object_size, target_size = np.array(region[1] - region[0]), np.array(target_size)
    size = (object_size * (target_size / object_size).min()).astype(int)
    shift = (target_size - size) // 2
    lt_rb = np.clip(np.stack((shift, size + shift), axis=0),
                    a_min=np.array([0, 0]),
                    a_max=target_size).reshape(2, 2).astype(int)
    shift = lt_rb[0].reshape(1, 1, 2)
    return lambda x: (x - shift) / np.array(size).reshape(1, 1, 2) * np.array(object_size).reshape(1, 1, 2) + \
                     + region[0].reshape(1, 1, 2)


def area(boxes):
    return np.prod(np.clip(boxes[..., 1, :] - boxes[..., 0, :], a_min=0, a_max=None), axis=-1)


def intersection_area_lt_rb(boxes_a, boxes_b):
    boxes_b = np.expand_dims(boxes_b, axis=-3)
    rb = np.minimum(boxes_a[..., 1, :], boxes_b[..., 1, :])
    lt = np.maximum(boxes_a[..., 0, :], boxes_b[..., 0, :])
    return np.prod(np.clip(rb - lt, a_min=0, a_max=None), axis=-1)


if __name__ == "__main__":
    ba = np.array([
        [0, 0, 100, 200],
        [0, 0, 100, 200],
        # [10, 10, 45, 50]
    ]).reshape(-1, 2, 2)

    bb = np.array([
        # [50, 50, 100, 200],
        # [50, 50, 100, 200],
        # [40, 40, 50, 50],
        # [50, 50, 100, 200],
        # [50, 50, 100, 200],
        [40, 40, 50, 50],

    ]).reshape(-1, 2, 2)

    print(area(bb).shape)
    print(intersection_area_lt_rb(bb, ba).shape)

    # print(intersection_area_lt_rb(bb, ba)==100)

    mp = get_inverse_letterbox_map(np.array([[0,0], [1920,1089]]), (640,640))
    print(ba.shape, mp(ba).shape)




    # import cv2
    # capture = cv2.VideoCapture("/home/ivan/Videos/videoplayback.mp4")
    # _, patch  = capture.read()
    #
    # cv2.imshow("a", patch)
    # cv2.imshow("b", letterbox(patch))
    # k = cv2.waitKey()
    # cv2.destroyAllWindows()
