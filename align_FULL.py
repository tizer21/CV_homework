import numpy as np
from skimage import transform
from skimage import measure


def mse_metric(img1, img2):
    w, h = img1.shape
    return np.sum(np.square(img1 - img2)) / (w * h)


def normalized_cross_correlation(img1, img2):
    return np.sum(img1 * img2) / np.sqrt(np.sum(np.square(img1) * np.square(img2)))


def cropping(img, start_coord, end_coord):
    return img[start_coord[0]: end_coord[0], start_coord[1]: end_coord[1]]


def get_channels(img):
    w, h = img.shape
    w = w // 3

    blue_channel = cropping(img, [0, 0], [w, h])
    green_channel = cropping(img, [w, 0], [2 * w, h])
    red_channel = cropping(img, [2 * w, 0], [3 * w, h])

    return w, h, [blue_channel, green_channel, red_channel]


def img_offset(img1, img2, cx, cy, scatter):
    dx = dy = 0
    diff = measure.compare_mse(img1, img2)

    w, h = img1.shape

    for x in range(cx - scatter, cx + scatter + 1):
        for y in range(cy - scatter, cy + scatter + 1):
            i1_x_fr, i2_x_fr = 0, abs(x)
            i1_x_to, i2_x_to = w - abs(x), w
            if x < 0:
                i1_x_fr, i2_x_fr = i2_x_fr, i1_x_fr
                i1_x_to, i2_x_to = i2_x_to, i1_x_to

            i1_y_fr, i2_y_fr = 0, abs(y)
            i1_y_to, i2_y_to = h - abs(y), h
            if y < 0:
                i1_y_fr, i2_y_fr = i2_y_fr, i1_y_fr
                i1_y_to, i2_y_to = i2_y_to, i1_y_to

            ndiff = measure.compare_mse(img1[i1_x_fr: i1_x_to, i1_y_fr: i1_y_to],\
                               img2[i2_x_fr: i2_x_to, i2_y_fr: i2_y_to])

            if ndiff < diff:
                diff = ndiff
                dx, dy = x, y

    return dx, dy


def pyramidal_combination(img1, img2):
    if img1.shape[0] <= 500:
        return img_offset(img1, img2, 0, 0, 10)

    n_img1 = transform.rescale(img1, 0.5)
    n_img2 = transform.rescale(img2, 0.5)

    offset = pyramidal_combination(n_img1, n_img2)
    n_offset = img_offset(img1, img2, 2 * offset[0], 2 * offset[1], 2)

    return n_offset


def basic_offset(img, g_coord):
    image_obj = np.asarray(img)

    w, h, channels = get_channels(image_obj)
    dw, dh = w // 20, h // 20

    for i in range(3):
        channels[i] = channels[i][dw: w - dw, dh: h - dh]

    b_offset = img_offset(channels[0], channels[1], 0, 0, 15)
    r_offset = img_offset(channels[2], channels[1], 0, 0, 15)

    align_img = img

    b_coord = (g_coord[0] - w - b_offset[0], g_coord[1] - b_offset[1])
    r_coord = (g_coord[0] + w - r_offset[0], g_coord[1] - r_offset[1])

    return align_img, b_coord, r_coord


def pyramidal_offset(img, g_coord):
    image_obj = np.asarray(img)

    w, h, channels = get_channels(image_obj)
    dw, dh = w // 20, h // 20

    for i in range(3):
        channels[i] = channels[i][dw: w - dw, dh: h - dh]

    b_offset = pyramidal_combination(channels[0], channels[1])
    r_offset = pyramidal_combination(channels[2], channels[1])

    align_img = img

    b_coord = (g_coord[0] - w - b_offset[0], g_coord[1] - b_offset[1])
    r_coord = (g_coord[0] + w - r_offset[0], g_coord[1] - r_offset[1])

    return align_img, b_coord, r_coord


def align(img, g_coord):
    if img.shape[1] > 500:
        return pyramidal_offset(img, g_coord)
    else:
        return basic_offset(img, g_coord)
