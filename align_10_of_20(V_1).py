import numpy as np
import scipy.misc as sc

def align(image_obj, g_coord):
    g_row, g_col = g_coord

    w, h = image_obj.shape
    w = w - w % 3
    h = h - h % 3

    image_obj = image_obj[0:w, 0:h]

    w = w // 3

    b_row, b_col = w // 20, h // 20
    b_x, b_y = 0, 0
    r_row, r_col = 2 * w + w // 20, h // 20
    r_x, r_y = 0, 0

    gr, gc = w + w // 20, h // 20

    b_img = image_obj[0:w, ...]
    g_img = image_obj[w:2 * w, ...]
    r_img = image_obj[2 * w:3 * w, ...]

    b_img = b_img[w // 20:19 * w // 20, h // 20:19 * h // 20]
    #sc.imsave('b_img.png', b_img)
    g_img = g_img[w // 20:19 * w // 20, h // 20:19 * h // 20]
    #sc.imsave('g_img.png', g_img)
    r_img = r_img[w // 20:19 * w // 20, h // 20:19 * h // 20]
    #sc.imsave('r_img.png', r_img)

    w, h = b_img.shape

    b_diff = np.sum(np.square(b_img - g_img)) / (w * h)
    for x in range(-20, 21):
        for y in range(-20, 21):
            p1 = b_img
            p2 = g_img
            if (x < 0):
                p1 = p1[-x:, ...]
                p2 = p2[:w + x, ...]
            else:
                p1 = p1[:w - x, ...]
                p2 = p2[x:, ...]
            if (y < 0):
                p1 = p1[..., -y:]
                p2 = p2[..., :h + y]
            else:
                p1 = p1[..., :h - y]
                p2 = p2[..., y:]

            sum = np.sum(np.square(p1 - p2)) / ((w - abs(x)) * (h - abs(y)))
            if (sum < b_diff):
                b_diff = sum
                b_x = x
                b_y = y

    r_diff = np.sum(np.square(r_img - g_img)) / (w * h)
    for x in range(-20, 21):
        for y in range(-20, 21):
            p1 = r_img
            p2 = g_img
            if (x < 0):
                p1 = p1[-x:, ...]
                p2 = p2[:w + x, ...]
            else:
                p1 = p1[:w - x, ...]
                p2 = p2[x:, ...]
            if (y < 0):
                p1 = p1[..., -y:]
                p2 = p2[..., :h + y]
            else:
                p1 = p1[..., :h - y]
                p2 = p2[..., y:]

            sum = np.sum(np.square(p1 - p2)) / ((w - abs(x)) * (h - abs(y)))
            if (sum < r_diff):
                r_diff = sum
                r_x = x
                r_y = y

    return (image_obj, (b_row + (g_row - gr) - b_x, b_col + (g_col - gc) - b_y), \
                       (r_row + (g_row - gr) - r_x, r_col + (g_col - gc) - r_y))
