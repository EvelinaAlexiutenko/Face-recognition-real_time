from make_smaller import make_smaller
from merge_image import merge_image


def augment_selfie(img, backg_img):
    img = make_smaller(img)
    res = merge_image(backg_img, img, 150, 150)
    return res
