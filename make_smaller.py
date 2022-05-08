import cv2


def make_smaller(img, scale_percent=15):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    dsize = (width, height)

    # resize image
    small_image = cv2.resize(img, dsize)
    return small_image
