import cv2 as cv
import numpy as np


def nothing(x):
    pass


def color_range():
    img = np.zeros((500, 500), np.uint8)
    img = cv.merge((img, img, img))
    cv.namedWindow('image')
    cv.createTrackbar('H', 'image', 0, 179, nothing)
    cv.createTrackbar('S', 'image', 0, 255, nothing)
    cv.createTrackbar('V', 'image', 0, 255, nothing)

    while True:

        if img is None:
            continue
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        h_value = cv.getTrackbarPos('H', 'image')
        s_value = cv.getTrackbarPos('S', 'image')
        v_value = cv.getTrackbarPos('V', 'image')
        h.fill(h_value)
        s.fill(s_value)
        v.fill(v_value)
        hsv = cv.merge((h, s, v))
        res_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('image', res_bgr)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
    color_range()
