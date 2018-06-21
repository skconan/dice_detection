'''
    File name: dice_detection.py
    Author: zeabus2018
    Date created: 05/25/2018
    Python Version: 3.6.1
'''
import os
import csv
import cv2 as cv
from lib import *
import numpy as np
from operator import itemgetter

'''
    Rewrite Code: old variable
'''
# pattern = []
# pattern_predict = []
img_result = None
mask_extend_size = 80
keypoints = []
des_list = []
'''
    Rewrite Code: old code
'''
# def load_pattern():
#     global pattern, pattern_predict
#     with open(CONST.ABS_PATH + 'dataset.csv', 'r') as csvfile:
#         csvfile.readline()
#         reader = csv.reader(csvfile, delimiter=',')
#         for row in reader:
#             pattern.append([int(v) for v in row[:-1]])
#             pattern_predict.append(int(row[-1]))


# def matching(var):
#     for (p, res) in zip(pattern, pattern_predict):
#         if var == p:
#             return res
#     return None


# def may_be_dice(mask):
#     result = get_point(mask)
#     is_one = np.count_nonzero(result)
#     if is_one >= 3:
#         return True, result
#     else:
#         return False, None
def load_keypoint():
    global keypoints, des_list
    img_list = []
    five = os.listdir(CONST.CONNECTED_LINE_PATH + '5/')
    six = os.listdir(CONST.CONNECTED_LINE_PATH + '6/')

    for name in five:
        img = cv.imread(CONST.CONNECTED_LINE_PATH + '5/' + name, 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        keypoints.append([5, kp])
        des_list.append([5, des])
        # img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imshow('img',img)
        # cv.waitKey(-1)

    for name in six:
        img = cv.imread(CONST.CONNECTED_LINE_PATH + '6/' + name, 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        keypoints.append([6, kp])
        des_list.append([6, des])


def sift_matching(mask):
    global des_list
    accuracy_max = 0.0
    predict_dice = 0

    for (dice, des) in des_list:
        sift = cv.xfeatures2d.SIFT_create()

        kp, des1 = sift.detectAndCompute(mask, None)
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        accuracy = float(len(good)) / len(matches)
        if accuracy > accuracy_max:
            accuracy_max = accuracy
            predict_dice = dice
    print(predict_dice, accuracy_max)
    return predict_dice, accuracy_max


def get_region(dice_size, x, y, radius):
    a, b = 0, 0
    region = []
    top = int(y - radius - int(0.5 * radius)) - a
    left = int(x - radius - int(0.5 * radius)) - a
    bottom = top + dice_size + b
    right = left + dice_size + b
    region.append([top, left, bottom, right])

    bottom = int(y + radius + int(0.5 * radius)) + a
    right = int(x + radius + int(0.5 * radius)) + a
    top = bottom - dice_size - b
    left = right - dice_size - b
    region.append([top, left, bottom, right])

    return region


def what_is_point(mask):
    mask_result = mask.copy()
    mask_result.fill(0)
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    res = []
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        area_ratio = area_cnt / area_real
        if (not CONST.POINT_AREA_LOWER <= area_cnt <= CONST.POINT_AREA_UPPER or
                area_ratio < CONST.EXPECTED_AREA_RATIO):
            continue
        x, y, radius = int(x), int(y), int(radius)
        cv.circle(mask_result, (x, y), radius, (255, 255, 255), -1)

        res.append([x, y])

    for i in res:
        for j in res:
            pts = np.array([i, j], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv.polylines(mask_result, [pts], True, (255, 255, 255), 2)

    _, mask = cv.threshold(mask_result, 127, 255, cv.THRESH_BINARY)
    _, contours, hierarchy = cv.findContours(
        mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    if hierarchy is None:
        return None, None

    hierarchy = hierarchy[0]

    x, y, width, height = 0, 0, 0, 0
    if len(contours) > 0:
        for (cnt, h) in zip(contours, hierarchy):
            if h[3] == -1:
                x, y, width, height = cv.boundingRect(cnt)
                cv.drawContours(mask_result, cnt, -1, (155, 155, 155), 2)
    else:
        return None, None
    rows, cols = mask.shape
    if x + y + width + height > 0:
        pts1 = np.float32(
            [[x, y], [x + width, y], [x, y + height], [x + width, y + height]])
        pts2 = np.float32([[0, 0], [CONST.DICE_SIZE, 0], [0, CONST.DICE_SIZE], [
                          CONST.DICE_SIZE, CONST.DICE_SIZE]])
        M = cv.getAffineTransform(pts1[:-1], pts2[:-1])
        mask = cv.warpAffine(mask, M, (cols, rows))
        cv.imwrite(CONST.IMG_PATH + 'roi/dice_' + '_' +
                       str(x + width+ y + height) + '.jpg', mask)
        predict_dice, accuracy = sift_matching(mask)
        return predict_dice, accuracy
    else:
        return None, None

    # cv.imshow('mask_point', mask)
    # cv.imshow('mask_result1', mask_result)
    # cv.waitKey(-1)

    # cv.waitKey(-1)
    # print(result)


def find_dice(mask, circles):
    global mask_extend_size
    data_list = []

    for cir in circles:
        # Interger
        x, y, radius = cir
        dice_size = int(radius * CONST.SIDE_PER_RADIUS)
        region_list = get_region(dice_size, x, y, radius)

        for region in region_list:
            top, left, bottom, right = region
            roi = mask.copy()[top:bottom, left:right]

            roi = cv.resize(roi, (CONST.DICE_SIZE, CONST.DICE_SIZE))

            '''
                Rewrite Code: old code
            '''
            # cv.imshow('roi', roi)
            # is_dice, point = may_be_dice(roi)

            # if not is_dice:
            #     continue

            # count = np.count_nonzero(point)
            # dice = matching(point[:-1])
            # if dice is None:
            #     continue
            # print(count, dice)
            # if count > dice:
            #     continue
            dice, accuracy = what_is_point(roi)
            if dice is None:
                continue
            if accuracy < 0.75:
                continue
            center = (int((left + right) / 2) - int(mask_extend_size / 2),
                      int((top + bottom) / 2) - int(mask_extend_size / 2))
            
            data_list.append([center, radius, dice, accuracy])
    data_dict = remove_redundant_dice(data_list)
    return data_dict


'''
    old code
'''
# def remove_redundant_dice(data):
#     result = {'2': None, '5': None, '6': None}
#     data = sorted(data, key=itemgetter(3))
#     for d in data:
#         index_dict = str(d[3])
#         if result[index_dict] is None or (result[index_dict][5] < d[5] or (result[index_dict][5] == d[5] and result[index_dict][1] > d[5])):
#             result[index_dict] = d

#     return result


def remove_redundant_dice(data):
    result = {'5': None, '6': None}
    data = sorted(data, key=itemgetter(2))
    for d in data:
        print(d)
        index_dict = str(d[2])
        if result[index_dict] is None or result[index_dict][3] < d[3]:
            result[index_dict] = d

    return result


def is_circle(area_ratio, len_approx, hierarchy):
    area_ratio_expected = 0.65
    len_approx_expected = 7
    hierarchy_expected = [-1, -1]
    hierarchy = list(hierarchy[2:]) 

    if  (area_ratio < area_ratio_expected or 
        len_approx < len_approx_expected or 
        not (hierarchy == hierarchy_expected)):
        print('false')
        return False
    print('true')
    return True


def find_point(img_bin):
    r, c = img_bin.shape
    result = np.zeros((r, c), np.uint8)
    result_data = []
    _, cnts, hierarchy = cv.findContours(
        img_bin, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]

    for (cnt, h) in zip(cnts, hierarchy):
        area_cnt = cv.contourArea(cnt)
        if len(cnt) < 5 or area_cnt > (r * c) * 0.05:
            continue
        print('state 1')
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        len_approx = len(approx)

        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))

        area_real = math.pi * radius * radius
        area_ratio = area_cnt / area_real

        if not is_circle(area_ratio, len_approx, h):
            continue
        print('state 2')
        
        radius = int(radius)
        result_data.append([x, y, radius])
        cv.circle(result, center, radius, (255, 255, 255), -1)

    return result, result_data


'''
    Rewrite Code: Now, Not use
'''
# def find_sure_bg(gray):
#     gray_map = color_mapping(gray)
#     b,g,r = cv.split(gray_map)
#     lowerb_blue = np.array([240,0,0],np.uint8)
#     upperb_blue = np.array([255,120,0],np.uint8)
#     blue = inRangeBGR(gray_map, lowerb_blue, upperb_blue)

#     lowerb_green = np.array([0,240,0],np.uint8)
#     upperb_green = np.array([0,255,120],np.uint8)
#     green = inRangeBGR(gray_map, lowerb_green, upperb_green)

#     lowerb_yellow = np.array([0,240,240],np.uint8)
#     upperb_yellow = np.array([0,255,255],np.uint8)
#     yellow = inRangeBGR(gray_map, lowerb_yellow, upperb_yellow)

#     sure_bg = cv.bitwise_or(green,blue,mask=None)
#     sure_bg = cv.bitwise_or(yellow,sure_bg,mask=None)
#     cv.imshow('blue',blue)
#     cv.imshow('green',green)
#     cv.imshow('yellow',yellow)
#     cv.imshow('sure_bg',sure_bg)

#     cv.imshow('map',gray_map)
#     return sure_bg


def pre_processing(img_bgr):
    result = cv.GaussianBlur(img_bgr, (5, 5), 0)
    return result


def find_mask_threshold(img_bgr):
    global mask_extend_size
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    r, c = gray.shape

    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v_avg = np.average(v)

    _, mask = cv.threshold(gray, v_avg / 1.5, 255, cv.THRESH_BINARY_INV)
    erode = cv.erode(mask, get_kernel('rect', (3, 3)))
    mask = cv.dilate(erode, get_kernel('rect', (3, 3)))

    tmp = np.zeros((r + 80, c + 80), np.uint8)
    tmp[40:-40, 40:-40] = mask
    mask = tmp

    return mask


def mask_dice(dict):
    global img_result, mask_extend_size
    color = {'2': (255, 0, 0), '5': (0, 255, 0), '6': (0, 0, 255)}
    for d in dict.keys():
        if dict[d] is None:
            continue
        center, radius, dice, accuracy = dict[d]
        x,y = center
        x -= int(mask_extend_size/2)
        y -= int(mask_extend_size/2)
        cv.circle(img_result, center, radius, color[d], -1)
        cv.circle(img_result, center, radius * 9, color[d], 2)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_result, str(dice), center,
                   font, 1, color[d], 2, cv.LINE_AA)
        cv.imshow('image_result', img_result)


def run(img):
    global img_result
    img = pre_processing(img)
    mask_th = find_mask_threshold(img)
    mask_circles, circles = find_point(mask_th)
    cv.imshow('mask_th', mask_th)
    dice_dict = find_dice(mask_circles, circles)
    cv.imshow('mask_cir', mask_circles)
    mask_dice(dice_dict)
    cv.imshow('result', img_result)
    cv.waitKey(1)


def main():
    global img_result
    # load_pattern()
    load_keypoint()
    cap = cv.VideoCapture(CONST.VDO_PATH + 'dice_02.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            continue
        frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        img_result = frame.copy()
        run(frame)
        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
