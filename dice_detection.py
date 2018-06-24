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
from get_features_training import *
from matplotlib import pyplot as plt
import time

'''
    Rewrite Code: old variable
'''
pattern = []
pattern_predict = []
img_result = None
mask_extend_size = 80
keypoints = []
des_list = []
table_pose = []
pose = [
    [10,10],[10,30],[10,50],
    [30,10],[30,30],[30,50],
    [50,10],[50,30],[50,50]
]

'''
    Rewrite Code: old code
'''
def load_pattern():
    global pattern, pattern_predict
    with open(CONST.ABS_PATH + 'dataset.csv', 'r') as csvfile:
        csvfile.readline()
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            pattern.append([int(v) for v in row[:-1]])
            pattern_predict.append(int(row[-1]))


def matching(var):
    count = np.count_nonzero(var)
    if count < 3:
        return False
    for (p, res) in zip(pattern, pattern_predict):
        if var == p:
            return res
    return False

def load_keypoint():
    global keypoints, des_list
    img_list = []
    five = os.listdir(CONST.CONNECTED_LINE_PATH + '5/')
    six = os.listdir(CONST.CONNECTED_LINE_PATH + '6/')

    for name in five:
        img = cv.imread(CONST.CONNECTED_LINE_PATH + '5/' + name, 1)
        img = cv.resize(img,(60,60))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        # keypoints.append([5, kp])
        des_list.append([5, des, kp, gray])
        # img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imshow('img',img)
        # cv.waitKey(-1)

    for name in six:
        img = cv.imread(CONST.CONNECTED_LINE_PATH + '6/' + name, 1)
        img = cv.resize(img,(60,60))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        # keypoints.append([6, kp])
        des_list.append([6, des, kp ,gray])


def sift_matching(mask):
    global des_list
    accuracy_max = 0.0
    predict_dice = 0
    print('============',len(des_list),'============')
    
    for (dice, des,kp,img) in des_list:
        sift = cv.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(mask, None)
        kp2, des2 = sift.detectAndCompute(img, None)
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks = 50)
        
        bf = cv.BFMatcher()
        # flann = cv.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(des1,des2,k=2)
        cv.imshow("des1",des1)
        cv.imshow("des2",des2)
        cv.imshow("mask1",mask)
        cv.imshow("mask2",img)
        # cv.waitKey(-1)
        matches = bf.knnMatch(des1, des2, k=2)
        if len(matches) == 0 or len(matches[0]) < 2:
            continue
        good = []
        print(matches)
        for m, n in matches:
            print(m.distance,n.distance)
            if m.distance < 0.7 * n.distance:
                good.append([m])
        print(len(good),len(matches))
        accuracy = float(len(good)) / len(matches)
        if  accuracy >= accuracy_max:
            accuracy_max = accuracy
            predict_dice = dice
            img3 = mask.copy()
            img3 = cv.drawMatchesKnn(mask,kp1,img,kp2,good, img3,flags=4)
            # plt.imshow(img3)
            print(dice,accuracy)
            cv.imshow('mask sift',mask)
            cv.imshow('im',img3)
            # cv.waitKey(-1)
            # plt.show()
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
    global table_pose
    result = [0]*9
    mask = cv.resize(mask,(60,60))    
    mask_result = mask.copy()
    mask_result.fill(0)
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    res = []
    x_min = 100000
    x_max = -1
    y_min = 100000
    y_max = -1
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        area_ratio = area_cnt / area_real
        if (not CONST.POINT_AREA_LOWER <= area_cnt <= CONST.POINT_AREA_UPPER or
                area_ratio < CONST.EXPECTED_AREA_RATIO):
            continue
        x, y, radius = int(x), int(y), int(radius)
        print(result)
        index = table_pose[int(x/2)][int(y/2)]
        # x, y = pose[table_pose[int(x/2)][int(y/2)]]
        print(index)
        result[index] = 1
        print(x,y)
        radius = 6
        a,b,c,d = int(x-radius/2.0),int(y-radius/2.0),int(x+radius/2.0),int(y+radius/2.0)
        cv.circle(mask_result,(x,y),radius,(255,255,255),-1)
        # cv.imwrite(CONST.IMG_PATH + 'roi/dice_'+ str(time.time())+'.jpg', mask_result)

        # if a < 25:
        #     if b < 15:
        #         cv.rectangle(mask_result,(a,b),(c,d),(255,255,255),-1)
        #     else:
        #         cv.rectangle(mask_result,(a,b),(c,d),(155,155,155),-1)
                
        # else:
        #     if b < 15:
        #         cv.circle(mask_result,(x,y),radius,(255,255,255),-1)
        #     else:
        #         cv.circle(mask_result,(x,y),radius,(155,155,155),-1)
                
        # if a < x_min:
        #     x_min = a
        # if b < y_min:
        #     y_min = b
        # if c > x_max:
        #     x_max = c
        # if d > y_max:
        #     y_max = c
        # i = [x-30,y]
        # j = [x+30,y]
        # pts = np.array([i, j], np.int32)
        # pts = pts.reshape((-1, 1, 2))
        # cv.polylines(mask_result, [pts], True, (155, 155, 155), 1)
                
    #     res.append([x, y])
    # if len(res) < 3:
    #     return None, None
    # cv.rectangle(mask_result,(x_min,y_min),(x_max,y_max),(255,255,255),1)
        
    # for i in res:
    #     for j in res:
    #         pts = np.array([i, j], np.int32)
    #         pts = pts.reshape((-1, 1, 2))
    #         cv.polylines(mask_result, [pts], True, (155, 155, 155), 1)

    # _, mask = cv.threshold(mask_result, 127, 255, cv.THRESH_BINARY)
    # _, contours, hierarchy = cv.findContours(
    #     mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # if hierarchy is None:
    #     return None, None

    # hierarchy = hierarchy[0]

    # x, y, width, height = 0, 0, 0, 0
    # if len(contours) > 0:
    #     for (cnt, h) in zip(contours, hierarchy):
    #         if h[3] == -1:
    #             x, y, width, height = cv.boundingRect(cnt)
    #             cv.drawContours(mask_result, cnt, -1, (155, 155, 155), 2)
    # else:
    #     return None, None
    # rows, cols = mask.shape
    # if x + y + width + height > 0:
    #     pts1 = np.float32(
    #         [[x, y], [x + width, y], [x, y + height], [x + width, y + height]])
    #     pts2 = np.float32([[0, 0], [CONST.DICE_SIZE, 0], [0, CONST.DICE_SIZE], [
    #                       CONST.DICE_SIZE, CONST.DICE_SIZE]])
    #     M = cv.getAffineTransform(pts1[:-1], pts2[:-1])
    #     mask = cv.warpAffine(mask, M, (cols, rows))
    # else:
    #     return None, None
    # mask = mask_result
    # cv.imshow('mask',mask)
    # cv.waitKey(-1)
    # predict_dice, accuracy = sift_matching(mask)
    # print(accuracy)
    # if accuracy >= 0.5:
        # cv.waitKey(-1)
    
    # return predict_dice, accuracy

    # cv.imshow('mask_point', mask)
    # cv.imshow('mask_result1', mask_result)
    # cv.waitKey(-1)

    # cv.waitKey(-1)
    # print(result)
    dice = matching(result)
    if not dice == False:
        pass
        # cv.imwrite(CONST.IMG_PATH + 'roi/dice_' + str(dice)+'_'+ str(time.time())+'.jpg', mask_result)
    return result

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
            try:
                roi = cv.resize(roi, (CONST.DICE_SIZE, CONST.DICE_SIZE))
            except:
                continue
            roi = mask.copy()[top:bottom, left:right]

            # dice, accuracy = what_is_point(roi)
            t = time.time()
            cv.imwrite(CONST.IMG_PATH + 'roi/dice_'+ str(t)+'.jpg', roi)
            roi = mask.copy()[top-20:bottom+20, left-20:right+20]
            cv.imwrite(CONST.IMG_PATH + 'roi/dice_'+ str(t)+'_1.jpg', roi)

            point = what_is_point(roi)
            dice = matching(point)
            # if dice is None:
            #     continue
            # if accuracy < 0.5:
            #     continue
            if dice == False:
                continue
            center = (int((left + right) / 2) - int(mask_extend_size / 2),
                      int((top + bottom) / 2) - int(mask_extend_size / 2))
            
            data_list.append([center, radius, dice, np.count_nonzero(point)])
    data_dict = remove_redundant_dice(data_list)
    return data_dict


def remove_redundant_dice(data):
    result = { '5': None, '6': None}
    data = sorted(data, key=itemgetter(2))
    for d in data:
        index_dict = str(d[2])
        if result[index_dict] is None or (result[index_dict][3] < d[3] or (result[index_dict][3] == d[3] and result[index_dict][3] > d[3])):
            result[index_dict] = d

    return result


# def remove_redundant_dice(data):
#     result = {'5': None, '6': None}
#     data = sorted(data, key=itemgetter(2))
#     for d in data:
#         print(d)
#         index_dict = str(d[2])
#         if result[index_dict] is None or result[index_dict][3] < d[3]:
#             result[index_dict] = d

#     return result


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
        # cv.circle(result, center, radius, (255, 255, 255), -1)
        cv.drawContours(result,cnt,-1,(255,255,255),-1)
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
    global img_result, table_pose
    img = pre_processing(img)
    mask_th = find_mask_threshold(img)
    mask_circles, circles = find_point(mask_th)
    # cv.imshow('mask_th', mask_th)
    dice_dict = find_dice(mask_circles, circles)
    cv.imshow('mask_cir', mask_circles)
    mask_dice(dice_dict)
    cv.imshow('result', img_result)
    cv.waitKey(1)


def main():
    global img_result,table_pose
    load_pattern()
    table_pose = create_table_pose()
    load_keypoint()
    # for i in range(1,10):
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
