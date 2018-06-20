'''
    File name: dice_detection.py
    Author: zeabus2018
    Date created: 05/25/2018
    Python Version: 3.6.1
'''
import cv2 as cv
import csv
import numpy as np
from lib import *
from operator import itemgetter
pattern = []
pattern_predict = []
img_result = None

def load_pattern():
    global pattern, pattern_predict
    with open(CONST.ABS_PATH + 'dataset.csv', 'r') as csvfile:
        csvfile.readline()
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            pattern.append([int(v) for v in row[:-1]])
            pattern_predict.append(int(row[-1]))


def matching(var):
    for (p, res) in zip(pattern, pattern_predict):
        if var == p:
            return res
    return None


def get_circle_in_frame(mask):
    area_ratio_expected = 0.6
    circles = []
    radius_list = []
    mask = cv.dilate(mask,np.array([[1,1,1],[1,1,1],[1,1,1]],np.uint8))
    cv.imshow('mask_before_find_circle',mask)
    _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        area_ratio = area_cnt / area_real
        # if area_cnt <= 15 or area_ratio < area_ratio_expected:
        #     continue
        x, y, radius = int(x), int(y), int(radius)
        circles.append([x, y, radius])
        radius_list.append(radius)
    radius_avg = np.average(radius_list)

    return circles, radius_avg


def may_be_dice(mask):
    result = get_point(mask)
    is_one = np.count_nonzero(result)
    if is_one >= 2:
        return True, result
    else:
        return False, None

# def check_in_frame(top, left, bottom, right):
#     global width, height
#     if (0 <= top < height or     
#         0 <= bottom < height or
#         0 <= left < width or
#         0 <= right < width):
#         return True
#     return False

def get_region(dice_size, x, y, radius):
    a,b = 0,0
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


def get_dice_position(mask, circles, radius_avg):
    dice_size = int(radius_avg * CONST.SIDE_PER_RADIUS)
    data_list = []

    for cir in circles:
        x, y, radius = cir
        region_list = get_region(dice_size, x, y, radius)
        x, y, radius = int(x), int(y), int(radius)
        for region in region_list:
            top, left, bottom, right = region
            roi = mask.copy()[top:bottom, left:right]
            center = (int((left + right) / 2), int((top + bottom) / 2))
            try:
                roi = cv.resize(roi, (CONST.DICE_SIZE, CONST.DICE_SIZE))
            except:
                continue
            is_dice, point = may_be_dice(roi)
            if not is_dice:
                continue
            cv.imwrite(CONST.IMG_PATH+'img_'+str(top)+'_'+str(left)+'_'+str(bottom)+'_'+str(right)+'.jpg',roi)
            count = np.count_nonzero(point)
            dice = matching(point[:-1])
            if dice is None:
                continue
            if count > dice:
                continue
            data_list.append([x, y, radius, dice, point, count, center])
    data_dict = remove_redundant_dice(data_list)
    return data_dict


def remove_redundant_dice(data):
    result = {'2': None, '5': None, '6': None}
    data = sorted(data, key=itemgetter(3))
    for d in data:
        index_dict = str(d[3])
        if result[index_dict] is None or (result[index_dict][5] < d[5] or (result[index_dict][5] == d[5] and result[index_dict][1] > d[5])):
            result[index_dict] = d

    return result


def mask_dice(img, dict):
    global img_result
    color = {'2': (255, 0, 0), '5': (0, 255, 0), '6': (0, 0, 255)}
    for d in dict.keys():
        if dict[d] is None:
            continue
        x, y, radius, dice, point, count, center = dict[d]
        cv.circle(img_result, center, radius, color[d], -1)
        cv.circle(img_result, center, radius*9, color[d], 2)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_result,str(dice),(x,y), font, 1,color[d],2,cv.LINE_AA)
        print(dice,point)
        cv.imshow('image_result', img_result)
        cv.waitKey(2000)
 

def find_certain_circle(img_bin):
    erode = cv.erode(img_bin,get_kernel('plus',(3,3)))
    dilate = cv.dilate(erode,get_kernel('plus',(3,3)))
    area_ratio_expected = 0.65
    res = img_bin.copy()
    res.fill(0)
    res_cnt = res.copy()
    _,cnts,hierarchy = cv.findContours(dilate,cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # print(len(hierarchy[0]),len(cnts))
    hierarchy = hierarchy[0]
    cv.drawContours(res_cnt,cnts,-1,(255,255,255),1)
    for (cnt,h) in zip(cnts,hierarchy):
        # print(h)
        if len(cnt) < 5:
            continue
        approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True)
        # piab
        (x,y),radius = cv.minEnclosingCircle(cnt)
        
        ellipse = cv.fitEllipse(cnt)
        # print(ellipse)
        center = (int(x),int(y))
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        area_ratio = area_cnt / area_real
        if area_ratio < area_ratio_expected or len(approx) < 7 or not h[2] == -1 or not h[3] == -1:
            continue
       
        # print(check,-len(h))
        
        cv.circle(res, center, int(radius), (255,255,255), -1)

    return res

def find_sure_bg(gray):
    gray_map = color_mapping(gray)
    b,g,r = cv.split(gray_map)
    lowerb_blue = np.array([240,0,0],np.uint8)
    upperb_blue = np.array([255,120,0],np.uint8)
    blue = inRangeBGR(gray_map, lowerb_blue, upperb_blue)

    lowerb_green = np.array([0,240,0],np.uint8)
    upperb_green = np.array([0,255,120],np.uint8)
    green = inRangeBGR(gray_map, lowerb_green, upperb_green)

    lowerb_yellow = np.array([0,240,240],np.uint8)
    upperb_yellow = np.array([0,255,255],np.uint8)
    yellow = inRangeBGR(gray_map, lowerb_yellow, upperb_yellow)

    sure_bg = cv.bitwise_or(green,blue,mask=None)
    sure_bg = cv.bitwise_or(yellow,sure_bg,mask=None)
    cv.imshow('blue',blue)
    cv.imshow('green',green)
    cv.imshow('yellow',yellow)
    cv.imshow('sure_bg',sure_bg)

    cv.imshow('map',gray_map)
    return sure_bg

def run(img):
    img = cv.GaussianBlur(img,(5,5),0)
    b,g,r = cv.split(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    h,s,v = cv.split(hsv)
    v_avg = np.average(v)
    print(v_avg)
    _, mask = cv.threshold(gray, v_avg/1.5, 255, cv.THRESH_BINARY_INV)
    mask_result = find_certain_circle(mask)
    cv.imshow('mask', mask)
    cv.imshow('mask_result', mask_result)
    circles, radius_avg = get_circle_in_frame(mask_result)
  
    if not radius_avg >=0  :
        print('='*20)
        print(radius_avg)
        return
    
    dice_dict = get_dice_position(mask, circles, radius_avg)
    mask_dice(img, dice_dict)
    

if __name__ == '__main__':
    load_pattern()
    cap = cv.VideoCapture(CONST.VDO_PATH+'dice_02.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            continue
        frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        img_result = frame.copy()
        run(frame)

        if img_result is not None:
            cv.imshow('img_result',img_result)
        k = cv.waitKey(1) & 0xff
        if k== ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()
