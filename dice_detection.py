'''
    File name: test.py
    Author: zeabus2018
    Date created: 05/25/2018
    Python Version: 3.6.1
'''
import cv2 as cv
import csv
import numpy as np
from get_features_training import *

abs_path = 'C:/Users/skconan/Desktop/Workspace/dice/'
img_path = abs_path + 'images/'
templates_path = {
    'dice_2_0': img_path + 'dice_2_0.jpg',
    'dice_2_1': img_path + 'dice_2_1.jpg',
    'dice_5_0': img_path + 'dice_5_0.jpg',
    'dice_6_0': img_path + 'dice_6_0.jpg',
    'dice_6_1': img_path + 'dice_6_1.jpg',
}
templates_img = {}
pattern = []
pattern_predict = []
# def load_template():
#     global templates_path, templates_img
#     templates_img = {}
#     for key in templates_path.keys():
#         img = cv.imread(templates_path[key],0)
#         img = cv.resize(img,(50,50))
#         templates_img[key] = img
#     templates_img

def load_pattern():
    global pattern, pattern_predict
    with open(abs_path+'dataset.csv', 'r') as csvfile:
        csvfile.readline()
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            pattern.append([int(v) for v in row[:-1]])
            pattern_predict.append(int(row[-1]))

def matching(var):
    for (p,res) in zip(pattern,pattern_predict):
        print(var,p)
        if var == p:
            return res
    return None

def linear_regression(var):
    coeff = [
        0.48092891,
        0.75817333,
        0.48092891,
        0.75817333,
        0.37778931,
        0.75817333,
        0.48092891,
        0.75817333,
        0.48092891,
    ]
    y = 3.398028022833422
    for i in range(1,10):
        y += var['x'+str(i)]*coeff[i-1]
    return y

def get_circle(img_bgr):
    # load_template()
    side_per_radius = 9
    result = []
    img_result = img_bgr.copy()
    area_ratio_expected = 0.7
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(gray)
    _, mask = cv.threshold(equ, 230, 255, cv.THRESH_BINARY_INV)
    _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        if area_cnt <= 20:
            continue
        area_ratio = area_cnt / area_real
        if area_ratio < area_ratio_expected:
            continue
        cv.drawContours(img_result, cnt, -1, (0, 255, 0), 1)
        result.append([int(x), int(y), int(radius), int(x**2) + int(y**2)])

    result = sorted(result, key=itemgetter(2))
    circles = []
    for res in result:
        circles.append(res[:-1])

    radius = []
    for cir in circles:
        x, y, r = cir
        radius.append(r)

    r_avg = np.average(radius)
    dice_size = int(r_avg * side_per_radius)
    print('average: ', r_avg)
    img_list = []
    data_list = []
    for cir in circles:
        x, y, r = cir
        left, top = int(x - r - int(0.5 * r)) - \
            5, int(y - r - int(0.5 * r)) - 5
        right, bottom = left + dice_size + 10, top + dice_size + 10
        # cv.rectangle(img_result, (left,top),(right, bottom), (0,255,0), 1)
        # font = cv.FONT_HERSHEY_SIMPLEX
        # cv.putText(img_result, str(ct), (x,y), font,0.5, (0, 255, 255),1,cv.LINE_AA)
        # ct += 1
        roi = mask.copy()[top:bottom, left:right]
        roi_black = np.zeros(roi.shape, np.uint8)

        _, contours, _ = cv.findContours(
            roi, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        ct = 0
        # cv.drawContours(roi, contours, -1, (215, 255, 0), 2)
        for cnt in contours:
            (xx,yy), rr = cv.minEnclosingCircle(cnt)
            print(rr, r_avg - 1)
            area_real = math.pi * rr * rr
            if rr < r_avg - 1:
                continue
            area_cnt = cv.contourArea(cnt)

            area_ratio = area_cnt / area_real
            if area_ratio < area_ratio_expected:
                continue
            print('pass: ', rr)
            ct += 1
            cv.circle(roi_black, (int(xx), int(yy)),
                      int(rr), (255, 255, 255), -1)
        print(ct)
        # cv.imshow('roi',roi)
        # cv.waitKey(-1)
        if ct >= 2:
            img_list.append(roi_black)
            data_list.append([x,y,r])
    
    for (img,data) in zip(img_list,data_list):
        img = cv.resize(img,(60,60))
        var = get_point(img)
        # print(linear_regression(var))
        var = list(var.values())[:-1]
        print(var)
        dice = matching(var)
        if dice is not None:
            xxx,yyy,rrr = data
            if dice == 6:
                cv.circle(img_result,(int(xxx),int(yyy)),int(rrr),(0,255,0), -1)
            elif dice == 5:
                cv.circle(img_result,(int(xxx),int(yyy)),int(rrr),(255,0,0), -1)
            elif dice == 2:
                cv.circle(img_result,(int(xxx),int(yyy)),int(rrr),(0,0,255), -1)
        cv.imshow('roi', img)
        cv.waitKey(-1)

    cv.imshow('image_result', img_result)
    cv.imshow('equ', equ)
    cv.imshow('gray', gray)
    cv.imshow('mask', mask)
    cv.waitKey(-1)


def main():
    load_pattern()
    img = cv.imread(img_path + 'dice.jpg', 1)
    img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
    img_result = img.copy()

    get_circle(img)


if __name__ == '__main__':
    main()
    # matching()