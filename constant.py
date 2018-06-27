'''
    File name: constant.py
    Author: zeabus2018
    Date created: 05/27/2018
    Python Version: 3.6.1
'''
import math

'''
    REAL SIZE
        SIDE by SIDE = 9x9 
        RADIUS = 1
    DICE_SIZE must be divided by 3 
'''

DICE_SIZE = 90
SIDE_PER_RADIUS = 9.0
POINT_RADIUS = float(DICE_SIZE) / SIDE_PER_RADIUS
POINT_AREA_LOWER = math.pi * (POINT_RADIUS-1.5)**2
POINT_AREA_UPPER = math.pi * (POINT_RADIUS+5)**2
ABS_PATH = 'C:/Users/skconan/Desktop/Workspace/dice/'
IMG_PATH = ABS_PATH + 'images/'
DATA_SET_PATH = IMG_PATH + 'data_set/'
CONNECTED_LINE_PATH = DATA_SET_PATH + 'connected_line/'
POINT_PATH = DATA_SET_PATH + 'point/'
VDO_PATH = ABS_PATH + 'videos/'
EXPECTED_AREA_RATIO = 0.65
