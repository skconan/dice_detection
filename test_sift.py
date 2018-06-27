import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import constant as CONST


img1 = cv.imread(CONST.CONNECTED_LINE_PATH+'6/dice-6-line-1.jpg',0)
for i in range(15):
    
    # img1 = cv.imread(CONST.IMG_PATH+'roi/mask sift_screenshot_22.06.2018.png',0)
    img1 = cv.resize(img1,(60,60))
    img2 = cv.imread(CONST.CONNECTED_LINE_PATH+'6/dice-6-line-'+str(i)+'.jpg',0)
    try:
        img2 = cv.resize(img2,(60,60))
    except:
        continue
    # img1 = cv.imread('box.png',0) # queryImage
    # img2 = cv.imread('box_in_scene.png',0) # trainImage
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv.BFMatcher()
    # Match descriptors.
    # matches = bf.match(des1,des2)
    matches = bf.knnMatch(des1,des2,k=2)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    good = []
    for m,n in matches:
        print(m.distance,n.distance)
        if m.distance < 0.75*n.distance:
            good.append([m])
    print(len(good),len(matches))
    print(len(good)/len(matches))
    # Draw first 10 matches.
    img3 = img1.copy()
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good, img3,flags=2)
    plt.imshow(img3),plt.show()