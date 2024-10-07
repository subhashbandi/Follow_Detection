import tensorflow as tf
import cv2 as cv
import os

path = 'D:\Coding\Follow_Detection\Sample'

for root, dir, files in os.walk(path):
    for file in files:
        if file.endswith('.jpeg'):
            img_path = os.path.join(root, file)
            img = cv.imread(img_path, cv.IMREAD_COLOR)
            cv.imshow(file, img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        