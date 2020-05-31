"""
car3b.jpg 处理不出来好看的数字
"""

import cv2

def show(img):
    cv2.imshow('image',img)
    cv2.waitKeyEx()
    cv2.destroyAllWindows()

img = cv2.imread('test.png')
card_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
show(card_gray)
ret, thresh = cv2.threshold(card_gray, 160, 255, cv2.THRESH_BINARY)
close = cv2.dilate(thresh,(5,5))
for i in range(1):
    close = cv2.morphologyEx(close,cv2.MORPH_TOPHAT,(5,5))
contours, hi = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,255,3)
print(len(contours))
show(close)