'''时间：2020/5/22   0：49—— 2020/5/31

   conding:utf-8
   基于opencv match(模板匹配) 识别银行卡数字(4 * 4 )'''

import cv2
import numpy as np
import os
def show(img):
    cv2.imshow('image',img)
    cv2.waitKeyEx()
    cv2.destroyAllWindows()
def sorted_contours(contours):
    '''对矩阵从左到右排序'''
    cont_dit = {}
    cont_list = []
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        cont_dit[x] = cnt
    sorted_list = sorted(cont_dit.items(),key=lambda x:x[0])
    for i in sorted_list:
        cont_list.append(i[1])
    return cont_list


#1.图片预处理，保存模板
#读取模板template
template_img = cv2.imread("template.png")
#show(template_img)

#灰度图
template_gray = cv2.cvtColor(template_img,cv2.COLOR_BGR2GRAY)
#show(template_gray)

#二值图
ret, tem_thersh = cv2.threshold(template_gray,127,255,cv2.THRESH_BINARY_INV)
h,w = tem_thersh.shape[:2]
# resize_thresh = tem_thersh[3:h-3,3:w-3]
#show(tem_thersh)

# #轮廓检测设立标签,外轮廓，点集
contours, hierachy, = cv2.findContours(tem_thersh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# dra = cv2.drawContours(template_img.copy(), contours,-1,(0,0,255))
# show(dra)
# print(len(contours))

# #外接矩形，坐标点，排序
cont_list = sorted_contours(contours)
template = {}#模板
# dra = cv2.drawContours(template_img.copy(), cont_list,-1,(0,0,255),3)
#show(dra)
#写入文件
for i in range(len(cont_list)):
    (x1,y1,w1,h1) =cv2.boundingRect(cont_list[i])
    # cv2.rectangle(template_img,(x1,y1),(x1+w1,y1+h1),255,3)
    # show(template_img)
    tem = tem_thersh[y1-1:y1+h1+1,x1-1:x1+w1+1]
    #重设标准大小,保存模板
    tem = cv2.resize(tem,(100,160))
    template[i] = tem
    if  not os.path.exists('tempelate/tempelate '+ str(i)+'.jpg'):
        show(tem)
        cv2.imwrite('tempelate/tempelate '+ str(i)+'.jpg',tem)
    else:
        pass



# #2.银行卡预处理
#     #读取
card_img = cv2.imread('card.png')  #现在只能识别这一张，嘻嘻
card_img =cv2.resize(card_img,(232,140))

#     #灰度图
card_gray = cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)
show(card_gray)

#     #初始化卷积核
rettkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))

#     #形态学操作->突出字体
card_mor = cv2.morphologyEx(card_gray,cv2.MORPH_TOPHAT,rettkernel)
# show(card_mor)

#     #梯度运算,为了将数字形成特征块
gradx = cv2.Sobel(card_mor, ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)#-1相当于3*3
card_gradx = cv2.convertScaleAbs(gradx)
# show(card_gradx)
card_close = cv2.morphologyEx(card_gradx,cv2.MORPH_CLOSE,rettkernel)
# show(card_close)

#     #二值化，cv2.THRESH_OTSU
ret , card_thresh = cv2.threshold(card_close, 0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
card_balck = cv2.morphologyEx(card_thresh,cv2.MORPH_OPEN,rettkernel)

show(card_balck)

#     ##长宽比例去除误差
black_contours, black_hierachy = cv2.findContours(card_balck.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(card_img,number_contours,-1,(0,0,255),3)
# show(card_img)

#     #遍历取出数字块
black_loc = []
for (i,c) in enumerate(black_contours):
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    if ar > 1.5 and ar < 3.5:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            cv2.rectangle(card_img,(x,y),(x+w,y+h),(0,0,255),3)
            black_loc.append([x,y,w,h])#加入目标
show(card_img)
black_loc = sorted(black_loc,key=lambda x:x[0])#进行一个排序

number = []#数字总集

#对每个数字块处理，提取出每个数字进行匹配
for (x,y,w,h) in black_loc:
    #每个块中数字的位置
    roi = card_gray[y-1:y + 1+ h, x-1:x+w+1]
    #突出字体
    # show(roi)

    #二值化
    ret, number_img = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    # #轮廓检测设立标签,外轮廓，点集
    number_contours, hierachy, = cv2.findContours(number_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # dra = cv2.drawContours(roi.copy(), number_contours,-1,(0,0,255),1)
    # show(dra)

    #获取每个数字的轮廓点
    number_loc = []
    for (i, c) in enumerate(number_contours):
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),3)
        number_loc.append([x, y, w, h])  # 加入目标
    show(roi)
    number_loc = sorted(number_loc, key=lambda x: x[0])  # 进行一个排序

    #对每一个数字进行匹配
    for (x,y,w,h) in number_loc:
        one_number = number_img[y-1:y + 1+ h, x-1: x+w+1]#从二值化过的数字块扣下来

        one_number = cv2.resize(one_number,(100,160))
        show(one_number)

        scores = []
        for i in range(len(template)):
            res = cv2.matchTemplate(one_number, template[i],cv2.TM_CCOEFF)#???
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            scores.append(max_val)

        number.append(np.argmax(scores))#返回最大值的索引值

print(number)
