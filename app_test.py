# -*- coding: utf-8 -*-
"""
@author: chineseocr
"""

import json
import os
import cv2
import numpy as np
from config import scale,maxScale,TEXT_LINE_SCORE
from dnn.image import rotate_cut_img,sort_box
from PIL import Image
from dnn.network_torch import CRNN as CRNN_TORCH
from dnn.network_keras import CRNN as CRNN_KERAS
from dnn.text import detect_lines
from dnn.keys import alphabetChinese
from config import ocrPath,GPU

nclass = len(alphabetChinese)+1
LSTMFLAG = True
alphabet = alphabetChinese

#crnn_torch = CRNN_TORCH( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=False,alphabet=alphabet)
#crnn_torch.load_weights(ocrPath)

crnn_keras = CRNN_KERAS( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=False,GPU=False,alphabet=alphabet)
crnn_keras.load_weights(ocrPath)

def drawDetectBox(img, resJson):
    for idx in range(len(resJson['data'])):
        box = resJson['data'][idx]['box']
        [x1,y1,x2,y2,x3,y3,x4,y4] = box
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        p3 = (int(x3), int(y3))
        p4 = (int(x4), int(y4))
        cv2.line(img, p1, p2, (0, 255, 0))
        cv2.line(img, p2, p3, (0, 255, 0))
        cv2.line(img, p3, p4, (0, 255, 0))
        cv2.line(img, p4, p1, (0, 255, 0))
        #cv2.putText(img, str(text_tags[idx]), (int(p1[0]), int(p1[1])), 1, 1, (0, 0, 255))
    cv2.imshow('detect', img)
    cv2.waitKey(0)

def sort_box(box):
    """
    对box进行排序
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box

def ocr_batch(img, boxes, leftAdjustAlph=0.01, rightAdjustAlph=0.01):
    """
    batch for ocr
    """
    im = Image.fromarray(img)
    newBoxes = []
    for index, box in enumerate(boxes):
        partImg, box = rotate_cut_img(im, box, leftAdjustAlph, rightAdjustAlph)
        box['img'] = partImg.convert('L')
        newBoxes.append(box)

        raw = crnn_keras.predict(box['img'])
        print(raw)
        #cvPartImg = np.array(partImg)
        #cvImg = cv2.cvtColor(cvPartImg, cv2.COLOR_RGB2BGR)
        #cv2.imshow('part', cvImg)
        #cv2.waitKey(0)

def detectText(img):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    boxes, scores = detect_lines(image, scale=scale, maxScale=maxScale)
    data = []
    n = len(boxes)
    for i in range(n):
        box = boxes[i]
        box = [int(x) for x in box]
        if scores[i] > TEXT_LINE_SCORE:
            data.append({'box': box, 'prob': round(float(scores[i]), 2), 'text': None})
    res = {'data': data, 'errCode': 0}
    return res


imgDir = 'C:/Users/Administrator/Desktop/chinese_ocr-master/test_images/'
img = cv2.imread(imgDir + 'demo3.jpg')
res = detectText(img)
print(res)

boxes = []
for idx in range(len(res['data'])):
    box = res['data'][idx]['box']
    boxes.append(box)

text_recs = sort_box(boxes)
ocr_batch(img, text_recs)

