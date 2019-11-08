#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image
@author: chineseocr
"""
import cv2
import numpy as np
from config import textPath,anchors
from helper.image import resize_img,get_origin_box,soft_max,reshape
from helper.detectors import TextDetector
from config import scale,maxScale,TEXT_LINE_SCORE
from dnn.image import rotate_cut_img,sort_box
from PIL import Image

modelDir = 'E:/work/Item/OCR/darknet-ocr-master/darknet-ocr-master/models/'
#textNet   =  cv2.dnn.readNetFromDarknet(textPath.replace('weights','cfg'),textPath)
textNet   =  cv2.dnn.readNetFromDarknet(modelDir+'text.cfg',modelDir+'text.weights')

def detect_box(image,scale=600,maxScale=900):
        H,W = image.shape[:2]
        image,rate = resize_img(image,scale,maxScale=maxScale)
        h,w = image.shape[:2]
        inputBlob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(w,h),swapRB=False ,crop=False);
        outputName = textNet.getUnconnectedOutLayersNames()
        textNet.setInput(inputBlob)
        out  = textNet.forward(outputName)[0]
        clsOut  = reshape(out[:,:20,...])
        boxOut  = reshape(out[:,20:,...])
        boxes  = get_origin_box((w,h),anchors,boxOut[0])        
        scores = soft_max(clsOut[0])
        boxes[:, 0:4][boxes[:, 0:4]<0] = 0
        boxes[:, 0][boxes[:, 0]>=w] = w-1
        boxes[:, 1][boxes[:, 1]>=h] = h-1
        boxes[:, 2][boxes[:, 2]>=w] = w-1
        boxes[:, 3][boxes[:, 3]>=h] = h-1
        
        return scores,boxes,rate,w,h
    
    
def detect_lines(image,scale=600,
                 maxScale=900,
                 MAX_HORIZONTAL_GAP=30,
                 MIN_V_OVERLAPS=0.6,
                 MIN_SIZE_SIM=0.6,
                 TEXT_PROPOSALS_MIN_SCORE=0.7,
                 TEXT_PROPOSALS_NMS_THRESH=0.3,
                 TEXT_LINE_NMS_THRESH = 0.9,
                 TEXT_LINE_SCORE=0.9
                ):
    MAX_HORIZONTAL_GAP = max(16,MAX_HORIZONTAL_GAP)
    detectors = TextDetector(MAX_HORIZONTAL_GAP,MIN_V_OVERLAPS,MIN_SIZE_SIM)
    scores,boxes,rate,w,h = detect_box(image,scale,maxScale)
    size = (h,w)
    text_lines, scores =detectors.detect( boxes,scores,size,\
           TEXT_PROPOSALS_MIN_SCORE,TEXT_PROPOSALS_NMS_THRESH,TEXT_LINE_NMS_THRESH,TEXT_LINE_SCORE)
    if len(text_lines)>0:
        text_lines = text_lines/rate
    return text_lines, scores

'''
def detect(img):
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

        cvPartImg = np.array(partImg)
        cvImg = cv2.cvtColor(cvPartImg, cv2.COLOR_RGB2BGR)
        cv2.imshow('part', cvImg)
        cv2.waitKey(0)
    #return res

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
    
imgDir = 'C:/Users/Administrator/Desktop/chinese_ocr-master/test_images/'
img = cv2.imread(imgDir + 'demo3.jpg')
res = detect(img)
print(res)

boxes = []
for idx in range(len(res['data'])):
    box = res['data'][idx]['box']
    boxes.append(box)
ocr_batch(img, boxes)

drawDetectBox(img, res)
'''

