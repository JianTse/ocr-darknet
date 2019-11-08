#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config
@author: chineseocr
"""
rootDir = 'E:/work/Item/OCR/darknet-ocr-master/darknet-ocr-master/models/'
#ocrPath  = rootDir + 'ocr-lstm.pth'
ocrPath  = rootDir + 'ocr-dense-keras.h5'
#ocrPath  = 'models/ocr.weights'
textPath = rootDir + 'text.weights'
darkRoot ='darknet/libdarknet.so' ##darknet 
TEXT_LINE_SCORE=0.85##text line prob
scale = 608
maxScale = 1800
GPU=False ## gpu for darknet cpu for opencv.dnn 
anchors = '16,11, 16,16, 16,23, 16,33, 16,48, 16,68, 16,97, 16,139, 16,198, 16,283'
