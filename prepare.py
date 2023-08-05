# -*- coding: utf-8 -*-
# @Time : 2023/7/28 11:21
# @Author : Leo_F
# @Email : Fshaolei.F@gmail.com
# @File : prepare.py
# @Software ：Pycharm
import random
import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
import cv2
from PIL import Image
import shutil
trainval_percent = 0.1
train_percent = 0.9
path = 'path/data_set/'  # 修改路径
classes = ['class']  # 标签
xmlfilepath = path + 'A'
txtsavepath = path + 'I'
sets = ['train', 'test', 'val']
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open(path + 'image_sets/trainval.txt', 'w')
ftest = open(path + 'image_sets/test.txt', 'w')
ftrain = open(path + 'image_sets/train.txt', 'w')
fval = open(path + 'image_sets/val.txt', 'w')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()


def convert(size, box):
    print(size[0], size)
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open(path + 'A/%s.xml' % (image_id), 'r', encoding='utf-8')       # 打开xml文件
    out_file = open(path + 'labels/%s.txt' % (image_id), 'w')           # 打开txt文件
    tree = ET.parse(in_file)    # 解析xml文件
    root = tree.getroot()    # 获取根节点
    size = root.find('size')    # 找到size节点
    w = int(size.find('width').text)    # 获取宽度
    h = int(size.find('height').text)   # 获取高度
    if w == 0 or h == 0:
        img_path = r'path/data_set/I/%s.jpg' % (image_id)
        try:
            img = cv2.imread(img_path)
            w, h = img.shape[0], img.shape[1]
        except Exception as e:
            img = Image.open(img_path)
            w, h = img.size[1], img.size[0]

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists(path + 'labels/'):        # 判断labels文件夹是否存在，不存在则创建
        os.makedirs(path + 'labels/')               # 创建labels文件夹
    image_ids = open(path + 'image_sets/%s.txt' % (image_set)).read().strip().split()   # 读取train.txt中的图片名
    list_file = open(path + '%s.txt' % (image_set), 'w')    # 创建train.txt文件
    for image_id in image_ids:
        print(image_id)
        list_file.write(path + 'I/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
