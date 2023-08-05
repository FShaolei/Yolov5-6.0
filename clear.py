# -*- coding: utf-8 -*-
# @Time : 2023/7/28 11:17
# @Author : Leo_F
# @Email : Fshaolei.F@gmail.com
# @File : clear.py
# @Software ：Pycharm

import xml.dom.minidom
from icecream import ic
import os



# 可不删图片
if __name__ == '__main__':
    xml_dirs = 'data_set/A/'
    test_image = 'data_set/I/'
    # print(data[0].split(';', 6))
    for image in os.listdir(test_image):
        # print(image)
        filename = os.path.splitext(os.path.split(image)[1])[0]
        if not os.path.exists(xml_dirs + filename + '.xml'):
            ic('delete:',image)
            os.remove(test_image + image)
    print('完成删除图片')

if __name__ == '__main__':
    xml_dirs = 'data_set/A/'
    test_image = 'data_set/I/'
    # print(data[0].split(';', 6))
    for xml_ in os.listdir(xml_dirs):
        # ic(xml_)
        filename = os.path.splitext(os.path.split(xml_)[1])[0]
        if not os.path.exists(test_image + filename + '.jpg'):
            ic('delete:',xml_)
            os.remove(xml_dirs + xml_)
    print('完成删除xml')




path = "data_set/A/"
files = os.listdir(path)  # 得到文件夹下所有文件名称
s = []
s_ = []
remove_path = []
for xmlFile in files:  # 遍历文件夹
    if  not os.path.isdir(os.path.join(path, xmlFile)) :  # 判断是否是文件夹,不是文件夹才打开
        # ic(os.path.join(path, xmlFile))
        try:
            dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  ###最核心的部分os.path.join(path,xmlFile),路径拼接,输入的是具体路径
        except Exception as e:                                                      #
            os.remove(os.path.join(path, xmlFile))
            continue

        root = dom.documentElement
        # ic(dom)  #dom: <xml.dom.minidom.Document object at 0x000001CFD797B228>
        # ic(root)  #root: <DOM Element: annotation at 0x1cfd7903638>

        dict_ = {}
        #获取图片宽、高
        size = root.getElementsByTagName('size')


        #1、第一种获取size的方法
        height = size[0].getElementsByTagName('height')
        height_data = height[0].firstChild.data
        # ic(height, height_data, type(height_data))  # height_data: '576' type(height_data): <class 'str'>
        dict_['height'] = str(height_data)

        width = size[0].getElementsByTagName('width')
        width_data = width[0].firstChild.data
        # ic(width, width_data)  # width_data: '704'
        dict_['width'] = str(width_data)
        ob = root.getElementsByTagName('object')  #ob: [<DOM Element: object at 0x2d0c53a3cc0>] # 根据标签名获取元素节点，是个列表
        if ob.__len__() == 0:  #如果没有ob节点，则没有标记框
            os.remove(os.path.join(path, xmlFile))
            ic('delete:',xmlFile)
            continue
        for o in ob:
            name = o.getElementsByTagName('name')
            name[0].firstChild.data = 'leaf'  #节点赋值
            bndbox = o.getElementsByTagName('bndbox')  #[<DOM Element: bndbox at 0x25c53c1c048>]
            for i in bndbox:
                xmin = i.getElementsByTagName('xmin')
                xmin_data = xmin[0].firstChild.data
                xmax = i.getElementsByTagName('xmax')
                xmax_data = xmax[0].firstChild.data
                ymin = i.getElementsByTagName('ymin')
                ymin_data = ymin[0].firstChild.data
                ymax = i.getElementsByTagName('ymax')
                ymax_data = ymax[0].firstChild.data
                s.append(
                    {'xmin': str(xmin_data), 'ymin': str(ymin_data), 'xmax': str(xmax_data), 'ymax': str(ymax_data)})

                # 判断坐标要求,不符合要求，则删除xml标记文件
                if int(xmin_data) < 0 or int(ymin_data) < 0 \
                        or int(xmin_data) > int(width_data) or int(ymin_data) > int(height_data) \
                        or int(xmax_data) > int(width_data) or int(ymax_data) > int(height_data) \
                        or int(xmax_data) < 0 or int(ymax_data) < 0:
                    remove_path.append(os.path.join(path, xmlFile))                     # 保存删除的xml文件路径

                    ic(xmlFile)
                    # ic(os.path.join(path, xmlFile))

                    try:
                        os.remove(os.path.join(path, xmlFile))
                    except Exception as e:
                        break
                    # ic(xmlFile)
                else:
                    s_.append({'xmin': str(xmin_data), 'ymin': str(ymin_data), 'xmax': str(xmax_data),
                               'ymax': str(ymax_data)})


ic(s)
ic(s_)
ic(remove_path)
print('写入name/pose OK!')



