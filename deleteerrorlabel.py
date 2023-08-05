import os
import xml.etree.ElementTree as ET
from collections import Counter
from icecream import ic

path = r'/root/miniconda3/data/fsl/project/ocr_hat/algorithm/oneStage/yolov5-coco-帽子-V6.0/data_set/A/'
classes = ['have_mask', 'no_mask']
stop = []

xml_list = os.listdir(path)
print(len(xml_list))
label_list = []

for xml in xml_list:
    content = open(path + xml, 'r', encoding='utf-8')
    root = ET.parse(content)
    tree = root.getroot()
    for labels in tree.iter('object'):
        label_list.append(labels.find('name').text)

labels_count = Counter(label_list)
ic(set(label_list))
ic(len(label_list))
ic(labels_count)

for i in range(len(label_list)):
    if label_list[i] not in classes:
        stop.append(label_list[i])

for axml in xml_list:
    path_xml = os.path.join(path, axml)
    tree = ET.parse(path_xml)
    root = tree.getroot()
    print("root",root)
    for child in root.findall('object'):
        name = child.find('name').text
        print("name",name)
        if name in stop:     # 这里可以反向写，不在Class的删掉
            root.remove(child)
            print('delete false label ' + name + ' in ' + path + path_xml)
    # 重写
    tree.write(os.path.join(path, axml))  # 记得新建annotations_new文件夹