"""
挑选出dectet.py识别出来的图像与txt文件，选择置信度在一定范围的图片
"""

import os
import cv2
import shutil

# jpg图片和对应的生成结果的txt标注文件，放在一起,为目标detect.py 生成结果
path_txt = ""
path_img = ""

# 保存最后
path_save = ""

img_total = []
txt_total = []

file = os.listdir(path_txt)
for filename in file:
    first, last = os.path.splitext(filename)
    print(first)
    print(last)
    # 保存以jpg结尾的文件名前缀
    if last == ".jpg":  # 图片的后缀名
        img_total.append(first)

        print(img_total)
    # 保存以txt结尾的文件名的前缀
    else:
        txt_total.append(first)

flag = 0

save_list = []
# 保存识别出来的目标检测有bbox图片的到指定文件夹
for j, i in enumerate(txt_total):
    result2_list = []
    filename_img = i + ".jpg"
    filename_txt = i + ".txt"

    path_filename_img = os.path.join(path_img, filename_img)  # 图片的路径
    path_filename_txt = os.path.join(path_txt, filename_txt)  # txt文件的路径

    try:
        with open(path_filename_txt, "r") as f:
            data = f.readline()
            if data.split(" ")[0] == "0" and float(data.split(" ")[5]) >= 0.1:
                print(path_filename_txt + "," + data.split(" ")[5])
                shutil.copy(path_filename_img, path_save + '/' + str(i) + '.jpg')
                print(f'挑选出图片数:{flag}')
                flag += 1
                save_list.append(path_filename_txt + "," + data.split(" ")[5] + "\n")


            else:
                continue
    except:
        pass

fresult = open("path_save/name.txt", 'w')  # w:只写，文件已存在则清空，不存在则创建
for i, file in enumerate(save_list):
    print(f'存入txt文件:{i}')
    fresult.write('path,blur_threshold:' + file)
fresult.close()
