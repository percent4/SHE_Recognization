# -*- coding: utf-8 -*-
import cv2
import os

i = 0

def face_extraction(directory, ImagePath):
    global i

    try:
        # 读取图片
        image = cv2.imread(directory+'/'+ImagePath)
        # 把图片转换为灰度模式
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 探测图片中的人脸
        # 获取训练好的人脸的参数数据,进行人脸检测
        face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(3, 3))

        if len(faces) == 1:
            i += 1
            x,y,w,h = faces[0]
            # 保存图片
            save_directory = directory.replace('_raw_image', '')
            cv2.imwrite('%s/%d.jpg'%(save_directory,i), image[y:y+h, x:x+w])
            print('获取第%d个人脸图片'%i)

    except Exception as err:
        print(err)

def main():
    directories = ['E://face_data/Ella_raw_image',
                   'E://face_data/Hebe_raw_image',
                   'E://face_data/Selina_raw_image'
                  ]
    for directory in directories:
        images = os.listdir(directory)
        for image in images:
            face_extraction(directory, image)

main()