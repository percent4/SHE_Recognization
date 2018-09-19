import cv2
import numpy as np
from TF_CNN import CNN

# 模型保存地址
MODEL_SAVE_PATH = 'E://logs/cnn_she_model.ckpt'
# CNN初始化
cnn = CNN(100, 0.0001, MODEL_SAVE_PATH)

# 利用CNN模型进行人脸识别
def reg(image_path):

    # 读取图片
    image = cv2.imread(image_path)
    # 把图片转换为灰度模式
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 探测图片中的人脸
    # 获取训练好的人脸的参数数据,进行人脸检测
    face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(3, 3))

    if len(faces) >= 1:
        results = []
        for face in faces:
            x,y,w,h = face
            vec = gray[y:y+h, x:x+w]

            # 显示图片
            # cv2.imshow('find faces!', cv2.resize(vec, (80, 80), interpolation=cv2.INTER_AREA))
            # cv2.waitKey(0)

            img = cv2.resize(vec, (80, 80), interpolation=cv2.INTER_AREA).ravel().reshape(1, 6400)
            # 标准化
            mean,std = np.mean(img), np.std(img)
            img = (img-mean)/std
            # print(img)
            pred = cnn.predict(img)
            reg_result = list(pred[0]).index(max(pred[0]))+1
            SHE_dict = {'1': 'Ella', '2': 'Selina', '3': 'Hebe'}
            # print('This is %s.'%SHE_dict[str(reg_result)])
            results.append(SHE_dict[str(reg_result)])

        print(results)
        # 如果存在重复识别
        if len(set(results)) < len(results):
            print('识别失败！')
        else:
            i = 0
            for face in faces:
                x, y, w, h = face
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, results[i], (x, y-7), 1, 1.5, (0, 255, 0), 2, 5)
                i += 1

            # 显示图片
            cv2.imshow('find faces!', image)
            cv2.waitKey(0)

    else:
        print('There is no human face.')

def main():

    image_path = "E://she3.jpg"
    reg(image_path)

main()