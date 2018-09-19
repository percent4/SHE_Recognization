import os
import cv2

# 统一将人脸调整到80*80的灰度图像
def resize(directory, path):
    ImagePath = directory+'/'+path
    image = cv2.imread(ImagePath)
    # 把图片转换为灰度模式
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (80,80), interpolation=cv2.INTER_AREA)
    cv2.imwrite(ImagePath, img)

def main():
    directory = ['E://face_data/Ella', 'E://face_data/Selina', 'E://face_data/Hebe']

    for dir in directory:

        images = os.listdir(dir)

        for image in images:
            resize(dir, image)

main()