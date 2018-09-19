import os
import cv2
import pandas as pd

# 获取图片的特征和标签

table = []

def sample(directory, path):
    ImagePath = directory+'/'+path
    image = cv2.imread(ImagePath, 0)
    label = directory.split('/')[-1]
    global table
    table.append(list(image.ravel())+[label])

def main():

    directory = ['E://face_data/Ella', 'E://face_data/Selina', 'E://face_data/Hebe']

    for dir in directory:

        images = os.listdir(dir)

        for image in images:
            sample(dir, image)
    columns = ['v' + str(i) for i in range(1, 80 * 80 + 1)] + ['label']
    df = pd.DataFrame(table, columns=columns)
    # print(df.head(n=10))
    df.to_csv('E://face_data/sample_test.csv', index=False)

main()

