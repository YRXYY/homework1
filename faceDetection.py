import os

import cv2

classifier = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

for i in range(1, 11):
    os.mkdir("test\\pred\\" + str(i))
    if i < 10:
        gtpath = "test\gt\FDDB-fold-0" + str(i) + ".txt"
    else:
        gtpath = "test\gt\FDDB-fold-10.txt"
    pics = open(gtpath, mode="r").readlines()
    for pic in pics:
        filepath = "test\images\\" + pic[0:5] + ".jpg"
        img = cv2.imread(filepath)  # 读取图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色
        color = (0, 255, 0)
        # 调用识别人脸
        faceRects = classifier.detectMultiScale3(
            gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32), outputRejectLevels=10
        )
        arrays = faceRects[0]  # 人脸坐标数组
        confLevel = faceRects[2]  # 置信度

        # 创建文件
        record = open("test\\pred\\" + str(i) + "\\" + pic[0:5] + ".txt", "w")
        # 写入文件
        record.writelines(pic[0:5] + "\n")
        record.writelines(str(len(arrays)) + "\n")


        for num in range(len(arrays)):
            x, y, w, h = arrays[num]
            record.writelines(
                str(x) + " " + str(y) + " " + str(x + h) + " " + str(y + w) + " " + str(
                    format(confLevel[num][0] / 10, '.3f')) + "\n")


