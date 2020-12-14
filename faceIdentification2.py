import face_recognition
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 路径参数配置
basefacefilespath = "face_identification\gallery"  # faces文件夹中放待识别任务正面图,文件名为人名,将显示于结果中
destfacefilepath = "face_identification\probe_test"  # 用于识别的目标图片目录


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


# 写入中文字符支持
def paint_chinese_opencv(im, chinese, pos, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('simsun.ttc', 14)
    fillColor = color  # (255,0,0)
    position = pos  # (100,100)
    # chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


# 余弦相似度识别阈值
tolerance = 0.96

# 加载待识别人脸图像并识别。
baseface_titles = []  # 图片名字列表
baseface_face_encodings = []  # 识别所需人脸编码结构集
# 读取人脸资源
for fn in os.listdir(basefacefilespath):  # fn 人脸文件名
    if (len(face_recognition.face_encodings(face_recognition.load_image_file(basefacefilespath + "/" + fn))) > 0):
        baseface_face_encodings.append(
            face_recognition.face_encodings(face_recognition.load_image_file(basefacefilespath + "/" + fn))[0])
        # fn = fn.split("_")[1]
        fn = fn[:(len(fn) - 4)]
        baseface_titles.append(fn)
        # print(fn)
print("================begin================")

res = open("pred_test.txt", "w")

# 从识别库中读取一张图片并识别
for fd in os.listdir(destfacefilepath):
    # 获取一张图片
    faceData = face_recognition.load_image_file(destfacefilepath + "/" + fd)
    print(fd)

    # 人脸检测,并获取帧中所有人脸编码
    # 参数说明，number_of_times_to_upsample可以理解为迭代次数，次数越大，人脸检测越准确，但是运行耗时越长。
    # model默认为hog，运行速度快，但是识别率不高。后面第2张图分辨率低，所以必须用cnn并且迭代次数为2
    face_locations = face_recognition.face_locations(faceData, number_of_times_to_upsample=2, model="hog")
    face_encodings = face_recognition.face_encodings(faceData, face_locations)
    # 遍历图片中所有人脸编码
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 与baseface_face_encodings匹配否?
        name = "?"
        for i, v in enumerate(baseface_face_encodings):
            # match = face_recognition.compare_faces([v], face_encoding,tolerance=0.5)
            match = cos_sim([v], face_encoding)
            name = "?"
            if match >= tolerance:
                name = baseface_titles[i]
                print("识别出：" + name)
                res.writelines(fd + " " + name + "\n")
                break

        # 围绕脸的框
        # cv2.rectangle(faceData, (left, top), (right, bottom), (0, 0, 255), 2)

        # 如果遇到没有识别出的人脸，则跳过
        if name == "?":
            continue

        # 框下的名字(即,匹配的图片文件名)
        # cv2.rectangle(faceData, (left, bottom), (right, bottom + 25), (0, 0, 255), cv2.FILLED)
        # faceData = cv2.putText(faceData, name,(left + 2, bottom + 12), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255),1)
        # faceData = paint_chinese_opencv(faceData, name, (left + 2, bottom + 4), (255, 255, 255))
    # frame = ft.draw_text(frame, (left + 2, bottom + 12), name, 16,  (255, 255, 255))

    # show结果图像
    # cv2.imshow(fd, cv2.cvtColor(faceData, cv2.COLOR_BGR2RGB))
res.close()
# cv2.waitKey()
# cv2.destroyAllWindows()
