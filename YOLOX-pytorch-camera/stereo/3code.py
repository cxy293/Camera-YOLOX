import cv2
import numpy as np
import time
import numpy as np
from PIL import Image

def cat2images(limg, rimg):
    HEIGHT = limg.shape[0]
    WIDTH = limg.shape[1]
    imgcat = np.zeros((HEIGHT, WIDTH * 2 + 20, 3))
    imgcat[:, :WIDTH, :] = limg
    imgcat[:, -WIDTH:, :] = rimg
    for i in range(int(HEIGHT / 32)):
        imgcat[i * 32, :, :] = 255
    return imgcat

capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
capture1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
ret,frame = capture.read()
ref1, frame1 = capture1.read()

frame_left = frame
frame_right = frame1



left_image = cv2.imread("./yolo/left/left_7.bmp")
right_image = cv2.imread("./yolo/right/right_7.bmp")
imgcat_source = cat2images(left_image, right_image)

#imgcat_source = cat2images(frame_left, frame_right)

HEIGHT = frame_left.shape[0]
WIDTH = frame_left.shape[1]
cv2.imwrite('./yolo/xiaozheng_qian1.jpg', imgcat_source)

camera_matrix0 = np.array([[757.894, -2.54024,  281.73356],
                               [0,     748.34388,  230.360],
                               [0,   0,   1]]).reshape((3, 3))  # 即上文标定得到的 cameraMatrix1

distortion0 = np.array(
    [-0.219606598267299,1.538265449459559,-0.008759917468026,-0.012829271853108,-15.006098042168361])  # 即上文标定得到的 distCoeffs1

camera_matrix1 = np.array([[756.2052,  -3.677445,282.60554],
                                [0,  751.20658,213.2578202],
                                [0,      0  ,1]]).reshape((3, 3))  # 即上文标定得到的 cameraMatrix2
distortion1 = np.array(
    [-0.045914586942381,-6.258346101399701,-0.008021351260658,-0.022897393264614,72.703142703152610])  # 即上文标定得到的 distCoeffs2

R = np.array([[0.999989229476667,-0.001079697704587,-0.004513887850642],
                           [9.617580928426970e-04,0.999660199821240,-0.026049182611556],
                           [0.004540479273416,0.026044560780057,0.999650471365738]]
             )  # 即上文标定得到的 R
T = np.array([-79.363393754544390,0.695577933585846,1.098609901964254])  # 即上文标定得到的T

(R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2) = \
    cv2.stereoRectify(camera_matrix0, distortion0, camera_matrix1, distortion1, np.array([WIDTH, HEIGHT]), R,
                      T)  # 计算旋转矩阵和投影矩阵

(map1, map2) = \
    cv2.initUndistortRectifyMap(camera_matrix0, distortion0, R_l, P_l, np.array([WIDTH, HEIGHT]),
                                cv2.CV_32FC1)  # 计算校正查找映射表

rect_left_image = cv2.remap(left_image, map1, map2, cv2.INTER_CUBIC)  # 重映射

# 左右图需要分别计算校正查找映射表以及重映射
(map1, map2) = \
    cv2.initUndistortRectifyMap(camera_matrix1, distortion1, R_r, P_r, np.array([WIDTH, HEIGHT]), cv2.CV_32FC1)

rect_right_image = cv2.remap(right_image, map1, map2, cv2.INTER_CUBIC)

imgcat_out = cat2images(rect_left_image, rect_right_image)
cv2.imwrite('./yolo/xiaozheng_hou1.jpg', imgcat_out)
