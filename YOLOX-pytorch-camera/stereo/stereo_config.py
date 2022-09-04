import numpy as np
import cv2

# 双目相机参数
cam_matrix_left = np.array([ [824.2550965681365, -6.828558820006515,  310.1707580237725],
                                  [0,              815.4139580995355,  249.1939684854760],
                                  [0,   0,   1]])
# 右相机内参
cam_matrix_right = np.array([ [822.9498573074617,  -6.900112764184275,297.7290467865062],
                                   [0,                  818.8382890544332,222.9237421035944],
                                   [0,      0  ,1]])

# 左右相机畸变系数:[k1, k2, p1, p2, k3]
distortion_l = np.array([-0.246284288363976,-0.361503832978841, -0.010028059937390,-0.015986382957965, 74.621557080121280])
distortion_r = np.array([-0.246284288363976,-0.361503832978841,-0.010618446977411,-0.026827033332358,73.641634617150690])
#-0.034424624717455,-5.965881599855729
# 旋转矩阵
R = np.array([[0.999989229476667,-0.001079697704587,-0.004513887850642],
                   [9.617580928426970e-04,0.999660199821240,-0.026049182611556],
                   [0.004540479273416,0.026044560780057,0.999650471365738]])
# 平移矩阵
T = np.array([[-79.431038317632310],[0.900649741211188],[1.327465851666942]])

# 焦距
focal_length = 819.83445 # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

# 基线距离
baseline = 79.431038317632310 # 单位：mm， 为平移向量的第一个参数（取绝对值）
width = 640
heigh = 480
size = (640,480)
# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cam_matrix_left,distortion_l,
                                                                  cam_matrix_right,distortion_r, size,
                                                                  R,T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(cam_matrix_left,distortion_l, R1, P1, size,
                                                   cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(cam_matrix_right,distortion_r, R2, P2, size,
                                                     cv2.CV_16SC2)

