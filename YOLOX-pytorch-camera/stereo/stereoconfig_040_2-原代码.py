import numpy as np


####################仅仅是一个示例###################################


# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([   [480.1827401743536,  -3.996193819153865, 313.2484204798977],
                                            [0,                 477.3816416456625,   240.3265232955334],
                                            [0,                       0,             1]])
        # 右相机内参
        self.cam_matrix_right = np.array([ [487.3771447246162,  -5.130691903427561,  274.7902060436339],
                                           [ 0,                 485.3107014237426,   252.7409169924596],
                                           [  0,                  0,                  1]])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.0891,-0.4382,0,0,1.3420]])
        self.distortion_r = np.array([[0.0461050900425548, -0.0472360135958562, 0,0,0.0220064865167176]])

        # 旋转矩阵
        self.R = np.array([ [0.996924143460104, -0.001204726123944, -0.078363261936352],
                            [0.002139716549006, 0.999927513315018, 0.011848617165509],
                            [0.078343307304638, -0.011979847687312, 0.996854457506190]])
        # 平移矩阵
        self.T = np.array([[-44.046409551751990], [0.058862432865362], [6.362708016358077]])

        # 焦距
        self.focal_length = 478.78219 # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

        # 基线距离
        self.baseline = 44.046409551751990 # 单位：mm， 为平移向量的第一个参数（取绝对值）

        


