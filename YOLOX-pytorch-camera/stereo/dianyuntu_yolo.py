 # -*- coding: utf-8 -*-
import cv2
import numpy as np

from stereo.stereoconfig_040_2 import stereoCamera
#from stereoconfig_040_2 import stereoCamera
#import stereoconfig_040_2
#import stereoconfig_040_2   #导入相机标定的参数
import pcl
import pcl.pcl_visualization
#config = stereoCamera()
# 预处理
#config = stereoconfig_040_2.stereoCamera()

def preprocess(img1, img2):
    # 彩色图->灰度图
    if(img1.ndim == 3):#判断为三维数组
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if(img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    #print("R{}".format(config.R))
    T = config.T
    height = int(height)
    width = int(width)
    #R1 = stereoCamera.R1
    # 计算校正变换
    # stereoRectify() 的作用是为每个摄像头计算立体校正的映射矩阵。
    # 所以其运行结果并不是直接将图片进行立体矫正，而是得出进行立体矫正所需要的映射矩阵。
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                    (width, height), R, T, flags=1, alpha=-1)
    #这个函数用于计算无畸变和修成转换关系
    #输出左图和右图的X和Y坐标的重映射参数
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)#cv2.CV_16SC2
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    #一幅图像中某位置的像素放置到另一个图片指定位置  #cv2.INTER_LINEAR
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)#cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    #-----------------------
    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    print("立体校正完成————————————")
    return output


# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):

#-----------------设置参数可调窗口显示-----------------
    num = cv2.getTrackbarPos("num", "set")
    blockSize = cv2.getTrackbarPos("blockSize", "set")
    if blockSize % 1 == 0:
        blockSize += 1
    if blockSize < 1:
        blockSize = 1
    if num < 2:
        num = 2
#---------------------------END-------------------------
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    # num = 8
    # blockSize = 3    #  3-->5--->
    paraml = {'minDisparity': 0,
             'numDisparities': 16 * num,
             #'numDisparities': 128,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 100,
             'speckleRange': 2,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)

    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right

#-----------------------New-SGBM算法--------------
def new_SGBM(left_image, right_image, down_scale = False):
    # -----------------设置参数可调窗口显示-----------------
    num = cv2.getTrackbarPos("num", "set")
    blockSize = cv2.getTrackbarPos("blockSize", "set")
    if blockSize % 1 == 0:
        blockSize += 1
    if blockSize < 1:
        blockSize = 1
    if num < 2:
        num = 2
    # ---------------------------END-------------------------
    # SGBM匹配参数设置
    if left_image.ndim == 2:  # python-opencv读取的灰度图像是二维列表（数组）,彩色图像是三位列表（数组），.ndim返回的是数组的维度
        img_channels = 1
    else:
        img_channels = 3
    # ------------------------------
    # blockSize = 3
    # ---------------end-------------
    paraml = {'minDisparity': 0,
              'numDisparities': 16 * num,  # 64
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 5,  # 63
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # 构建SGBM对象
    stereo = cv2.StereoSGBM_create(**paraml)
    disp = stereo.compute(left_image, right_image)

    # 转换为单通道图片
    #disp = cv2.normalize(disp, disp, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return disp




#-----------------------BM算法-----------------------------------
def stereoMatchBM(left_image, right_image):
    # BM算法用到的
    #cv2.namedWindow("left")
    #cv2.namedWindow("right")
    cv2.namedWindow("depth")
    #cv2.moveWindow("left", 0, 0)
    #cv2.moveWindow("right", 600, 0)
    cv2.createTrackbar("num", "depth", 0, 20, lambda x: None)
    cv2.createTrackbar("blockSize", "depth", 1, 25, lambda x: None)
    # 两个trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "depth")
    blockSize = cv2.getTrackbarPos("blockSize", "depth")
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5
    # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
    stereo = cv2.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)
    # stereo = cv2.StereoSGBM_create(numDisparities=16 * num, blockSize=blockSize)
    disparity = stereo.compute(left_image, right_image)
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disp
#-----------------------------------------------------------

# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_


# 深度、颜色转换为点云
def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols

    points_ = hw3ToN3(points_3d)
    colors_ = hw3ToN3(colors).astype(np.int64)

    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)

    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

    # 将坐标+颜色叠加为点云数组
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)

    # 删掉一些不合适的点
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]

    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 10000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack((remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)

    return pointcloud_1


# 点云显示
def view_cloud(pointcloud):
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_array(pointcloud)

    try:
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorACloud(cloud)
        v = True
        while v:
            v = not (visual.WasStopped())
    except:
        pass


if __name__ == '__main__':

    #i = 1
    #string = ''
    # 读取数据集的图片
    iml = cv2.imread('./yolo/left/left_37.bmp' )  # 左图
    imr = cv2.imread('./yolo/right/right_37.bmp' ) # 右图
    height, width = iml.shape[0:2]
    #*****************************************************************
    print("左图的尺寸{}：".format(iml.shape))
    print("右图的尺寸{}：".format(imr.shape))
    #*****************************************************************
    print("width =  %d \n"  % width)
    print("height = %d \n"  % height)
    

    # 读取相机内参和外参
    config = stereoCamera()

    # 立体校正
    # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)

    print("Print Q!")
    print(Q[2,3])

    # 绘制等间距平行线，检查立体校正的效果
    line = draw_line(iml_rectified, imr_rectified)
    cv2.imwrite('./yolo/1_j.png', line)

    # 消除畸变
    iml = undistortion(iml, config.cam_matrix_left, config.distortion_l)
    imr = undistortion(imr, config.cam_matrix_right, config.distortion_r)

    # 立体匹配
    iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以

    iml_rectified_l, imr_rectified_r = rectifyImage(iml_, imr_, map1x, map1y, map2x, map2y)

    disp, _ = stereoMatchSGBM(iml_rectified_l, imr_rectified_r, True) 
    cv2.imwrite('./yolo/1.png', disp)

    

    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)  # 可以使用上文的stereo_config.py给出的参数

    #points_3d = points_3d

        # 鼠标点击事件
    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('点 (%d, %d) 的三维坐标 (x:%.3fm, y:%.3fm, z:%.3fm)' % (x, y, points_3d[y, x, 0]/1000, points_3d[y, x, 1]/1000, points_3d[y, x, 2]/1000))
            dis = ( (points_3d[y, x, 0] ** 2 + points_3d[y, x, 1] ** 2 + points_3d[y, x, 2] **2) ** 0.5) / 1000
            print('点 (%d, %d) 距离左摄像头的相对距离为 %0.3f m' %(x, y, dis) )

        # 显示图片
    cv2.namedWindow("disparity",0)
    cv2.imshow("disparity", disp)
    cv2.setMouseCallback("disparity", onMouse, 0)

    

    # 构建点云--Point_XYZRGBA格式
    pointcloud = DepthColor2Cloud(points_3d, iml)

    # 显示点云
    view_cloud(pointcloud)

    cv2.waitKey(0)
    cv2.destroyAllWindows()