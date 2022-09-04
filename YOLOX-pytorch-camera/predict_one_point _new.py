
import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
#**********************双目测距调用的函数*******************************
from stereo import stereoconfig_040_2
from stereo.dianyuntu_yolo import preprocess, undistortion, getRectifyTransform, draw_line, rectifyImage,\
    stereoMatchSGBM, hw3ToN3, DepthColor2Cloud, view_cloud,stereoMatchBM,new_SGBM




#---------------------------------单插头的双目相机---------------------------------------
#-------------------------------------------------------------------------------------
'''
           ref,frame=capture.read()
           # 格式转变，BGRtoRGB
       #----------------------------------------测试代码-------------------------------------------------------
           #1280 480 left[0:480, 0:640]
           #2560 720 left[0:720, 0:1280]
           #1000 320 left[0:320,0:500]
           left_img = frame[0:480, 0:640]  #----测试
           left_frame = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
           left_frame = Image.fromarray(np.uint8(left_frame))
       #--------------------------------------------------------------------------------------------------------
           frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
           # 转变成Image
           frame = Image.fromarray(np.uint8(frame))
           # 进行检测
       #------------原代码  （以注释）------- 检测输出目标物的信息，包括类别和置信度分数
           #label,frame,xy = np.array(yolo.detect_image(frame))
           frame = np.array(frame)
           # RGBtoBGR满足opencv显示格式
           frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
#------------------------------------------------------------------------------------------------------------------------
#
#                              目标检测 + 双目视觉三维定位和目标测距
#
#------------------------------------------------------------------------------------------------------------------------
           # 添加一个是否有标签的判断（如果有标签，则出现了要检测的目标物）
           #if label:
           height_0, width_0 = frame.shape[0:2]  # height_0 = 480  width_0 = 1280
           #cv2.imwrite('./stereo/result/111.bmp',frame)
           print("height_0:{},width_0:{}".format(height_0,width_0))
           iml = frame[0:int(height_0), 0:int(width_0 / 2)]
           imr = frame[0:int(height_0), int(width_0 / 2):int(width_0)]  # iml 和  imr 都是（480,640,3）的图像
           # print("***************************{}".format(iml.shape))
           # print("***************************{}".format(imr.shape))
           height, width = iml.shape[0:2]  # 左相机图片的高和宽
           config = stereoconfig_040_2.stereoCamera()  # 读取相机的参数
           #  获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
           map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
           # 畸变校正和立体校正
           iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
           # 消除畸变
           iml = undistortion(iml, config.cam_matrix_left, config.distortion_l)
           imr = undistortion(imr, config.cam_matrix_right, config.distortion_r)
           # 立体匹配
           iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
           iml_rectified_l, imr_rectified_r = rectifyImage(iml_, imr_, map1x, map1y, map2x, map2y)
           # stereoMatchSGBM视差计算（得到的左视差和右视差）
           disp, _ = stereoMatchSGBM(iml_rectified_l, imr_rectified_r, True)
           # 计算像素点的3D坐标（左相机坐标系下）*******
           points_3d = cv2.reprojectImageTo3D(disp, Q)
           print("points_3d的维度是:{}".format(points_3d.shape))
       #----------------------------------目标检测代码--------------------------------------------------
       #---------------------------------------------------------------------------------------------
           label, frame, xy = np.array(yolo.detect_image(left_frame))
           #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
           frame = np.array(frame)
           xy = np.array(xy)
           #为了调用，初始化为0一些变量，这样在后面也就可以直接对变量进行使用
           x = 0
           y = 0
           dis = 0
           x_left = 0
           x_top = 0
           x_right = 0
           #----------------是否有标签进行判断（有标签即代表检测到物体）---------------------------------------------------------------------------
           if label:
               #print(points_3d)
               #print("points_3d第二维度的最大值为：{}".format(max(points_3d[1])))
               # 这里是做了一个判断，就是对左相机的三维坐标做一个判断
       #----  x 和 y 的值  （x 和 y 应该是检测到的物体预测框的中心点）
               x = xy[0]
               print(x)
               print("x的数值为(修正前)：{}".format(x))
               y = xy[1]
               print("y的维度为(修正前)：{}".format(y))

               x_left = xy[2]
               x_top = xy[3]
               x_right = xy[4]
               #存在的问题：在没有目标物的时候，x和y为空值，也就导致下面的判断没有接收到值:解决（加了一个判断）
               #存在的问题：x和y在有目标的情况下，其坐标值是互通两个相机的，也就是两个相机并没有分开：(因为检测的图像已设置只检测左相机，所以不存在检测框互通的情况）
               #而且在对相机进行校正的时候，读取的是左右两个帧的图像，而检测目标只调用了左摄像头（已解决）
               #points_3d = (480 , 640 ,3)
               #----------
               if x == None or y == None:
                   pass
               elif x > 640:
                   pass
               else:
                   if x > 640:
                       x = 1280 - x
                       print("x的数值为(修正后)：{}".format(x))
                   count = 0
                   # try:

                   while ((points_3d[int(y), int(x), 2] < 0) | (points_3d[int(y), int(x), 2] > 2500)):
                       count += 1
                       x += count
                       if (0 < points_3d[int(y), int(x), 2] < 2300):
                           break
                       y += count
                       if (0 < points_3d[int(y), int(x), 2] < 2300):
                           break
                       count += 1
                       x -= count
                       if (0 < points_3d[int(y), int(x), 2] < 2300):
                           break
                       y -= count
                       if (0 < points_3d[int(y), int(x), 2] < 2300):
                           break
               count = 0
               #while((points_3d[int(y), int(x), 2] < 0) | (points_3d[int(y), int(x), 2] > 2500)):
               # 对x和y的值加一个限制，以免x和y的值不在图像像素内
               #1280 480  (bool((x < 640) & (y < 480))
               #2560 720  (bool((x < 1280) & (y < 720))
               #1000 320  (bool((x < 500) & (y < 320))

               while (bool((x < 640) & (y < 480))):  # out of index
                   count += 1
                   x += count
                   if (x >= 640):   #640  1280  500
                       x = 638      #638  1278  498
                       break
                   if (0 < points_3d[int(y), int(x), 2] < 2500):
                       break
                   y += count
                   if (y >= 480):   #480  720  320
                       y = 478     #478  718   318
                       break
                   if (0 < points_3d[int(y), int(x), 2] < 2500):
                       break
                   count += 1
                   x -= count
                   if (0 < points_3d[int(y), int(x), 2] < 2500):
                       break
                   y -= count
                   if (0 < points_3d[int(y), int(x), 2] < 2500):
                       break
               if (x >= 640):  #640   1280    500
                   x = 638     #638    1278   498
                   print("x is out of index!")
               if (y >= 480):   #480   720    320
                   y = 478      #478    718   318
                   print("y is out of index!")


   #-------------------------各参数依次是：图片，添加的文字，左下角坐标，字体，字体大小，颜色，字体粗细其中字体可以选择
               #frame = frame.copy()
               #text_cxy = "*"
               #frame = cv2.putText(frame, text_cxy, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
   #------------------------------------------------------------------------------------------
               #输出某个点的三维坐标
               print('修正后-点 (%d, %d) 的三维坐标 (x:%.1fcm, y:%.1fcm, z:%.1fcm)' % (int(x), int(y),
                                                                          points_3d[int(y), int(x), 0] / 10,
                                                                          points_3d[int(y), int(x), 1] / 10,
                                                                          points_3d[int(y), int(x), 2] / 10))

               dis = ((points_3d[int(y), int(x), 0] ** 2 + points_3d[int(y), int(x), 1] ** 2 + points_3d[
                   int(y), int(x), 2] ** 2) ** 0.5) / 10
               print('修正后-点 (%d, %d) 的 %s 距离左摄像头的相对距离为 %0.1f cm' % (x, y, label, dis))

       #*************************************************************************************************************************
           #-------- 原网络模型代码
           fps  = ( fps + (1./(time.time()-t1)) ) / 2
           print("fps= %.2f"%(fps))
           # **********各参数依次是：图片，添加的文字，左下角坐标，字体，字体大小，颜色，字体粗细   其中字体可以选择
           frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       #---------------新加的代码（实现在窗口显示像素点的三维坐标）----------------------------------------------------------------------
           x = int(x)
           y = int(y)
           text_cxy = "*"
       #--------------------------在窗口显示目标的像素坐标(x,y)-----------------------------
           frame = cv2.putText(frame, text_cxy, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
           if text_cxy:
               text_x = "x:%.1fcm" % (points_3d[int(y), int(x), 0] / 10)
               text_y = "y:%.1fcm" % (points_3d[int(y), int(x), 1] / 10)
               text_z = "z:%.1fcm" % (points_3d[int(y), int(x), 2] / 10)
               text_dis = "dis:%.1fcm" % dis

       #-----------------------在窗口显示像素点(x,y)的三维世界坐标--------------------------------
               x_left = int(x_left)
               x_top = int(x_top)
               x_right = int(x_right)
               cv2.putText(frame, text_x, (x_left + (x_right - x_left) + 5, x_top + 30), cv2.FONT_ITALIC, 0.5, (0, 255, 255), 1)
               cv2.putText(frame, text_y, (x_left + (x_right - x_left) + 5, x_top + 65), cv2.FONT_ITALIC, 0.5, (0, 255, 255), 1)
               cv2.putText(frame, text_z, (x_left + (x_right - x_left) + 5, x_top + 100), cv2.FONT_ITALIC, 0.5, (0, 255, 255), 1)
               cv2.putText(frame, text_dis, (x_left + (x_right - x_left) + 5, x_top + 145), cv2.FONT_ITALIC, 0.5, (0, 255, 255), 1)

       #---------------------测试代码---------将双目相机分割两个窗口显示
           frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
           #---这里实际上只是显示了左摄像机的一个窗口（作为目标检测框和对目标的三维定位（测距）框来使用）
           #1280 480 left_frame = frame[0:480, 0:640]
           #2560 720 left_frame = frame[0:720, 0:1280]
           #1000 320 left_frame = frame[0:320, 0:500]
           left_frame = frame[0:480, 0:640]
           #right_frame = frame[0:480, 640:1280]
           #---对桌面显示的窗口进行重命名（目标检测和目标定位）
           cv2.imshow("Object detection and object localization",left_frame)
           if label:
               i += 1
               name = str(label[0]).split(' ')[0]
               cv2.imwrite('./stereo/result/{}_{}.bmp'.format(name,i),frame)
           #cv2.imshow("right_video",frame)
           '''
#-------------------------------------------------------------------------------------
#---------------------------------------END-------------------------------------------
if __name__ == "__main__":
    yolo = YOLO()

    mode = "video"

#-----------------窗口可调参数设置--------------
    cv2.namedWindow("set")
    cv2.createTrackbar("num", "set", 0, 20, lambda x: None)
    cv2.createTrackbar("blockSize", "set", 1, 20, lambda x: None)
#-----------------------end----------------------------
    #----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #-----------------------------------------------------------------------------------------------------------
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #-------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    xywh = []
    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
                #cv2.imshow("detect",r_image)
    elif mode == "video":
        i  = 0
        capture = cv2.VideoCapture(0)
        #-----使用建东的双目相机
        capture1 = cv2.VideoCapture(1)
#***************************实现双目相机检测，两个同时检测目标*****************************************************************
        #1280 480
        #2560 720
        #1000  320
    #***********   以注释    *********
        #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#***********************************************************************************************************************
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)


        def cat2images(limg, rimg):
            HEIGHT = limg.shape[0]
            WIDTH = limg.shape[1]
            imgcat = np.zeros((HEIGHT, WIDTH * 2 + 20, 3))
            imgcat[:, :WIDTH, :] = limg
            imgcat[:, -WIDTH:, :] = rimg
            for i in range(int(HEIGHT / 32)):
                imgcat[i * 32, :, :] = 255
            return imgcat
        fps = 0.0
        while(True):

            t1 = time.time()

            # 读取某一帧
            ref,  frame  = capture.read()      #---右
            ref1, frame1 = capture1.read()     #---左
            # 格式转变，BGRtoRGB
        #----------------------------------------测试代码-------------------------------------------------------
            #1280 480 left[0:480, 0:640]
            #2560 720 left[0:720, 0:1280]
            #1000 320 left[0:320,0:500]
            left_img   = frame1                        #---------------------可能是左摄像头
            left_frame = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            left_frame = Image.fromarray(np.uint8(left_frame))

            right_frame = frame                        #---------------------可能是右摄像头
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
            right_frame = Image.fromarray(np.uint8(right_frame))
        #--------------------------------------------------------------------------------------------------------
            frame_left  = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame_right = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame_left  = Image.fromarray(np.uint8(frame_left))
            frame_right = Image.fromarray(np.uint8(frame_right))
            # 进行检测
        #------------原代码  （以注释）------- 检测输出目标物的信息，包括类别和置信度分数
            #label,frame,xy = np.array(yolo.detect_image(frame))
            frame_left = np.array(frame_left)
            frame_right = np.array(frame_right)

            # RGBtoBGR满足opencv显示格式
            frame_left = cv2.cvtColor(frame_left,cv2.COLOR_RGB2BGR)
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
#------------------------------------------------------------------------------------------------------------------------
#
#                              目标检测 + 双目视觉三维定位和目标测距
#
#------------------------------------------------------------------------------------------------------------------------
        #------------  SGBM算法实现  ------------
            # 添加一个是否有标签的判断（如果有标签，则出现了要检测的目标物）
            #if label:
            height_0, width_0 = frame_left.shape[0:2]  # height_0 = 480  width_0 = 1280
            #cv2.imwrite('./stereo/result/111.bmp',frame)
            print("左视图的height是:{},左视图的宽width是:{}".format(height_0,width_0))
            iml = frame_left[0:int(height_0), 0:int(width_0)]
            imr = frame_right[0:int(height_0), 0:int(width_0)]  # iml 和  imr 都是（480,640,3）的图像
            # print("***************************{}".format(iml.shape))
            # print("***************************{}".format(imr.shape))
            height, width = iml.shape[0:2]  # 左相机图片的高和宽
            config = stereoconfig_040_2.stereoCamera()  # 读取相机的参数
            #  获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
            map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
            # 畸变校正和立体校正
    #---------------   将立体校正的图像保存    -------------
            # iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
            # imgcat_out = cat2images(iml_rectified,imr_rectified)
            # cv2.imwrite('./NO_USE_CODE/imgcat_out.jpg', imgcat_out)

            # 消除畸变
            iml = undistortion(iml, config.cam_matrix_left, config.distortion_l)
            imr = undistortion(imr, config.cam_matrix_right, config.distortion_r)
            # 立体校正
            #iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
            iml_, imr_= rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
            iml_rectified_l, imr_rectified_r= preprocess(iml_, imr_)  # 灰度处理

            #SGBM立体匹配算法----stereoMatchSGBM视差计算（得到的左视差和右视差）
            #disp, _ = stereoMatchSGBM(iml_rectified_l, imr_rectified_r, False)
            #------newSGBM算法
            disp = new_SGBM(iml_rectified_l, imr_rectified_r, False)
            # -----BM立体匹配算法
            #disp = stereoMatchBM(iml_rectified_l,imr_rectified_r)
            # 计算像素点的3D坐标（左相机坐标系下）******
            points_3d = cv2.reprojectImageTo3D(disp, Q)
            print("points_3d的维度是:{}".format(points_3d.shape))
        #----------------------------------目标检测代码--------------------------------------------------
        #---------------------------------------------------------------------------------------------
            label, frame,xy = np.array(yolo.detect_image(left_frame))
            #print("xy:{}".format(xy))
            print("label:{}".format(label))
            #label, frame,xy,list_top,list_left,list_bottom,list_right = np.array(yolo.detect_image(left_frame))
            # xywh.append(list_top)
            # xywh.append(list_left)
            # xywh.append(list_bottom)
            # xywh.append(list_right)
            # print("xywh:{}".format(xywh))
#————————————————————————————————————————————————————————————————————
            frame = np.array(frame)
            # xy = np.array(xy)
            # #为了调用，初始化为0一些变量，这样在后面也就可以直接对变量进行使用
            x = 0
            y = 0
            # dis = 0
            x_left = 0
            x_top = 0
            x_right = 0
            x_bottom = 0
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            #----------------是否有标签进行判断（有标签即代表检测到物体）---------------------------------------------------------------------------
            #xy = [top,left,bottom,right]
            if label:
                list  = np.array(xy).reshape(-1,4)     #    n行4列
                #label = np.array(label).reshape(-1,1)  #    n行1列   label包含：标签和置信度
                #print("label_one:{}".format(label))
                #  循环n行（次）
                for temp_list in list:

                    y_temp = 0
                    x_temp = 0
                    idx = 0
                    for values in temp_list:

                        idx += 1
                        if idx == 1:
                            x_top = values
                            y_temp += values
                        if idx == 2:
                            x_left = values
                            x_temp += values

                        if idx == 3:
                            x_bottom = values
                            y_temp += values
                        if idx == 4:
                            x_right = values
                            x_temp += values
                        #print("values:{}".format(values[0]]))
                        # print(f"values: ->  {values}")
                    print(f"修正前x的值为: {x_temp // 2}")
                    print(f"修正前y的值为: {y_temp // 2}")
                    x = (x_temp // 2)
                    y = (y_temp // 2)

                    print("x_top:{}".format(x_top))
                    '''
                    if x == None or y == None:
                        pass
                    elif x > 640:
                        pass
                    else:
                        if x > 640:
                            x = 1280 - x
                            print("x的数值为(修正后)：{}".format(x))
                        count = 0
                        # try:
        
                        while ((points_3d[int(y), int(x), 2] < 0) | (points_3d[int(y), int(x), 2] > 2500)):
                            count += 1
                            x += count
                            if (0 < points_3d[int(y), int(x), 2] < 2300):
                                break
                            y += count
                            if (0 < points_3d[int(y), int(x), 2] < 2300):
                                break
                            count += 1
                            x -= count
                            if (0 < points_3d[int(y), int(x), 2] < 2300):
                                break
                            y -= count
                            if (0 < points_3d[int(y), int(x), 2] < 2300):
                                break
                    '''
                    count = 0
                    #while((points_3d[int(y), int(x), 2] < 0) | (points_3d[int(y), int(x), 2] > 2500)):
                    # 对x和y的值加一个限制，以免x和y的值不在图像像素内
                    #1280 480  (bool((x < 640) & (y < 480))
                    #2560 720  (bool((x < 1280) & (y < 720))
                    #1000 320  (bool((x < 500) & (y < 320))

                    while (bool(( x_left< x < x_right) & ( x_top< y < x_bottom))):  # out of index
                        count += 1
                        x += count
                        if (x >= x_right):   #640  1280  500
                            x = x_right - 2     #638  1278  498
                            break
                        if (x <= x_left):
                            x = x_left + 1
                            break
                        if (0 < points_3d[int(y), int(x), 2] < 2500):
                            break
                        y += count
                        if (y >= x_bottom):   #480  720  320
                            y = x_bottom - 2     #478  718   318
                            break
                        #if (y <= x_top):
                        #    y = x_top + 2
                        if (0 < points_3d[int(y), int(x), 2] < 2500):
                            break
                        count += 1
                        x -= count
                        if (0 < points_3d[int(y), int(x), 2] <2500):
                            break
                        y -= count
                        if (0 < points_3d[int(y), int(x), 2] < 2500):
                            break
                    if (x >= x_right):  #640   1280    500
                        x = x_right - 2     #638    1278   498
                        print("x is out of index!")
                    if (y >= x_bottom):   #480   720    320
                        y = x_bottom - 2      #478    718   318
                        print("y is out of index!")



        #-------------------------各参数依次是：图片，添加的文字，左下角坐标，字体，字体大小，颜色，字体粗细其中字体可以选择
                    #frame = frame.copy()
                    #text_cxy = "*"
                    #frame = cv2.putText(frame, text_cxy, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #------------------------------------------------------------------------------------------
                    #输出某个点的三维坐标
                    print('修正后-点 (%d, %d) 的三维坐标 (x:%.1fcm, y:%.1fcm, z:%.1fcm)' % (int(x), int(y),
                                                                               points_3d[int(y), int(x), 0] / 10,
                                                                               points_3d[int(y), int(x), 1] / 10,
                                                                               points_3d[int(y), int(x), 2] / 10))

                    dis = ((points_3d[int(y), int(x), 0] ** 2 + points_3d[int(y), int(x), 1] ** 2 + points_3d[
                        int(y), int(x), 2] ** 2) ** 0.5) / 10
                    print('修正后-点 (%d, %d) 的 %s 距离左摄像头的相对距离为 %0.1f cm' % (x, y, label, dis))

    #————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
                    x = int(x)
                    y = int(y)
                    text_cxy = "*"
                    # --------------------------在窗口显示目标的像素坐标(x,y)-----------------------------
                    frame = cv2.putText(frame, text_cxy, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if text_cxy:
                        text_x = "x:%.1fcm" % (points_3d[int(y), int(x), 0] / 10)
                        text_y = "y:%.1fcm" % (points_3d[int(y), int(x), 1] / 10)
                        text_z = "z:%.1fcm" % (points_3d[int(y), int(x), 2] / 10)
                        text_dis = "dis:%.1fcm" % dis

                        # -----------------------在窗口显示像素点(x,y)的三维世界坐标--------------------------------
                        x_left = int(x_left)
                        x_top = int(x_top)
                        x_right = int(x_right)
                        cv2.putText(frame, text_x, (x_left + (x_right - x_left) + 5, x_top + 30), cv2.FONT_ITALIC, 0.5,
                                    (0, 255, 255), 1)
                        cv2.putText(frame, text_y, (x_left + (x_right - x_left) + 5, x_top + 65), cv2.FONT_ITALIC, 0.5,
                                    (0, 255, 255), 1)
                        cv2.putText(frame, text_z, (x_left + (x_right - x_left) + 5, x_top + 100), cv2.FONT_ITALIC, 0.5,
                                    (0, 255, 255), 1)
                        cv2.putText(frame, text_dis, (x_left + (x_right - x_left) + 5, x_top + 145), cv2.FONT_ITALIC,
                                    0.5, (0, 255, 255), 1)

                #*************************************************************************************************************************
                #-------- 原网络模型代码
                fps  = ( fps + (1./(time.time()-t1)) ) / 2
                print("fps= %.2f"%(fps))
                # **********各参数依次是：图片，添加的文字，左下角坐标，字体，字体大小，颜色，字体粗细   其中字体可以选择
                frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #---------------新加的代码（实现在窗口显示像素点的三维坐标）----------------------------------------------------------------------
            #     x = int(x)
            #     y = int(y)
            #     text_cxy = "*"
            # #--------------------------在窗口显示目标的像素坐标(x,y)-----------------------------
            #     frame = cv2.putText(frame, text_cxy, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #     if text_cxy:
            #         text_x = "x:%.1fcm" % (points_3d[int(y), int(x), 0] / 10)
            #         text_y = "y:%.1fcm" % (points_3d[int(y), int(x), 1] / 10)
            #         text_z = "z:%.1fcm" % (points_3d[int(y), int(x), 2] / 10)
            #         text_dis = "dis:%.1fcm" % dis
            #
            # #-----------------------在窗口显示像素点(x,y)的三维世界坐标--------------------------------
            #         x_left = int(x_left)
            #         x_top = int(x_top)
            #         x_right = int(x_right)
            #         cv2.putText(frame, text_x, (x_left + (x_right - x_left) + 5, x_top + 30), cv2.FONT_ITALIC, 0.5, (0, 255, 255), 1)
            #         cv2.putText(frame, text_y, (x_left + (x_right - x_left) + 5, x_top + 65), cv2.FONT_ITALIC, 0.5, (0, 255, 255), 1)
            #         cv2.putText(frame, text_z, (x_left + (x_right - x_left) + 5, x_top + 100), cv2.FONT_ITALIC, 0.5, (0, 255, 255), 1)
            #         cv2.putText(frame, text_dis, (x_left + (x_right - x_left) + 5, x_top + 145), cv2.FONT_ITALIC, 0.5, (0, 255, 255), 1)

            #---------------------测试代码---------将双目相机分割两个窗口显示
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                #frame11 = cv2.cvtColor(frame11, cv2.COLOR_RGB2BGR)
                #---这里实际上只是显示了左摄像机的一个窗口（作为目标检测框和对目标的三维定位（测距）框来使用）
                #1280 480 left_frame = frame[0:480, 0:640]
                #2560 720 left_frame = frame[0:720, 0:1280]
                #1000 320 left_frame = frame[0:320, 0:500]
                left_frame = frame[0:480, 0:640]
                #right_frame = frame11[0:480, 0:640]
                #---对桌面显示的窗口进行重命名（目标检测和目标定位）
                cv2.imshow("Object detection and object localization", left_frame)
                #---------------------  将检测到的实时画面保存为图片 -----------------------
                #if label:
                #   i += 1
                #   name = str(label[0]).split(' ')[0]
                #   cv2.imwrite('./stereo/result/left/{}_{}.bmp'.format(name,i),left_frame)
                #   cv2.imwrite('./stereo/result/right/{}_{}.bmp'.format(name, i), right_frame)
    #---------------------------------------------------------------------------------------------

#---------------------------------------   目标检测  + 三维定位  +  测距  -------------------------------------------------------
#                                                      任务完成
#-----------------------------------------------------------------------------------------------------------------------------
            c = cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")