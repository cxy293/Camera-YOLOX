#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
#**********************双目测距调用的函数*******************************
from stereo import stereoconfig_040_2
from stereo.dianyuntu_yolo import preprocess, undistortion, getRectifyTransform, draw_line, rectifyImage,\
    stereoMatchSGBM, hw3ToN3, DepthColor2Cloud, view_cloud,stereoMatchBM

if __name__ == "__main__":
    yolo = YOLO()

    mode            = "video"

    #---------------------双目匹配算法参数设置----------------
    # -----------------窗口可调参数设置--------------
    cv2.namedWindow("set")
    cv2.createTrackbar("num", "set", 0, 20, lambda x: None)
    cv2.createTrackbar("blockSize", "set", 1, 20, lambda x: None)
    # -----------------------end----------------------------

    crop            = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #----------------------------------------------------------------------------------------------------------#
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

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop)
                r_image.show()

    elif mode == "video":
        #capture = cv2.VideoCapture(video_path)
    #-----------------------双目调用-------------------------
        i = 0
        capture = cv2.VideoCapture(0)
        # -----使用建东的双目相机
        capture1 = cv2.VideoCapture(1)
    #------------------------------------------------------
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    #-------------------------------画图---------------------------
        def cat2images(limg, rimg):
            HEIGHT = limg.shape[0]
            WIDTH = limg.shape[1]
            imgcat = np.zeros((HEIGHT, WIDTH * 2 + 20, 3))
            imgcat[:, :WIDTH, :] = limg
            imgcat[:, -WIDTH:, :] = rimg
            for i in range(int(HEIGHT / 32)):
                imgcat[i * 32, :, :] = 255
            return imgcat
    #--------------------------------------------------------------
        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            ref1, frame1 = capture1.read()
            # 格式转变，BGRtoRGB
            # ----------------------------------------测试代码-------------------------------------------------------
            # 1280 480 left[0:480, 0:640]
            # 2560 720 left[0:720, 0:1280]
            # 1000 320 left[0:320,0:500]
            left_img = frame  # ----测试
            left_frame = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            left_frame = Image.fromarray(np.uint8(left_frame))

            right_frame = frame1
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
            right_frame = Image.fromarray(np.uint8(right_frame))
            # --------------------------------------------------------------------------------------------------------
            frame_left = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_right = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame_left = Image.fromarray(np.uint8(frame_left))
            frame_right = Image.fromarray(np.uint8(frame_right))
            # 进行检测
            # ------------原代码  （以注释）------- 检测输出目标物的信息，包括类别和置信度分数
            # label,frame,xy = np.array(yolo.detect_image(frame))
            frame_left = np.array(frame_left)
            frame_right = np.array(frame_right)

            # RGBtoBGR满足opencv显示格式
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
            # ------------------------------------------------------------------------------------------------------------------------
            #
            #                              目标检测 + 双目视觉三维定位和目标测距
            #
            # ------------------------------------------------------------------------------------------------------------------------
            # ------------  SGBM算法实现  ------------
            # 添加一个是否有标签的判断（如果有标签，则出现了要检测的目标物）
            #if label:
            height_0, width_0 = frame_left.shape[0:2]  # height_0 = 480  width_0 = 1280
            # cv2.imwrite('./stereo/result/111.bmp',frame)
            print("height_0:{},width_0:{}".format(height_0, width_0))
            iml = frame_left[0:int(height_0), 0:int(width_0)]
            imr = frame_right[0:int(height_0), 0:int(width_0)]  # iml 和  imr 都是（480,640,3）的图像
            # print("***************************{}".format(iml.shape))
            # print("***************************{}".format(imr.shape))
            height, width = iml.shape[0:2]  # 左相机图片的高和宽
            config = stereoconfig_040_2.stereoCamera()  # 读取相机的参数
            #  获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
            map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
            # 畸变校正和立体校正
            iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
            # ---------------   将立体校正的图像保存    -------------
            imgcat_out = cat2images(iml_rectified, imr_rectified)
            cv2.imwrite('./NO_USE_CODE/imgcat_out.jpg', imgcat_out)

            # 消除畸变
            iml = undistortion(iml, config.cam_matrix_left, config.distortion_l)
            imr = undistortion(imr, config.cam_matrix_right, config.distortion_r)
            # 立体校正
            iml_ , imr_ = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
            #---灰度处理
            iml_rectified_l, imr_rectified_r = preprocess(iml_, imr_)  # 预处理，一般可以削弱光照不均的影响，不做也可以
            # SGBM立体匹配算法----stereoMatchSGBM视差计算（得到的左视差和右视差）
            disp, _ = stereoMatchSGBM(iml_rectified_l, imr_rectified_r, False)
            # -----BM立体匹配算法
            # disp = stereoMatchBM(iml_rectified_l,imr_rectified_r)
            # 计算像素点的3D坐标（左相机坐标系下）******
            points_3d = cv2.reprojectImageTo3D(disp, Q)
            print("points_3d的维度是:{}".format(points_3d.shape))
            # ----------------------------------目标检测代码--------------------------------------------------
            # ---------------------------------------------------------------------------------------------
            label, frame, xy = np.array(yolo.detect_image(left_frame))
            print("xy:{}".format(xy))
            # label, frame,xy,list_top,list_left,list_bottom,list_right = np.array(yolo.detect_image(left_frame))
            # xywh.append(list_top)
            # xywh.append(list_left)
            # xywh.append(list_bottom)
            # xywh.append(list_right)
            # print("xywh:{}".format(xywh))
            # ————————————————————————————————————————————————————————————————————
            frame = np.array(frame)
            # xy = np.array(xy)
            # #为了调用，初始化为0一些变量，这样在后面也就可以直接对变量进行使用
            x = 0
            y = 0
            x1 = 0
            y1 = 0
            x2 = 0
            y2 = 0
            # dis = 0
            x_left = 0
            x_top = 0
            x_right = 0
            x_bottom = 0
            # ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ----------------是否有标签进行判断（有标签即代表检测到物体）---------------------------------------------------------------------------
            # xy = [top,left,bottom,right]
            if xy:
                list = np.array(xy).reshape(-1, 4)  # n行4列
                print("list:{}".format(list))
                label = np.array(label).reshape(-1, 1)  # n行1列   label包含：标签和置信度
                print("label:{}".format(label))
                # print("label_one:{}".format(label))
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
                        # print("values:{}".format(values[0]]))
                        # print(f"values: ->  {values}")
                    print(f"修正前x的值为: {x_temp // 2}")
                    print(f"修正前y的值为: {y_temp // 2}")
                    x = (x_temp // 2)
                    y = (y_temp // 2)
                    #print("x_top:{}".format(x_top))
    #--------------------------边缘点----想在这加一个判断-------
                    if (x_right - x_left) > (x_bottom - x_top):
                        x1 = x_left + 1
                        y1 = (x_bottom + x_top) / 2
                        x2 = x_right - 5
                        y2 = (x_bottom + x_top) / 2
                    else:
                        x1 = (x_left + x_right) / 2
                        y1 = x_top + 2
                        x2 = (x_left + x_right) / 2
                        y2 = x_bottom - 2


                    count = 0
                    # while((points_3d[int(y), int(x), 2] < 0) | (points_3d[int(y), int(x), 2] > 2500)):
                    # 对x和y的值加一个限制，以免x和y的值不在图像像素内
                    # 1280 480  (bool((x < 640) & (y < 480))
                    # 2560 720  (bool((x < 1280) & (y < 720))
                    # 1000 320  (bool((x < 500) & (y < 320))

                    while (bool((x_left < x < x_right) & (x_top < y < x_bottom))):  # out of index
                        count += 1
                        x += count
                        if (x >= x_right):  # 640  1280  500
                            x = x_right - 2  # 638  1278  498
                            break
                        if (x <= x_left):
                            x = x_left + 1
                            break
                        if (0 < points_3d[int(y), int(x), 2] < 2500):
                            break
                        y += count
                        if (y >= x_bottom):  # 480  720  320
                            y = x_bottom - 2  # 478  718   318
                            break
                        # if (y <= x_top):
                        #    y = x_top + 2
                        if (0 < points_3d[int(y), int(x), 2] < 2500):
                            break
                        count += 1
                        x -= count
                        if (0 < points_3d[int(y), int(x), 2] < 2500):
                            break
                        y -= count
                        if (0 < points_3d[int(y), int(x), 2] < 2500):
                            break
                    if (x >= x_right):  # 640   1280    500
                        x = x_right - 2  # 638    1278   498
                        print("x is out of index!")
                    if (y >= x_bottom):  # 480   720    320
                        y = x_bottom - 2  # 478    718   318
                        print("y is out of index!")


                    # -------------------------各参数依次是：图片，添加的文字，左下角坐标，字体，字体大小，颜色，字体粗细其中字体可以选择
                    # frame = frame.copy()
                    # text_cxy = "*"
                    # frame = cv2.putText(frame, text_cxy, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # ------------------------------------------------------------------------------------------
                    # 输出某个点的三维坐标
                    print('修正后-点 (%d, %d) 的三维坐标 (x:%.1fcm, y:%.1fcm, z:%.1fcm)' % (int(x), int(y),
                                                                                   points_3d[int(y), int(x), 0] / 10,
                                                                                   points_3d[int(y), int(x), 1] / 10,
                                                                                   points_3d[int(y), int(x), 2] / 10))

                    dis = ((points_3d[int(y), int(x), 0] ** 2 + points_3d[int(y), int(x), 1] ** 2 + points_3d[
                        int(y), int(x), 2] ** 2) ** 0.5) / 10
                    print('修正后-点 (%d, %d) 的 %s 距离左摄像头的相对距离为 %0.1f cm' % (x, y, label, dis))
        # -------------------  对x,y中心点做修改  ---------------------
                    dis = ((points_3d[int(y1), int(x1), 0] ** 2 + points_3d[int(y1), int(x1), 1] ** 2 + points_3d[
                        int(y1), int(x1), 2] ** 2) ** 0.5) / 10
                    dis1 = ((points_3d[int(y2), int(x2), 0] ** 2 + points_3d[int(y2), int(x2), 1] ** 2 + points_3d[
                        int(y2), int(x2), 2] ** 2) ** 0.5) / 10
                    dis2 = ((points_3d[int(y), int(x), 0] ** 2 + points_3d[int(y), int(x), 1] ** 2 + points_3d[
                        int(y), int(x), 2] ** 2) ** 0.5) / 10
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
        #------------------------------------------------------------
        # ---------------------------在窗口显示目标像素的坐标中心点(x,y)-------------------------
                    text_cxy = "*"
                    frame = cv2.putText(frame, text_cxy, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 125), 1)

                    # --------------------------在窗口显示目标的像素坐标(x1,y1)-----------------------------
                    text_cxy = "*"
                    frame = cv2.putText(frame, text_cxy, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # --------------------------在窗口显示目标的像素坐标(x2,y2)-----------------------------
                    text_cxy = "*"
                    frame = cv2.putText(frame, text_cxy, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        #---------------------------------------------------------------------
                    if text_cxy:
                        # -----x1点坐标--------   "x:%.1fcm"，，"y:%.1fcm"，，"z:%.1fcm"
                        text_x = "x:%.1f" % (points_3d[int(y1), int(x1), 0] / 10)
                        text_y = "y:%.1f" % (points_3d[int(y1), int(x1), 1] / 10)
                        text_z = "z:%.1f" % (points_3d[int(y1), int(x1), 2] / 10)
                        text_dis = "dis:%.1fcm" % dis
                        # -----x2点坐标
                        text_x1 = "x:%.1f" % (points_3d[int(y2), int(x2), 0] / 10)
                        text_y1 = "y:%.1f" % (points_3d[int(y2), int(x2), 1] / 10)
                        text_z1 = "z:%.1f" % (points_3d[int(y2), int(x2), 2] / 10)
                        text_dis1 = "dis:%.1fcm" % dis1
                        # -----x_center点坐标
                        text_xcenter = "x:%.1f" % (points_3d[int(y), int(x), 0] / 10)
                        text_ycenter = "y:%.1f" % (points_3d[int(y), int(x), 1] / 10)
                        text_zcenter = "z:%.1f" % (points_3d[int(y), int(x), 2] / 10)
                        text_dis_center = "dis:%.1fcm" % dis2

            # -----------------------在窗口显示像素点(x,y)的三维世界坐标--------------------------------
                        x_left = int(x_left)
                        x_top = int(x_top)
                        x_right = int(x_right)
            # ---------------------绘制x1点在视窗口的显示
                        cv2.putText(frame, text_x, (x_left + (x_right - x_left) + 5, x_top + 10), cv2.FONT_ITALIC, 0.5,
                                    (0, 0, 255), 1)
                        cv2.putText(frame, text_y, (x_left + (x_right - x_left) + 5, x_top + 30), cv2.FONT_ITALIC, 0.5,
                                    (0, 0, 255), 1)
                        cv2.putText(frame, text_z, (x_left + (x_right - x_left) + 5, x_top + 50), cv2.FONT_ITALIC, 0.5,
                                    (0, 0, 255), 1)
                        cv2.putText(frame, text_dis, (x_left + (x_right - x_left) + 5, x_top + 65), cv2.FONT_ITALIC,
                                    0.5, (0, 0, 255), 1)
                        # ---------------------绘制x2点在视窗口的显示
                        cv2.putText(frame, text_x1, (x_left + (x_right - x_left) + 5, x_top + 80), cv2.FONT_ITALIC, 0.5,
                                    (0, 255, 255), 1)
                        cv2.putText(frame, text_y1, (x_left + (x_right - x_left) + 5, x_top + 100), cv2.FONT_ITALIC,
                                    0.5,
                                    (0, 255, 255), 1)
                        cv2.putText(frame, text_z1, (x_left + (x_right - x_left) + 5, x_top + 120), cv2.FONT_ITALIC,
                                    0.5,
                                    (0, 255, 255), 1)
                        cv2.putText(frame, text_dis1, (x_left + (x_right - x_left) + 5, x_top + 140), cv2.FONT_ITALIC,
                                    0.5, (0, 255, 255), 1)
                        # --------------------绘制x_center点在视窗口的显示
                        cv2.putText(frame, text_xcenter, (x_temp // 2 - 60, y_temp // 2), cv2.FONT_ITALIC, 0.5,
                                    (255, 0, 125), 1)
                        cv2.putText(frame, text_ycenter, (x_temp // 2 - 60, y_temp // 2 + 15), cv2.FONT_ITALIC,
                                    0.5,
                                    (255, 0, 125), 1)
                        cv2.putText(frame, text_zcenter, (x_temp // 2 - 60, y_temp // 2 + 30), cv2.FONT_ITALIC,
                                    0.5,
                                    (255, 0, 125), 1)
                        cv2.putText(frame, text_dis_center, (x_temp // 2 - 60, y_temp // 2 + 45), cv2.FONT_ITALIC,
                                    0.5, (255, 0, 125), 1)

                fps  = ( fps + (1./(time.time()-t1)) ) / 2
                print("fps= %.2f"%(fps))
                frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # frame11 = cv2.cvtColor(frame11, cv2.COLOR_RGB2BGR)
                # ---这里实际上只是显示了左摄像机的一个窗口（作为目标检测框和对目标的三维定位（测距）框来使用）
                # 1280 480 left_frame = frame[0:480, 0:640]
                # 2560 720 left_frame = frame[0:720, 0:1280]
                # 1000 320 left_frame = frame[0:320, 0:500]
                left_frame = frame[0:480, 0:640]
                # right_frame = frame11[0:480, 0:640]
                # ---对桌面显示的窗口进行重命名（目标检测和目标定位）
                cv2.imshow("Object detection and object localization", left_frame)

                # ---------------------  将检测到的实时画面保存为图片 -----------------------
                if xy:
                  i += 1
                  name = str(label[0]).split(' ')[0]
                  cv2.imwrite('./stereo/result/left/{}_{}.bmp'.format(name,i),left_frame)
                  #cv2.imwrite('./stereo/result/right/{}_{}.bmp'.format(name, i), right_frame)


            #cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        # print("Video Detection Done!")
        # capture.release()
        # if video_save_path!="":
        #     print("Save processed video to the path :" + video_save_path)
        #     out.release()
        # cv2.destroyAllWindows()
        
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
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
