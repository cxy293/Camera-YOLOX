# Camera-YOLOX
YOLOX project that can range distance with binocular camera
YOLOX code carried from Mr. Bubbliiiing, on the basis of this code added binocular ranging program.
The main principle is as follows: when the computer is connected to the binocular camera, the YOLOX code detects the target through the binocular camera (at the same time, the left and right cameras can calculate the left and right disparity according to the image read);
When a target is detected, the YOLOX detection code will give the target a prediction box. The center point of the box is used to calculate the disparity.At this time, the three-dimensional world coordinates of the detected object are obtained.
可以用双目相机测距的YOLOX项目。来自Bubbliiiing先生的YOLOX代码，在此代码的基础上增加了双目测距程序。主要原理如下：当电脑连接双目摄像头时，YOLOX代码通过双目摄像头检测目标（同时左右摄像头可以根据读取的图像计算左右视差）；当检测到目标时，YOLOX 检测代码会给目标一个预测框，而框的中心点用于计算视差，此时得到的就是被检测目标的三维世界坐标。
## 双目测距代码准备
在该项目的stereo文件夹下，需要自己对stereo_config.py代码中的双目相机参数进行修改。（只需要在原有的代码中，填入自己的数据，双目相机的一些参数可以通过matlab自带的双目标定APP得到），修改好后，训练好代码，就可以**使用**predict.py同时进行检测及测距。
## 使用
双目测距的代码主要是在predict.py文件中，由于对预测文件进行了修改，可以通过predict_one_point.py进行目标物中心点的三维测距和检测，也可以通过predict_three_point.py进行目标物“上 中 下”三个点的三维测距和检测。
## 训练
YOLOX网络的训练和Mr. Bubbliiiing先生的代码一模一样，故只需要参考其仓库的YOLOX代码训练教程即可。
（注：自己根据自己的任务建立model_data文件夹：用于存放预训练权值和voc_lass标签目录或coco_class标签目录；建立logs文件夹，用于存放训练过程中的权值文件）第一次公开以及上传自己的项目，有点小问题
## 训练结果
首先阐述一下存在的问题：立体匹配算法为opencv中实现的SGBM算法，当然通过参数微调和最终定位点的有效查找，一定是可以达到有效的定位精度。在我的实验中，并不是一个大项目，只算是小实验，识别定位地上的水瓶，整体来说，误差在10cm以内，若是没有其他干扰物的存在下，误差会更小。改进的地方还有很多，比如更换立体匹配算法等。最关键的还是关键点的获取，这里可能有小伙伴不懂，这个关键点是什么？为此多解释一下：定位的先决条件是先得到目标检测的预测框，此时左右相机计算视差，要在预测框范围内寻找一点，该点通过多个坐标转换能够代表该目标物距离我们最准确的距离，所以这个关键点的获取很关键，代码中也有实现，感兴趣的朋友可以深入研究。
