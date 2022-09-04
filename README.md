# Camera-YOLOX
YOLOX project that can range distance with binocular camera
YOLOX code carried from Mr. Bubbliiiing, on the basis of this code added binocular ranging program.
The main principle is as follows: when the computer is connected to the binocular camera, the YOLOX code detects the target through the binocular camera (at the same time, the left and right cameras can calculate the left and right disparity according to the image read);
When a target is detected, the YOLOX detection code will give the target a prediction box. The center point of the box is used to calculate the disparity.
