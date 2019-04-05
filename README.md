
## Abstract
This is a tensorflow re-implementation of Faster rcnn by LongJun<br>
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)<br>

Train on VOC2007 trainval and test on VOC2007 test, the mean_ap is 71.0%<br>
Train on VOC2007+VOC2012 trainval and test on VOC2007 test, the mean_ap is <br>
(You can change the train set and test set for more results)<br>
Test image show:
![test_img](https://github.com/LongJun123456/Faster-rcnn-tensorflow/blob/master/test_img/148.jpg)
![test_img](https://github.com/LongJun123456/Faster-rcnn-tensorflow/blob/master/test_img/39.jpg)
![test_img](https://github.com/LongJun123456/Faster-rcnn-tensorflow/blob/master/test_img/9.jpg)
![test_img](https://github.com/LongJun123456/Faster-rcnn-tensorflow/blob/master/test_img/70.jpg)
# My Development Environment
1.python 3.6(anaconda recommend)<br>
2.cuda9.0<br>
3.opencv(cv2)<br>
4.tfplot(You might need to install this lib by pip)<br>
5.tensorflow==1.9<br>


