
## Abstract
This is a tensorflow re-implementation of Faster rcnn by LongJun<br>
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)<br>

With VGG16(conv5_3):
Train on VOC2007 trainval and test on VOC2007 test, the mean_ap is 71.0%<br>
Train on VOC2007+VOC2012 trainval and test on VOC2007 test, the mean_ap is 76.3%<br>
(You can change the train set and test set for more results)<br>
More backbone network will be supported in the feature

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

# Pascal voc data_set download
[VOC2007_trainval_06](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)<br>
[VOCtest_06-Nov-2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)<br>
[VOCtrainval_11-May-2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)<br>

# Data Format
```
├── VOCdevkit
│   ├── VOC2007_trainval
│       ├── Annotation
│       ├── JPEGImages
|       ├── ImageSets
│   ├── VOC2007_test
│       ├── Annotation
│       ├── JPEGImages
|       ├── ImageSets
│   ├── VOC2012_trainval
│       ├── Annotation
│       ├── JPEGImages
│       ├── ImageSets
```
# Download pretrained Model
You can download pretrained model and put into $PATH_ROOT/output, and test on pascal data set or your own data set(Please convert your data set into pascal format before testing):<br>
```
For map_compute:
cd $PATH_ROOT
python test.py
```
```
For img_show:
cd $PATH_ROOT
python test_show_image.py
```


