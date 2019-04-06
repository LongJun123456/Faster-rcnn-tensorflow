
# Abstract
## This is a tensorflow re-implementation of Faster rcnn by LongJun<br>
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)<br>

With VGG16(conv5_3):<br>
Train on VOC2007 trainval and test on VOC2007 test, the mean_ap is 71.0%<br>
Train on VOC2007+VOC2012 trainval and test on VOC2007 test, the mean_ap is 76.3%<br>
(You can change the train set and test set for more results)<br>
More backbone network(such as Resnet) will be supported in the feature

Test image show:
![test_img](https://github.com/LongJun123456/Faster-rcnn-tensorflow/blob/master/test_img/87.jpg)
![test_img](https://github.com/LongJun123456/Faster-rcnn-tensorflow/blob/master/test_img/39.jpg)
![test_img](https://github.com/LongJun123456/Faster-rcnn-tensorflow/blob/master/test_img/9.jpg)
![test_img](https://github.com/LongJun123456/Faster-rcnn-tensorflow/blob/master/test_img/138.jpg)
![test_img](https://github.com/LongJun123456/Faster-rcnn-tensorflow/blob/master/test_img/140.jpg)
# My Development Environment
1.python 3.6(anaconda recommend)<br>
2.cuda9.0<br>
3.opencv(cv2)<br>
4.tfplot(You might need to install this lib by pip)<br>
5.tensorflow==1.9<br>

# Pascal VOC data_set download(if you only train your net on VOC2007, you can just download VOC2007_trainval_06 and VOCtest_06-Nov-2007)
[VOC2007_trainval_06](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)<br>
[VOCtest_06-Nov-2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)<br>
[VOCtrainval_11-May-2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)<br>

Please change the data format as fellow 
```
├── dataset
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
# Download pretrained Model<br>
[Model download](https://pan.baidu.com/s/1RWXD_aSB3rqGcXc5XeNltw). pass key is c0eb <br>
(If you want to train your own model, you can skip this step)<br>
You can download pretrained model and put into $PATH_ROOT/output after uncompressing<br>
You can also train your own model by the method below<br>
test on pascal data set or your own data set(Please convert your data set into pascal format before testing):<br>
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
# Train your own model
1. Download VGG16 pretrained weights [VGG16_weights](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)
2. uncompress the VGG16 pretrained weights and put the weight into folder 'model_pretrained'
3. Modify parameters (such as train_imdb_name, test_imdb_name, MAX_ITER, etc.) in $PATH_ROOT/config.py
Then input:
```
cd $PATH_ROOT
python train.py
```

# Tensorboard
```
tensorboard --logdir $SUMMARY_PATH
```
The summary is in the folder 'summary_out/year_month_day_hour_minute', For example:<br>
tensorboard --logdir /LongJun/Faster-rcnn-tensorflow/summary_out/2018_04_19_21_13   You need to choose the latest folder so you can see the newest summary
![tensorboard](https://github.com/LongJun123456/Faster-rcnn-tensorflow/blob/master/test_img/11.png)
![tensorboard](https://github.com/LongJun123456/Faster-rcnn-tensorflow/blob/master/test_img/22.png)

# Eval and test
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
You can change the img_save_num for the num of image showing in config.py, default 2

# End
More dection network  re-implementation of tensorflow will be uploaded in the feature, FPN,YOLO..<br>
If it helps, please give me a star. Thank you very much



