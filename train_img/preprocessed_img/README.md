# caffe模型训练、测试步骤

### 本项目通过级联的方式，通过减少网络深度来达到减少计算量的目的、提高速度的目的。需要训练23个独立的网络，网络可以分为3个层次。

## 层次1: 

对F1全脸(39×39)，NM1鼻嘴区域(39×31)，EN1眼鼻区域(39×31)的网络.
![](https://github.com/CongWeilin/DCNN-caffe/blob/master/tools/net.png)
训练方法：使用`tools/cropface.py`进行人脸采割,修改参数可以裁剪缩放出来(39×39)(39×31)的图片.通过当前文件夹`show_face_key_points.py`可以看点的位置是否正确.确保正确的点的位置之后，通过`convert_imagelist_2_hdf5.py`转换数据格式成hdf5.进入`train_img/preprocessed_img/F1/caffe`,运行`./train.sh`进行训练,训练日志保存在`LOG`文件夹下。可以使用caffe自带工具看loss下降曲线，为了方便我拷贝到了`tools`文件夹下。首先把LOG文件夹下的日志重命名为`*.log`的格式(去掉后缀就可以)

`./parse_log.py LOG/*.log` `./plot_training_log.py.example 0 result.png *.log`就可以看效果了

## 层次2：

通过两个15×15的网络对第一层输出的每一个特征点进行更精确的位置估计.例如LE21和LE22分别对左眼进行两次不同尺度的估计,对上一次的点进行校准.
      
训练方法：使用`tools/cropface_Organ.py`进行人脸采割,修改参数可以裁剪并缩放出来15×15的图片。然后通过当前文件夹下`show_face_key_points.py`可以看点的位置是否正确。确保正确的点的位置之后，通过`convert_imagelist_2_hdf5_organ.py`转换数据格式成hdf5(caffe支持的多标签格式).进入`train_img/preprocessed_img/LE21/caffe`,运行`./train.sh`进行训练,训练日志保存在LOG文件夹下.可以使用caffe自带工具看loss下降曲线，为了方便我拷贝到了`tools`文件夹下。首先把`LOG`文件夹下的日志重命名为`*.log`的格式(去掉后缀就可以)
 
`./parse_log.py LOG/*.log` `./plot_training_log.py.example 0 result.png *.log`就可以看效果了

## 层次3: 

通过两个15×15的网络对第一层输出的每一个特征点进行更精确的位置估计。例如LE31和LE32分别对左眼进行两次不同尺度的估计，对上一次的点进行校准。
      
训练方法：使用`tools/cropface_Organ.py`进行人脸采割，通过修改参数可以裁剪并缩放出来15×15的图片。然后通过`show_face_key_points.py`可以看点的位置是否正确。确保正确的点的位置后，`convert_imagelist_2_hdf5_organ.py`转换数据格式成hdf5进入`train_img/preprocessed_img/LE21/caffe`,运行`./train.sh`进行训练,训练日志保存在LOG文件夹下.可以使用caffe自带工具看loss下降曲线,为了方便我拷贝到了tools文件夹下.首先把LOG文件夹下的日志重命名为`*.log`的格式(去掉后缀就可以)  
   
`./parse_log.py LOG/*.log` `./plot_training_log.py.example 0 result.png *.log`就可以看效果了
   
## 综合测试：
   
`test_img/calFeaturePts.py`可以进行测试
