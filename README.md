# Semantic-segmentation-for-aerial

## 数据集
使用的数据集是`kaggle`的[Semantic segmentation of aerial imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)

## utils
### dataConvert.py
dataConvert中主要包含数据的变换过程
|函数|作用|
|--|--|
**loadColorMap**|用于加载标签的颜色映射
**voc_colormap2label**|获取颜色标签到数值标签的映射关系
**voc_rand_crop**|用于裁剪数据
**voc_label_indices**|将RGB标签转换成数值标签
**one hot**|将标签转换成one hot

### dataLoader.py
dataLoader.py包含数据的加载过程
|类/函数|作用|
|--|--|
**SemanticDataset**|数据加载类，包含数据归一化，数据裁剪过程，用于加载数据
**load_data_voc**|调用SemanticDataset批量加载训练集和测试集
### losses.py
定义损失函数，本项目中采用的是Focal loss和Dice loss相加作为损失函数

### model.py
包含U-net模型和deeplabv3+两种模型，在训练和测试过程可以通过修改参数进行选择

## prepare模块
这个模块在训练前执行，是整个项目的准备工作
### 函数
|函数|作用|
|--|--|
**semantic2dataset**|航拍数据集转换成语义分割的数据集
**trainValSplit**|分割训练集和测试集
**getMeanStd**|获取均值和方差
**writeColorClasses**|保存颜色和类别
### 参数
参数只有两个，就是颜色映射和类别，在本项目中这两个参数为

```python
VOC_COLORMAP = [[226, 169, 41], [132, 41, 246], [110, 193, 228], [60, 16, 152], [254, 221, 58], [155, 155, 155]]
VOC_CLASSES = ['Water', 'Land (unpaved area)', 'Road', 'Building', 'Vegetation', 'Unlabeled']
```

## train模块

### 函数
**train**
根据传入的参数进行训练
### 参数
|参数|作用|
|--|--|
|**batch_size**|批量大小，在语义分割中可以设置小一些
|**crop_size**  |裁剪图像大小
|**model_choice**|模型的选择，可选U-net、deeplabv3+
|**in_channels**|输入图像通道数，RGB图像为3，灰度图为1
|**out_channels**|输出标签类别，本项目中为6
|**num_epochs** |训练总轮次
|**auto_save** |自动保存权值的间隔轮次
|**lr**|学习率
 |**device** |训练使用的环境，当cuda可用时自动设为cuda，否则自动设为cpu
## predict模块
predict模块只是浅测一下模型的精度和效果，如果需要应用可以调用predict函数进行预测并与实际应用结合
### 函数
|函数|作用|
|--|--|
|**label2image**|数值标签转换成RGB标签
**predict**|单张图片的预测
**read_voc_images**|读取图片
**plotPredictAns**|画出测试结果
### 参数
|参数|作用|
|--|--|
**voc_dir**|测试数据的路径
**means**|图像均值
**stds**|图像方差
 |**device** |训练使用的环境，当cuda可用时自动设为cuda，否则自动设为cpu
**batch_size**|批量大小
|**model_choice**|模型的选择，可选U-net、deeplabv3+

# 权值文件与数据
百度网盘
链接：https://pan.baidu.com/s/1ESyD9IOGYli5eFuH5t9BhA?pwd=gh97 
提取码：gh97
