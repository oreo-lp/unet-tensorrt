# 1. UNet (tensorrt)
[UNet](http://www.arxiv.org/pdf/1505.04597.pdf)网络在语义分割领域中有着非常广泛的应用，并在医疗图像处理以及自动驾驶任务上都发挥中非常重要的作用。在实际工程应用中，需要对UNet算法进行推理加速以期满足算法实时推理的要求。因此，本文使用TensorRT加速技术对UNet算法进行推理加速。本项目已经实现了FP16和INT8的推理加速。<br>

# 2. Requirements

TensorRT 7.2<br>
Cuda 11.1<br>
Python 3.7<br>
opencv 3.4<br>
cmake 3.18<br>
# 3. Convert .pth file to .wts file
使用gen_wts.py脚本程序将pytorch的.pth权重文件转成.wts文件。这样做的目的是为了c++在加载权重文件的时候比较方便和简单。<br>
其中，.wts文件第一行保存着该网络有多少个参数，此后每一行都以[参数名称] [参数个数] [参数值]的顺序进行排列。比如：<br>
```
128
inc.double_conv.0.weight 1728 01b64746 ...
inc.double_conv.0.bias 6 64 01a09524 ...
...
```
其中128表明UNet网络共有128个参数(weights, bias, running_mean, running_var)<br>
inc.double_conv.0.weight表明网络层的名称，1728表示该网络层的weight参数共有1728个值，01b64746等表示该网络层的具体参数值。后面的数据排列依次类推。<br>

提醒：需要修改gen_wts.py文件中间.pth权重文件的目录地址。

# 4. generate engine file and infer
创建build文件夹，然后再build文件夹下进行遍历，遍历之后会在本文件夹下生成unet可执行文件。
```
mkdir build
cd build
cmake ..
make
```

## 4.3 generate TensorRT engine file and infer image
使用TensorRT进行加速推理主要有三步：首先，需要使用TensorRT的API构建UNet网络；其次，将构建好的engine模型进行序列化并保存在硬盘上，方便下次直接加载；最后，反序列化engine文件进行推理加速。</br>
(1) 序列化engine模型
```
./unet -s
```
经过上面一段代码处理之后，会在UNet文件下生成一个engine文件。如果需要使用FP16进行计算，可以打开#define USE_FP16的开关。使用FP16数据的关键代码就是给config设置一个flag，从而在生成engine的时候支持FP16的数据。<br>
```
config->setFlag(BuilderFlag::kFP16);
```
(2) 反序列化engine模型并进行推理
在这一部分主要就是初始化engine模型，然后分配显存再进行推理加速。
```
unet -d 
```

# 5. INT8 Quantification
首先，在使用INT8进行加速推理的时候，需要一个校准数据集，本文使用的coco_calib校准数据集，将其放在build文件夹下即可。其中coco_calib校准数据集下载地址如下：[GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh<br>
然后，打开#define USE_INT8的开关，再次进行编译即可，程序运行之后会得到一个量化表格，下次再运行的时候就不需要再次生成一个量化表格，会加载已存在的量化表。coco_calib校准数据集中有1000张图像，所以时间会有点久，大约四五个小时，甚至更久。
在INT8量化的过程中，主要有三个步骤：<br>
(1) 判断平台是否支持INT8量化。
```
std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "Yes!" : "No!") << std::endl;
assert(builder->platformHasFastInt8());
config->setFlag(BuilderFlag::kINT8);
```
(2) 创建一个图像读取类(需要继承Int8EntropyCalibrator2类，并重写相关方法)。<br>
详细代码请看calibrator.cpp文件代码。<br>
(3) 给config设置INT8的flag并添加图像读取类对象。
```
Int8EntropyCalibrator2* calib = new Int8EntropyCalibrator2(1, INPUT_WIDTH, INPUT_HEIGHT, "./coco_calib/", "../unet_int8calib.table", INPUT_BLOB_NAME);
config->setInt8Calibrator(calib);
```
其中，Int8EntropyCalibrator2类需要知道矫正数据集地址(coco_calib)、网络输入层的名称(INPUT_BLOB_NAME)以及校正表名称(unet_int8calib.table)。

# 6. Efficiency
下面列出了加速前后的时间损耗 (测试环境为：Tesla V100)

 PyTorch | FP32 | FP16 | INT8
 ---- | ----- | ------  | ------
 512x512  | 512x512 | 512x512 | 512x512
  31ms | 16ms  | 8ms  | 9ms

从上表中可以返现经过TensorRT加速之后，模型的推理速度有了显著的提高，但是有一个很奇怪的点就是INT8的推理时间竟然比FP16的加速时间还要长，并且经过试验验证INT8的效果比FP16的还要差，前一个问题暂时还没有解决，后一个问题主要是因为UNet用于像素级别的分类，因此INT8所带来的数据损耗比FP16要大很多，因而造成了INT8的推理结果要差很多。