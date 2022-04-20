# GAN网络的实现以及拓展

--------

## 项目介绍

这是自己学完GAN网络后的大（中）作业，写完就顺便保存一下。

功能一共有三个：torch.nn搭建的DCGAN；用torchgan封装好的模型；评价指标。

项目已经上传至： www.baidu.com

## 文件及目录介绍

### GAN-Metrics

> * 评价指标程序 （修改自 https://github.com/xuqiantong/GAN-Metrics 的程序） 
> * 使用方法：`python GAN-Metrics/demo_dcgan.py --dataset mine --cuda --dataroot imgFolder --outf outFolder --sampleSize 2000`
> > * imgFolder 下为一个或者多个文件夹，表示各种类别的图片
> > * --dataroot 为数据集存放目录，该文件夹下有类别文件夹，再之下为数据集
> > * --sampleSize 为评价生成的图片数量。
> * 可在`GAN-Metrics/net.py`中修改两个网络的结构

### Gan_sifar

老师给的样例程序，在sifar数据集上训练GAN模型，自己把其中的网络封装到 `net.py`中了。没啥好说的。

### net.py

装有用torch.nn搭建的GAN网络。魔改网络的话在这里改就行。不过感觉这个网络效果已经很不错了。

### Gan_2img.py

用五万张96*96的二次元图片来训练一个GAN网络。相比`Gan_sifar.py` 就是读取数据集的部分改了下以及参数调了亿点点。其他没啥改动。

数据集下载路径：https://zhuanlan.zhihu.com/p/351083489

数据集存放路径： mycf_Gan_net/gan_2img/1    把数据集都扔1那个文件夹里面（自己新建文件夹吧）

### torchGan.py

基于torchgan库来实现GAN网络。不得不说封装的是真的好，换模型或者调参或者加功能都十分方便，还有tensorboard可以实时看训练过程！但就是目前没跑出过很好的效果。我觉得原因在我QAQ。


