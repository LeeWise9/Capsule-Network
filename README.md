# Capsule-Network
This project will explore the capsule network, take MNIST as an example, the code is based on keras


神经网络很好地胜任了分类任务、回归任务和目标检测等任务。但神经网络是一个典型的“黑盒子”，盒子里面的参数可能是有规律的，也有可能是有缺陷的。比如对于一张人脸图片，如果更改五官的相对位置，人可以察觉这种变化，而神经网络不行，后者依然会将图片识别为人脸。<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E8%83%B6%E5%9B%8A%E7%BD%91%E7%BB%9C%E4%BA%BA%E8%84%B8.png" alt="Sample"  width="300">
</p>

神经网络分析图片依赖于特征提取，它更倾向于判断特征是否存在，而不关心特征之间是否关联，所以神经网络对于特征之间的相互关系不敏感。

为了让神经网络克服上述问题，[胶囊网络](https://arxiv.org/pdf/1710.09829.pdf)（capsule networks）被提出。胶囊网络解决这个问题的方法是，实现对空间信息进行编码同时也计算物体的存在概率。这可以用向量来表示，向量的模表示特征存在的概率，向量的方向表示特征的姿态信息。其工作原理归纳成一句话就是，所有特征的状态信息，都将以向量的形式被胶囊封装。

下图是胶囊神经网络的结构示意图：<br>
<p align="center">
	<img src="https://image.jiqizhixin.com/uploads/editor/bcdc9a37-9371-4a2e-a105-a80a1e76f1c9/640.png" alt="Sample"  width="600">
</p>

可以看到，输入是一张手写数字图片。首先对这张图片做 9x9 常规卷积，得到 ReLU Conv1；然后再对 ReLU Conv1 做 9x9 卷积，并将输出调整成向量神经元层 PrimaryCaps（8 个一组，共32组）。再将 PrimaryCaps 转换为 DigitCaps，DigitCaps 中一共10个向量，每个向量中元素的个数为 16。对这 10 个向量求模，模值最大的向量代表的就是图片概率最大的那个分类。胶囊网络用向量模的大小衡量某个实体出现的概率，模值越大，概率越大。

胶囊网络最重要的想法就是用向量来记录特征信息。在 DigitCaps 层中，分类的类别数为 10，每一个类别使用一个长度为 16 的向量来表示，最后通过计算向量的模值来确定类别。而常规 CNN 在分类层通常采用长度为分类类别数的全连接层计算类别。相比之下，胶囊网络的 DigitCaps 层包含更高维的信息。

除了分类，胶囊网络还能由DigitCaps 层重建图片信息，依赖以下的解码器结构：<br>
<p align="center">
	<img src="http://5b0988e595225.cdn.sohucs.com/images/20180328/5c0bb065da184881ac44fe456dbb3042.jpeg" alt="Sample"  width="500">
</p>

可以看到，解码器主要包含若干全连接层。以 MNIST 数据集为例，每张图片形状为 28x28，解码器的输出层为一个长度为 784 的向量，通过 reshape 重构为图片。






