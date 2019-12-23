# Capsule-Network
This project will explore the capsule network, take MNIST as an example, the code is based on keras.


神经网络很好地胜任了分类任务、回归任务和目标检测等任务。但神经网络是一个典型的“黑盒子”，盒子里面的参数可能是有规律的，也有可能是有缺陷的。比如对于一张人脸图片，如果更改五官的相对位置，人可以察觉这种变化，而神经网络不行，后者依然会将图片识别为人脸。<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E8%83%B6%E5%9B%8A%E7%BD%91%E7%BB%9C%E4%BA%BA%E8%84%B8.png" alt="Sample"  width="300">
</p>

神经网络分析图片依赖于特征提取，它更倾向于判断特征是否存在，而不关心特征之间是否关联，所以神经网络对于特征之间的相互关系不敏感。

为了让神经网络克服上述问题，[胶囊网络](https://arxiv.org/pdf/1710.09829.pdf)（capsule networks）被提出。胶囊网络解决这个问题的方法是，实现对空间信息进行编码同时也计算物体的存在概率。这可以用向量来表示，向量的模表示特征存在的概率，向量的方向表示特征的姿态信息。

胶囊网络的工作原理归纳成一句话就是，所有特征的状态信息，都将以向量的形式被胶囊封装。比起“胶囊网络”这个称呼，向量神经元 (vector neuron) 或者张量神经元 (tensor neuron) 形容起来似乎更贴切。


## 网络结构<br>
下图是胶囊神经网络的结构示意图：<br>
<p align="center">
	<img src="https://image.jiqizhixin.com/uploads/editor/bcdc9a37-9371-4a2e-a105-a80a1e76f1c9/640.png" alt="Sample"  width="600">
</p>

可以看到，输入是一张手写数字图片。

第一步，对图片做常规卷积，用了 256 个 stride 为 1 的 9x9 卷积核，得到 20x20x256 的 ReLU Conv1，这一步主要是对图像做一次局部特征检测；

第二步，对 ReLU Conv1 继续做卷积，用了 32 个 stride 为 2 的 9x9x256 的卷积核做了 8 次卷积，得到的输出为向量神经元层 Primary Capsule，图中的排列方式是 8 个 6x6 的一组，共32组；

第三步，将 Primary Capsule 转换为 Digit Capsule，这两层的连接依靠迭代动态路由 (iterative dynamic routing) 算法确定。这两层是全连接的，但不是像传统神经网络标量和标量相连，而是向量与向量相连。PrimaryCaps 里面有 6x6x32 元素，每个元素是一个 1x8 的向量，而 DigitCaps 有 10 个元素 (因为有 10 个数字类别)，每个元素是一个 1x16 的向量。为了让 1x8 向量与 1x16 向量全连接，需要 6x6x32 个 8x16 的矩阵。现在 PrimaryCaps 有 6x6x32 = 1152 个向量，而 DigitCaps 有 10 个向量，那么对于权重 Wij，i= 1,2, …, 1152, j = 0,1, …, 9。

第四步，对Digit Capsule 中的10 个向量求模，模值最大的向量代表的就是图片概率最大的那个分类。胶囊网络用向量模的大小衡量某个实体出现的概率，模值越大，概率越大。需要注意的是，Capsule 输出的概率总和并不等于 1，也就是 Capsule 有同时识别多个物体的能力。与传统 CNN 的全连接分类层相比，胶囊网络的 DigitCaps 层显然包含更多信息。


## 重构表示<br>
除了分类，胶囊网络还能由DigitCaps 层重建图片信息，依赖以下的解码器结构：<br>
<p align="center">
	<img src="http://5b0988e595225.cdn.sohucs.com/images/20180328/5c0bb065da184881ac44fe456dbb3042.jpeg" alt="Sample"  width="500">
</p>

可以看到，解码器主要包含若干全连接层。重构的时候单独取出需要重构的向量(上图橘色) ，使用全连接网络重构。以 MNIST 数据集为例，图片形状为 28x28，解码器的输出层为一个长度为 784 的向量，通过 reshape 重构为图片。


## 胶囊结构<br>
输入与输出


## 损失函数<br>
由于 Capsule 允许多个分类同时存在，所以不能直接用传统的交叉熵 (cross-entropy) 损失，作者采用的是是用间隔损失 (margin loss)。<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/Margin%20loss%20for%20digit%20existence.png" alt="Sample"  width="450">
</p>

其中：k 是分类；Tk 是分类的指示函数 (k 类存在为 1，不存在为 0)；m+ 为上界，惩罚假阳性(false positive) ，即预测 k 类存在但真实不存在，识别出来但错了的样本；m- 为下界，惩罚假阴性(false negative) ，即预测 k 类不存在但真实存在，没识别出来的样本；λ 是比例系数，调整两者比重总的损失，是各个样例损失之和。

论文中 m+= 0.9, m-= 0.1, λ = 0.5。即：如果 k 类存在，||vk|| 不会小于 0.9；如果 k 类不存在，||vk|| 不会大于 0.1；惩罚假阳性的重要性大概是惩罚假阴性的重要性的 2 倍。

最终的总体损失包含上述用于分类的的间隔损失，还包含图片重构的重构损失，重构损失由 MSE 计算并乘以系数 α，即：总体损失 = 间隔损失 + α·重构损失。

其中 α = 0.0005，所以间隔损失还是占主导地位。


## 动态路由<br>




## 非线性变换 Squash<br>




## 仿射变换 Affine Transform<br>















