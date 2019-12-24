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
胶囊网络的整体结构如下图所示：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E8%83%B6%E5%9B%8A%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%843.png" alt="Sample"  width="500">
</p>

胶囊网络其实可以被任认为是一种 Encoder-Decoder 结构的网络。右支为 Encoder，编码的同时完成分类任务；左支由右支的倒数第二层分化出来，完成图像重构，同时也参与梯度误差的反向传播，帮助优化 Encoder 的权值。

### Encoder<br>
下图是胶囊神经网络的 Encoder 结构示意图：<br>
<p align="center">
	<img src="https://image.jiqizhixin.com/uploads/editor/bcdc9a37-9371-4a2e-a105-a80a1e76f1c9/640.png" alt="Sample"  width="700">
</p>

可以看到，输入是一张手写数字图片，形状为 28(长)x28(宽)x1(通道数) 。

第一步，对图片做常规卷积，用了 256 个 stride 为 1 的 9x9 卷积核，得到 20(长)x20(宽)x256(通道数) 的 ReLU Conv1，这一步主要是对图像做一次局部特征检测；

第二步，对 ReLU Conv1 继续做卷积，用 8 个9x9x256 的卷积核与输入做卷积，输出为 6(长)x6(宽)x8(胶囊向量维度)x32(通道数)，图中显示为32组。

第三步，将 Primary Capsule 转换为 Digit Capsule。Digit Capsule 是 10 个长度为 16 的向量。每一个向量由 6x6x32=1152 个 8 维向量与 1152 个 8x16 的矩阵相乘得到，可以想见胶囊网络的计算量是非常大的。这两层的转变是胶囊网络的核心。详细计算步骤请阅读下文“胶囊结构”。

第四步，对Digit Capsule 中的10 个向量求模，模值最大的向量代表的就是图片概率最大的那个分类。胶囊网络用向量模的大小衡量某个实体出现的概率，模值越大，概率越大。需要注意的是，Capsule 输出的概率总和并不等于 1，也就是 Capsule 有同时识别多个物体的能力。与传统 CNN 的全连接分类层相比，胶囊网络的 DigitCaps 层显然包含更多信息。


### Decoder<br>
Encoder 完成分类和编码，由DigitCaps 层可以重建图片信息，依赖以下结构：<br>
<p align="center">
	<img src="http://5b0988e595225.cdn.sohucs.com/images/20180328/5c0bb065da184881ac44fe456dbb3042.jpeg" alt="Sample"  width="500">
</p>

可以看到，解码器主要包含若干全连接层。重构的时候单独取出需要重构的向量(上图橘色) ，使用全连接网络重构。以 MNIST 数据集为例，图片形状为 28x28，解码器的输出层为一个长度为 784 的向量，通过 reshape 重构为图片。


## 胶囊结构<br>
所谓“胶囊”就是向量的集合，网络结构由 Primary Capsule 层转换为 Digit Capsule 层的过程可描述为“胶囊变换”。胶囊结构的输入输出、计算方法与普通的神经网络的不同可由下图来表述：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E8%83%B6%E5%9B%8A%E7%BB%93%E6%9E%84.png" alt="Sample"  width="500">
</p>

vector(ui) 表示胶囊，scalar(xi) 表示普通神经元。

vector(ui) 的输入输出都是向量，中间依次要进行 Affine Transform（仿射变换），Weighting & Sum（加权求和）和 Nonlinear Activation（Squash 非线性变换）。


为了直观理解，这里是胶囊层计算结构图：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E8%83%B6%E5%9B%8A%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%842.png" alt="Sample"  width="400">
</p>

注意，上图中除c1、c2、c3为实数之外其余变量均为向量或矩阵。


### 仿射变换 Affine Transform<br>
简言之，仿射变换是对图片的一种线性变换，变换前后保持对象的平直性和平行性。

仿射变换计算式将对象左乘一个变换矩阵。一些常用的转换矩阵如下所示：<br>
<p align="center">
	<img src="http://images2015.cnblogs.com/blog/120296/201602/120296-20160222070732869-1123994329.png" alt="Sample"  width="500">
</p>

胶囊网络中，仿射变换要对向量 ui 左乘一个矩阵 W。

注意：W 的角标 ij 不表示 W 中的元素，而是 W 的索引，即有 ij 个不同的仿射矩阵 W 参与对 i 个向量 u 的仿射变换计算，得到 ij 个输出（中间变量 u-hat）。


### 加权求和 Weighting & Sum<br>
加权求和这一步计算结果 s 为向量，权值参数 c 确定了 u-hat 和输出的关系，其值由动态路由（Dynamic Routing）算法确定，其步骤如下图所示：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E5%8A%A8%E6%80%81%E8%B7%AF%E7%94%B1%E7%AE%97%E6%B3%95.png" alt="Sample"  width="600">
</p>

Dynamic Routing 算法中 u-hat 表示前一步的输出，r 表示循环迭代的次数（文中为 3 次），l 为层代号。

对于包含 i 个胶囊的第 l 层网络和包含 j 个胶囊的第 (l+1) 层网络，初始化权值参数 bij 为 0.<br>
每一次循环将按照以下步骤进行计算：<br>
* 1.对层 l 中的每一个胶囊 i，由 bi（bij 取第 i 行） 的 softmax 值计算得 ci；<br>
* 2.对层 (l+1) 中的每一个胶囊 j，由 cij （ci 共有 j 行）和 u-hat 线性求和得 sj；<br>
* 3.对层 (l+1) 中的每一个胶囊 j，对 sj 做 squash 非线性变换，得 vj；<br>
* 4.对层 l 和 (l+1) 所有胶囊，将 u-hat 和 vj 做点积，由此更新 bij。<br>

注意，向量点积是动态规划算法的精华。两个向量的点积为标量，标量的大小按照向量的夹角有五种情况：(1)夹角为0，有最大正值；(2)夹角为锐角，结果为正；(3)夹角90°，结果为0；(4)夹角为钝角，结果为负；(5)夹角180°，有最大负值。

由向量点积来迭代参数的过程，实际上取决于 vj 与 u-hat 的方向匹配度：夹角越小增益越大。权值的更新过程实际上就是将方向匹配的向量集合到一起的过程。

Dynamic Routing 算法的理论可以追溯到最大期望算法（Expectation-maximization algorithm），其具体原理本项目不做过多阐述，感兴趣的读者可以查阅更多资料。


### 非线性变换 Squash<br>
如何对向量做非线性激活呢，答案是 squash 变换：<br>
<p align="center">
	<img src="https://ss1.baidu.com/6ONXsjip0QIZ8tyhnq/it/u=1956916309,2761577401&fm=173&app=49&f=JPEG?w=640&h=230&s=49A43C7283B07D8A1E59D1C70000F0B1" alt="Sample"  width="200">
</p>

这个变换将输出值归一化到 0~1 之间，并保留了向量原有的方向信息。当 ||sj|| 很大时，输出 vj 接近 1，当 ||sj|| 很小时，输出 vj 接近 0。



## 损失函数<br>
由于 Capsule 允许多个分类同时存在，所以不能直接用传统的交叉熵 (cross-entropy) 损失，作者采用的是是用间隔损失 (margin loss)。<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/Margin%20loss%20for%20digit%20existence.png" alt="Sample"  width="500">
</p>

其中：k 是分类；Tk 是分类的指示函数 (k 类存在为 1，不存在为 0)；m+ 为上界，惩罚假阳性(false positive) ，即预测为存在但真实不存在，识别出来但错了的样本；m- 为下界，惩罚假阴性(false negative) ，即预测不存在但真实存在，没识别出来的样本；λ 是比例系数，调整两者比重总的损失，是各个样例损失之和。

论文中 m+= 0.9, m-= 0.1, λ = 0.5。即：如果 k 类存在，||vk|| 不会小于 0.9；如果 k 类不存在，||vk|| 不会大于 0.1；惩罚假阳性的重要性大概是惩罚假阴性的重要性的 2 倍。

最终的总体损失包含上述用于分类的的间隔损失，还包含图片重构的重构损失，重构损失由 MSE 计算并乘以系数 α，即：总体损失 = 间隔损失 + α·重构损失。

其中 α = 0.0005，所以间隔损失还是占主导地位。







