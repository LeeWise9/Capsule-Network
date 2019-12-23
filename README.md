# Capsule-Network
This project will explore the capsule network, take MNIST as an example, the code is based on keras


神经网络很好地胜任了分类任务、回归任务和目标检测等任务。但神经网络是一个典型的“黑盒子”，盒子里面的参数可能是有规律的，也有可能是有缺陷的。比如对于一张人脸图片，如果更改五官的相对位置，人可以察觉这种变化，而神经网络不行，后者依然会将图片识别为人脸。神经网络分析图片依赖于特征提取，而特征之间的关联度可能并不足够，使得神经网络对于特征之间的相互关系不敏感。

为了让神经网络克服上述问题，胶囊网络被提出。

下图是胶囊神经网络的结构示意图<br>
<p align="center">
	<img src="https://image.jiqizhixin.com/uploads/editor/bcdc9a37-9371-4a2e-a105-a80a1e76f1c9/640.png" alt="Sample"  width="250">
</p>
