# 医学图像处理相关知识(Keras)
## 本项目主要包含医学图像处理学习中常用的基础代码知识主要包括：
1.数据预处理部分  
&nbsp;&nbsp;&nbsp;&nbsp;a.ROI区域的提取（2D,3D）  
&nbsp;&nbsp;&nbsp;&nbsp;下图为2DROI的提取过程，首先根据mask确定病灶的中心，在原图以病灶为中心选定150\*150(可以根据自己的数据集调整大小)的矩形框进行裁剪。这里虽然我们选定了矩形框的边长为150\*150,但它并不是最优的，因为存在部分病灶区域面积仍大于150\*150,朋友们可以集思广益，有好的想法随时联系。
![image](https://user-images.githubusercontent.com/61354006/125883143-5d1c0922-b897-4047-a668-7b5d35abd0ec.png)  
&nbsp;&nbsp;&nbsp;&nbsp;b.数据集的划分——对分类数据集按照指定比例将其划分为训练集、验证集和测试集（7:2:1）  
2.模型(该部分仅为最基础的网络，可参考性不大)  
&nbsp;&nbsp;&nbsp;&nbsp;a.数据生成  
&nbsp;&nbsp;&nbsp;&nbsp;b.分割  
&nbsp;&nbsp;&nbsp;&nbsp;c.分类  
3.论文作图  
&nbsp;&nbsp;&nbsp;&nbsp;a.分割效果对比图：将模型分割结果与GroundTruth作到同一张图上，更好的展示模型分割性能(红色为GT，绿色为pred)  
&nbsp;&nbsp;&nbsp;&nbsp;![mask1](https://user-images.githubusercontent.com/61354006/126169287-add31d45-7d2e-4dad-ba68-cbe0df83191a.png)  
&nbsp;&nbsp;&nbsp;&nbsp;使用作图代码需要建立的文件夹目录如下：  
&nbsp;&nbsp;&nbsp;&nbsp;![root](https://user-images.githubusercontent.com/61354006/126169596-1019826a-f3c0-45c4-8a27-c05766ab6ef0.png)  
&nbsp;&nbsp;&nbsp;&nbsp;b.可视化类激活图  
&nbsp;&nbsp;&nbsp;&nbsp;![1](https://user-images.githubusercontent.com/61354006/126245779-aeeeca24-18bd-4732-9c39-e8527c1cd4f5.png)

......持续更新中  
