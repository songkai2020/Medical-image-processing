# 医学图像处理相关知识(Keras)
## 本项目主要包含医学图像处理学习中常用的基础代码知识主要包括：
1.数据预处理部分  
&nbsp;&nbsp;&nbsp;&nbsp;a.ROI区域的提取（2D,3D）  
&nbsp;&nbsp;&nbsp;&nbsp;下图为2DROI的提取过程，首先根据mask确定病灶的中心，在原图以病灶为中心选定150\*150(可以根据自己的数据集调整大小)的矩形框进行裁剪。这里虽然我们选定了矩形框的边长为150\*150,但它并不是最优的，因为存在部分病灶区域面积仍大于150\*150,朋友们可以集思广益，有好的想法随时联系。
![image](https://user-images.githubusercontent.com/61354006/125883143-5d1c0922-b897-4047-a668-7b5d35abd0ec.png)  
&nbsp;&nbsp;&nbsp;&nbsp;b.数据集的划分——对分类数据集按照指定比例将其划分为训练集、验证集和测试集（7:2:1）  
2.模型
&nbsp;&nbsp;&nbsp;&nbsp;a.数据增强  
&nbsp;&nbsp;&nbsp;&nbsp;b.分割  
&nbsp;&nbsp;&nbsp;&nbsp;b.分类  
......持续更新中  
