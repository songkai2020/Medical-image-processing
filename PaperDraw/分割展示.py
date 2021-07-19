#  -*- coding: utf-8 -*- 
import cv2
import os
import numpy as np

def Edge_Extract(root,img_masks,img_edge):
    #mask边缘提取函数
    img_root = os.path.join(root,img_masks)			# 修改为保存图像的文件名
    edge_root = os.path.join(root,img_edge)			# 结果输出文件
    if not os.path.exists(edge_root):
        os.mkdir(edge_root)
    file_names = os.listdir(img_root)
    img_name = []
    for name in file_names:
        if not name.endswith('.png'):
            assert "This file %s is not PNG"%(name)
        img_name.append(os.path.join(img_root,name[:-4]+'.png'))
    index = 0
    for image in img_name:
        img = cv2.imread(image,0)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(edge_root+'/'+file_names[index],cv2.Canny(img,30,100))
        index += 1
    return 0

def mask_plot(root,originalfolder,savefolder,img_masks,img_edge,color):
    masks=os.listdir(root+'/'+img_masks)
    for mask in masks:
        print(mask)
        img_org = cv2.imread(root+'/'+originalfolder+'/'+mask, 1)
        print(root+'/'+originalfolder+'/'+mask)
        img_org = cv2.resize(img_org, (128, 128), interpolation=cv2.INTER_LINEAR)
        img = cv2.imread(root+'/'+img_edge+'/'+mask, -1)
        for i in range(128):
            for j in range(128):
                if img[i][j] == 255:
                    img_org[i][j] = color
        if not os.path.exists(root+'/'+savefolder):
            os.mkdir(root+'/'+savefolder)
        cv2.imwrite(root+'/'+savefolder+'/'+mask, img_org)

#
if __name__ == '__main__':
    #必须调整的参数
    root = r'C:\Users\Kevin\Desktop\mask'  #上级文件夹路径
    color_gt = (0, 0,255)  #GroundTruth边框颜色
    color_pred=(255,0,0)   #Prediction边框颜色
   
    #可选参数，一般无需调整
    originalfolder_gt=r'original'
    savefolder_gt=originalfolder_pred=r'masks_plot'
    savefolder_pred=r'Final'
    img_masks_gt=r'img_masks_gt'
    img_edge_gt=r'img_edge_gt'
    img_masks_pred = r'img_masks_pred'
    img_edge_pred = r'img_edge_pred'
    #plot 分割mask
    Edge_Extract(root,img_masks_gt,img_edge_gt)
    mask_plot(root,originalfolder_gt,savefolder_gt,img_masks_gt,img_edge_gt,color_gt)
    #plot 预测mask
    Edge_Extract(root,img_masks_pred,img_edge_pred)
    mask_plot(root,originalfolder_pred,savefolder_pred,img_masks_pred,img_edge_pred,color_pred)
