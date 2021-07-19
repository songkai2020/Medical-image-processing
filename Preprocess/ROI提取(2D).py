import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import cv2
from PIL import Image
import imageio
import os
import csv
import pydicom
from pydicom import dcmread

#注：代码中可能有部分内容需根据不同的数据集进行相应的修改，分别为104,111,138,145,150行
def Save_img(input_path, save_path):
    # 将mask数据3Dnii数据切片，生成2Dslice，函数输入为nii文件的文件夹路径
    files = os.listdir(input_path)
    for img in files:

        # 获取文件名
        file_folder = os.path.basename(img).split('.')[0]
        if os.path.exists(save_path + '/' + file_folder):
            continue
        else:
            os.mkdir(save_path + '/' + file_folder)
        img_load = nib.load(input_path + '/' + img)
        img_fdata = img_load.get_fdata()
        total = img_fdata.shape[-1]

        for index in range(total):
            img_fdata_ = img_fdata[:, :, index]
            # 交换两个维度像素点
            img_index = np.array(np.swapaxes(img_fdata_, 0, 1))
            # 二值化
            ret, binary = cv2.threshold(img_index, 0, 255, cv2.THRESH_BINARY)
            # 保存图像
            imageio.imwrite(save_path + '/' + file_folder + '/' + file_folder + '_' + str(index) + '_' + '.png', binary)


def Rect_img(path, csv_writer):
    # 粗略框出ROI区域的坐标信息，x,y,w,h，输入为切割后nii图像文件夹路径
    # 全黑图
    black = np.zeros((512, 512))
    file_folder = os.listdir(path)
    for folder in file_folder:
        imgs = os.listdir(path + '/' + folder)
        for file in imgs:
            # 读取图片
            binary = cv2.imread(path + '/' + folder + '/' + file, -1)
            # img = cv2.imread(path+'/'+folder+'/'+file)
            if file == '.ipynb_checkpoints':  # google colab 跳过该文件
                continue
            else:
                # 全黑图跳过
                if binary.any() == black.any():
                    pass
                else:
                    # 框出ROI区域
                    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    x, y, w, h = cv2.boundingRect(contours[0])
                    location = [x, y, w, h]
                    ID = file.split('h')[0]
                    num = file.split('_')[2]
                    # 将生成的ROI区域最小邻接矩阵坐标数据写入csv文件
                    csv_writer.writerow([ID, num, x, y, w, h])

def data_csv(imgpath, savepath):
    # 将坐标信息写入csv文件,输入各个数据子文件夹的父文件夹目录即可
    f = open(savepath + '/' + 'datainfo.csv', 'w', encoding='utf-8',newline='')
    # 基于文件对象构建csv写入对象
    csv_writer = csv.writer(f)
    # 表头
    csv_writer.writerow(['ID', 'num', 'x', 'y', 'w', 'h'])
    Rect_img(imgpath, csv_writer)
    # 文件关闭
    f.close()


def Read_csv(path):
    # 读取坐标信息文件
    file = open(path, 'r')
    rows = csv.reader(file)
    coordinate = []
    for row in rows:
        # 转换成list形式输出
        coordinate.append(row)
    return coordinate


def Read_CT(csv_path, img_path, save_path):
    # 读入坐标数据
    location = Read_csv(csv_path)
    # 获得带名字文件夹列表
    file_folder = os.listdir(img_path)
    for file in file_folder:
        # 遍历带名字文件夹的子文件夹，获得编号+hrT2a的文件夹
        sub_folder = os.listdir(img_path + '/' + file)
        for sub_file in sub_folder:
            if os.path.exists(save_path + '/' + sub_file):
                pass
            else:
                os.mkdir(save_path + '/' + sub_file)
            # 获取文件夹id，注：由于本项目采用数据集的文件夹命名方式为'256hrT2a'格式，这里需要根据不用数据集的命名方式采用不同的分割符
            id_ = sub_file.split('h')[0]
            # 从1开始跳过表头
            for i in range(1, len(location)):
                # 查看数据表中有无相同ID
                Id = location[i][0]
                if Id == id_:
                    # #读取对应IM文件 注：根据不同的数据模态，采用不同的读取方式
                    print(img_path + '/' + file + '/' + sub_file + '/' + 'IM' + str(location[i][1]))
                    dataset = dcmread(img_path + '/' + file + '/' + sub_file + '/' + 'IM' + str(location[i][1]))
                    x = int(location[i][2])
                    w = int(location[i][4])
                    y = int(location[i][3])
                    h = int(location[i][5])
                    # 对比w，h哪个大，大的内个长度作为边长
                    cx = (x + x + w) // 2
                    cy = (y + y + h) // 2
                    print(cx, cy)
                    x = cx - 71
                    y = cy - 71
                    new_img = dataset.pixel_array[y:(y + 142), x:(x + 142)]
                    new_img = cv2.normalize(new_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    imageio.imwrite(save_path + '/' + sub_file + '/' + sub_file + location[i][1] + '.png', new_img)
                    

#此步非必须，对dcm文件进行排序
def sort_dicom(img_path, save_path):
    # 将dicom图像排序，根据sliclocation
    file_folder = os.listdir(img_path)
    # 获取带名字的文件夹
    for file in file_folder:
        # 获取带名字文件夹中的子文件夹
        sub_folder = os.listdir(img_path + '/' + file)
        for sub_file in sub_folder:
            # 跳过不是hrT2a文件夹，注：可能一个病人会包含多个包含数据的文件夹，这里我们直接跳过包含无用数据的文件夹
            if os.path.exists(save_path + '/' + file + '/' + sub_file) or sub_file == '.ipynb_checkpoints' or \
                    sub_file.split('h')[1] != 'rT2a':
                pass
            else:
                os.mkdir(save_path + '/' + file)
                os.mkdir(save_path + '/' + file + '/' + sub_file)
            # 注：这里根据不同的文件夹命名方式进行相应的修改
            if os.path.basename(sub_file).split('h')[1] == 'rT2a':
                dic = os.listdir(img_path + '/' + file + '/' + sub_file)
                # print(img_path+'/'+file+'/'+sub_file)
                dict = {}
                # 获取各图像对应的slice locaion  注意.ipynb_checkpoints也算，如果不是第一次执行记得减去
                for i in range(len(dic)):
                    ds = pydicom.read_file(img_path + '/' + file + '/' + sub_file + '/' + 'IM' + str(i))
                    dict['IM' + str(i)] = float(ds.SliceLocation)
                    # 使用匿名函数依据value进行排序
                dict_sort = sorted(dict.items(), key=lambda x: x[1])
                # print(dict_sort)
                count = 0
                for item in dict_sort:
                    # 重命名将排序后的文件另存
                    os.rename(img_path + '/' + file + '/' + sub_file + '/' + item[0],
                              save_path + '/' + file + '/' + sub_file + '/IM' + str(count))
                    count += 1

def rect_ROI(nii_path,nii_slice_save,csv_save,dicom_path,sort_path,rec_save_path):
    print('切片nii文件中.......')
    Save_img(nii_path,nii_slice_save)
    print('切片结束，生成病灶信息csv文件中.....')
    data_csv(nii_slice_save,csv_save)
    print('生成csv，进行对dicom文件进行序列校正中.......')
    sort_dicom(dicom_path,sort_path)
    print('序列校正完毕，切割图像中......')
    csv_path=csv_save+'/datainfo.csv'
    Read_CT(csv_path,sort_path,rec_save_path)
    print('切割完毕！其中csv文件位于：'+csv_save)

#按顺序执行所有步骤
if __name__ == "__main__":
    nii_path = r'G:\recta_cancer_T2\ROI_T2'  # nii 文件存放路径
    nii_slice_save = r'G:\recta_cancer_T2\nii_slice'  # nii切片后存放路径
    csv_save = r'G:\recta_cancer_T2'  # csv信息文件存放路径
    dicom_path = r'G:\recta_cancer_T2\T2_case1-200'  # dicom文件存放路径
    sort_path = r'G:\recta_cancer_T2\T2_case1-200_sorted'  # 排序后dicom文件存放路径
    rec_save_path = r'G:\recta_cancer_T2\rec_save'  # 切割文件存放的路径
    rect_ROI(nii_path, nii_slice_save, csv_save, dicom_path, sort_path, rec_save_path)

#步执行
# # 第一步切割nii为png，输入为两个路径 存放nii文件文件夹路径 存放切片后nii文件的路径
# nii_path='/content/sample_data/RECT/ROI_T2'
# nii_slice_save='/content/sample_data/nii'
# Save_img(nii_path,nii_slice_save)
#
# # 框出病灶区域，返回 ID 病人编号 num 有病灶的切片编号 x,y 病灶区域左上角坐标 w,h 宽度高度，输入为切片后nii文件的路径和csv的存放路径
# nii_slice_save='/content/sample_data/RECT/nii_slice'
# csv_save='/content/sample_data/RECT'
# data_csv(nii_slice_save,csv_save)
#
# #DICOM排序
# dicom_path='/content/sample_data/RECT/T2_case_2'
# sort_path='/content/sample_data/RECT/sort_im'
# sort_dicom(dicom_path,sort_path)

# 切割病灶
# csv_path='/content/sample_data/RECT/datainfo.csv'
# sort_path='/content/sample_data/RECT/sort_im'
# save_path='/content/sample_data/RECT/rect'
# Read_CT(csv_path,sort_path,save_path)
# 切割病灶nii区域
