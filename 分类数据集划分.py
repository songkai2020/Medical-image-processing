import os
import numpy as np
import shutil

#二分类数据集划分，分别用0,1表示两种不同的类
#训练、测试、验证数据集所占比例
TRAIN=0.7
VALID=0.2
TEST=0.1

def Divide(totalpath,trainpath,validpath,testpath):
  imgs = os.listdir(totalpath)
  index = [i for i in range(len(imgs))]
  count=0
  gen1=0
  gen2=0
  val_gen1=0
  val_gen2=0
  val_count=0
  count3=0
  for i in index:
    if count<(int(len(imgs)*TEST)):
      #划分数据集考虑到均衡性
      if imgs[i].split('_')[1]=='0': 
        if gen1<(int(len(imgs)*TEST)//2): 
          shutil.copy(totalpath+'/'+imgs[i],testpath)
          gen1+=1
          count+=1
        else:
          if val_count<(int(len(imgs)*VALID)):
            if val_gen1<(int(len(imgs)*VALID)//2):
              shutil.copy(totalpath+'/'+imgs[i],validpath)
              val_gen1+=1
              val_count+=1
            else:
              shutil.copy(totalpath+'/'+imgs[i],trainpath)
              count3+=1
          else:
            shutil.copy(totalpath+'/'+imgs[i],trainpath)
            count3+=1

      elif imgs[i].split('_')[1]=='1':
        if gen2<(int(len(imgs)*TEST)//2):
          shutil.copy(totalpath+'/'+imgs[i],testpath)
          gen2+=1
          count+=1
        else:
          if val_count<(int(len(imgs)*VALID)):
            if val_gen2<(int(len(imgs)*VALID)//2):
              shutil.copy(totalpath+'/'+imgs[i],validpath)
              val_gen2+=1
              val_count+=1
            else:
              shutil.copy(totalpath+'/'+imgs[i],trainpath)
              count3+=1
          else:
            shutil.copy(totalpath+'/'+imgs[i],trainpath)
            count3+=1
    else:
      if val_count<(int(len(imgs)*VALID)):
        if imgs[i].split('_')[1]=='0':
          if val_gen1<(int(len(imgs)*VALID)//2):
            shutil.copy(totalpath+'/'+imgs[i],validpath)
            val_gen1+=1
            val_count+=1
          else:
            shutil.copy(totalpath+'/'+imgs[i],trainpath)
            count3+=1
        elif imgs[i].split('_')[1]=='1':
          if val_gen2<(int(len(imgs)*VALID)//2):
            shutil.copy(totalpath+'/'+imgs[i],validpath)
            val_gen2+=1
            val_count+=1
          else:
            shutil.copy(totalpath+'/'+imgs[i],trainpath)
            count3+=1
      else:
          shutil.copy(totalpath+'/'+imgs[i],trainpath)
          count3+=1
  print(count3,count,val_count)

Divide('/content/sample_data/rec_total','/content/sample_data/traindata/train','/content/sample_data/validdata/valid','/content/sample_data/testdata/test')
