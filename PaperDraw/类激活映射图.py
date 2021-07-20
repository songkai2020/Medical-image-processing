import cv2
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import Input,UpSampling2D,Conv2D,Activation,MaxPooling2D,Conv2DTranspose,concatenate,GlobalAveragePooling2D,Dense,Dropout,Layer,add,Convolution2D,MaxPool2D,AveragePooling2D,Flatten,Add,Reshape
from keras.models import load_model



def CAM(modelpath,define,layername,No,layernum,originalpath,savepath):
    
    #读取图片
    img1 = cv2.imread(originalpath, -1)
    #img1=cv2.resize(img1,(128,128),interpolation=cv2.INTER_LINEAR)
    img1=img1[:,:,np.newaxis]
    img1 = np.array(img1).astype('float32')
    img1 = np.expand_dims(img1, axis=0)
    img1 /= 255.0
    
    #装载模型
    model = load_model(modelpath, custom_objects=define)
    model.summary()
    result = model.predict(img1)
    if No!=False:
        class_idx = np.argmax(result[No])
    else:
        class_idx = np.argmax(result)
    class_output = model.output[1][:, class_idx]
    last_conv_layer = model.get_layer(layername)
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function(model.input, [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img1)
    for i in range(layernum):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
       
    #保存热力图
    img = cv2.imread(originalpath)
    #img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(savepath, superimposed_img)
    return 0

    


if __name__ == '__main__':
    #模型路径
    modelpath='/content/sample_data/model_Two_0.h5'
    #模型中自定义的层与损失函数
    define={'MSPAM': MSPAM, 'MSCAM': MSCAM,'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss}
    #可视化的卷积层名
    layername='conv19'
    #模型是否为多任务，如果是，No为模型多个输出中分类结果的索引(从0开始)，如果不是，则No为Flase
    No=1
    #加权的通道数，一般为所选可视化卷积层的通道数，如果效果不好可以调小试试
    layernum=256
    #原图路径
    originalpath='/content/sample_data/3hrT2a8_1_.png'
    #热力图保存路径
    savepath='/content/sample_data/123.png'
    
    CAM(modelpath,define,layername,No,layernum,originalpath,savepath)
    