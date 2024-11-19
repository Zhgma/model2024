import os,time
import mmcv
import cv2
import numpy as np
from PIL import Image
from skimage import morphology
from skimage import img_as_float
from skimage import img_as_ubyte
from mmdet.apis import init_detector, inference_detector
import time
np.set_printoptions(threshold=1000000)
def Get_pathlist(imgPath):
    filelist = os.listdir(imgPath)
    imglist=[]
    namelist=[]
    for i in filelist:
        path = os.path.join(os.path.abspath(imgPath), i)
        imglist.append(path)
        namelist.append(i)
    return imglist,namelist

def resize600(cutimg):
    if cutimg.shape[0]/cutimg.shape[1]<0.99 or cutimg.shape[0]/cutimg.shape[1]>1.01:
        longside=max(cutimg.shape[0],cutimg.shape[1])
        gap=int(0.5*(max(cutimg.shape[0],cutimg.shape[1])-min(cutimg.shape[0],cutimg.shape[1])))
        if len(cutimg.shape)>2:
            imgtmp11=np.ones([max(cutimg.shape[0],cutimg.shape[1]),max(cutimg.shape[0],cutimg.shape[1]),cutimg.shape[2]])*255
            if cutimg.shape[0]>cutimg.shape[1]:
                imgtmp11[:,gap:gap+cutimg.shape[1],:]=cutimg
            else:
                imgtmp11[gap:gap+cutimg.shape[0],:,:]=cutimg
        else:
            imgtmp11=np.ones([max(cutimg.shape[0],cutimg.shape[1]),max(cutimg.shape[0],cutimg.shape[1])])*255
            if cutimg.shape[0]>cutimg.shape[1]:
                imgtmp11[:,gap:gap+cutimg.shape[1]]=cutimg
            else:
                imgtmp11[gap:gap+cutimg.shape[0],:]=cutimg
        resimg=cv2.resize(imgtmp11,(600,600))
    else:
        resimg=cv2.resize(cutimg,(600,600))
    return resimg


config_file = 'denoise_faster_rcnn_x101_32x4d_fpn_2x_coco.py'
checkpoint_file = 'denoise_epoch.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test a list of images and write the results to image files
imgPath ="test/denoise/"
dst_Path="test/20240329/"
if not os.path.exists(dst_Path):
    os.makedirs(dst_Path)
imgs,names=Get_pathlist(imgPath) 
#print(names)
model.CLASSES=['c']
i=0
for img in imgs:
    name=names[i]
    i=i+1
    print()
    print(name)
    start=time.time()
    if name.find('TIF')<0:
        continue
    newname=name.replace("A.TIF", "A0.png")
    newname1=name.replace("A.TIF", "A1.png")
    #print(name)
    result=inference_detector(model, img)[0]
    print('time pass:', time.time()-start, 's')
    I=cv2.imread(img)
    #print(I.shape)
    if len(result)>1:
        area=[]
        for j in range(len(result)):
            bbox=result[j]
            area.append((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        result=result[area.index(max(area))]
    #    print(result)
    #else:
    #    print(result)
    #print(result.shape)
    if result.shape[0]==0:
        continue
    if len(result.shape)>1:
        result=result[0]
    region=min(int(result[2])-int(result[0]),int(result[3])-int(result[1]))
    #print(region)
    region1=int(region*0.06)
    while region1>0:
        if int(result[1])-region1>0 and int(result[3])+region1<I.shape[0] and int(result[0])-region1>0 and int(result[2])+region1<I.shape[1]:
            break
        region1=region1-1
    I_crop=I[int(result[1])-region1:int(result[3])+region1,int(result[0])-region1:int(result[2])+region1,:]###
    img0=resize600(I_crop)
    cv2.imwrite(os.path.join(dst_Path, newname),img0)
    img1=img0
    I_gray = cv2.cvtColor(I_crop,cv2.COLOR_RGB2GRAY)
    gradsize=int(0.1*min(I_gray.shape))
    ret, binary = cv2.threshold(I_gray, 0, 255, cv2.THRESH_OTSU*2)
    binary = 255-binary
    binary_ = img_as_float(binary)
    size=int(0.0005*binary_.shape[0]*binary_.shape[1])
    binary_ = morphology.remove_small_objects(binary,size)
    binary = img_as_ubyte(binary_)
    binary = binary/255
    for k in range(I_gray.shape[0]):
        for j in range(I_gray.shape[1]):
            if binary[k][j]==0:
                I_gray[k][j]=255
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(gradsize,gradsize))
    imgtmp1 = clahe.apply(I_gray)
    imgtmp1 = np.around(0.5*imgtmp1+0.5*I_gray)
    imgtmp1=resize600(imgtmp1)#
    img1[:,:,0]=imgtmp1
    img1[:,:,1]=imgtmp1
    img1[:,:,2]=imgtmp1
    cv2.imwrite(os.path.join(dst_Path, newname1),img1)
