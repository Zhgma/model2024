#encoding:utf-8
import mmcv
import os
import cv2,math
import torch
import torchvision
from math import *
#from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, init_detector
import numpy as np, pycocotools.mask as maskUtils
import matplotlib.pyplot as plt
from mmdet.core import get_classes
from glob import glob
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from PIL import Image
#from LoadModel import MainModel
import pickle
import pandas as pd
#from resnet import resnet50, resnet152, resnet101
#from lapsolver import solve_dense
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--backbone', dest='backbone',default='resnet50', type=str)
    parser.add_argument('--b', dest='batch_size',default=1, type=int)
    parser.add_argument('--nw', dest='num_workers',default=1, type=int)
    parser.add_argument('--ver', dest='version',default='val', type=str)
    parser.add_argument('--name', dest='name',default='val', type=str)
    parser.add_argument('--size', dest='resize_resolution',default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',default=448, type=int)
    parser.add_argument('--ss', dest='save_suffix',default=None, type=str)
    parser.add_argument('--acc_report', dest='acc_report',action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],nargs=2, metavar=('swap1', 'swap2'),type=int, help='specify a range')
    args = parser.parse_args()
    return args

args = parse_args()
config_path = 'segment_mask_rcnn_r50_fpn_2x_coco.py'
model_path = 'segment_epoch.pth'
name=args.name
print(name)
img_list = glob('test/20240329/*A1.png')
img_list.sort()
cfg = mmcv.Config.fromfile(config_path)
cfg.model.pretrained = None

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LoadConfig_class(object):
    def __init__(self, args):
        self.swap_num = args.swap_num
        self.backbone = args.backbone
        self.use_dcl = True
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False
        self.weighted_sample = False
        self.cls_2 = True
        self.cls_2xmul = False
        self.numcls = 3###

class Chrom:

    def __init__(self, model_path):
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) 
        ])
    def detect(self, image):
        image = self.transform(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        outputs = self.model(image)
        prob = F.softmax(outputs, dim=1)
        if prob.shape[1]>3:
            top3_val, top3_pos=torch.topk(prob, 7)
        else:
            top3_val, top3_pos=torch.topk(prob, prob.shape[1])
        top3_pos = top3_pos.cpu().numpy()
        top3_val = top3_val.cpu().detach().numpy()
        return top3_pos[0],top3_val[0]

    def detectcase(self, images):
        img_tensors = torch.empty(len(images), 3, 224, 224)
        casenum=0
        for image in images:
            image = self.transform(image)
            img_tensors[casenum,:,:,:] = image
            casenum=casenum+1
        img_tensors = img_tensors.to(device)
        outputs, outputspola, outputspair = self.model(img_tensors)
        #print(outputs.shape)
        #print(outputspola.shape)
        #print(outputspair.shape)
        prob = F.softmax(outputs, dim=1)
        if prob.shape[1]>3:
            top3_val, top3_pos=torch.topk(prob, 7)
        else:
            top3_val, top3_pos=torch.topk(prob, prob.shape[1])
        top3_pos = top3_pos.cpu().numpy()
        top3_val = top3_val.cpu().detach().numpy()
        return top3_pos, top3_val, outputspola


class Chrom_regression:
    def __init__(self, model_path):
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) 
        ])
    def detect(self, image):
        image = self.transform(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        outputs,outputs2 = self.model(image)
        outputs = outputs.cpu().detach().numpy()
        outputs2 = outputs2.cpu().detach().numpy()
        return outputs[0],outputs2[0]

def find_start_row(label,single_range):
    lineend=[5,12,18,24]
    time=0
    label=int(label)
    for i in lineend:
        if label<=i:
            if time==0:
                return (label-1)*2*single_range+10
            if time>0:
                return (label-lineend[time-1]-1)*2*single_range+10
        time=time+1

def myenh(inimg):
    imgYUV = cv2.cvtColor(inimg, cv2.COLOR_BGR2HSV)
    channelsYUV = list(cv2.split(imgYUV))
    #print(channelsYUV[2].shape,channelsYUV[1].shape,channelsYUV[0].shape)
    before=channelsYUV[2]
    imggray = cv2.cvtColor(inimg, cv2.COLOR_BGR2GRAY)
    th=250
    _,bi=cv2.threshold(imggray,th,255,cv2.THRESH_BINARY)
    bi_=255-bi
    [width,height]=bi_.shape
    destribution=np.zeros(256)
    destribution_sum=np.zeros(256)
    count=0
    min_before=before.min()
    for p1 in range(width):
        for p2 in range(height):
            if bi_[p1,p2]==255:
                destribution[before[p1,p2]]+=1
                count+=1
    destribution=destribution/count
    p2=0
    for p1 in range(len(destribution)):
        for p2 in range(p1+1):
            destribution_sum[p1]=destribution_sum[p1]+destribution[p2]
    #print(th-min_before)
    if th-min_before<170:
        min_before=80
    min_before=50
    tmp=np.zeros([width,height])
    for p1 in range(width):
        for p2 in range(height):
            if bi_[p1,p2]==255:
                tmp[p1,p2]=int(destribution_sum[before[p1,p2]]*(th-min_before))+min_before
            else:
                tmp[p1,p2]=255
    #print(tmp.shape)
    channelsYUV[2]=tmp.astype(np.uint8)
    channels = cv2.merge(channelsYUV)
    result = cv2.cvtColor(channels, cv2.COLOR_HSV2BGR)
    return result

def re_assignment(allpred, allprob, allpred5, allprob5):
    
    special_pos=[22,23]
    costarray1=np.ones((len(allpred),46))#22+xx
    print(costarray1.shape)
    for i in range(len(allpred)):
        for j in range(len(allpred5[i])):
            positiony=allpred5[i][j]-1
            costij=1-allprob5[i][j]
            if positiony != 23:
                costarray1[i,2*positiony]=costij
                costarray1[i,2*positiony+1]=costij
    costarray2=np.ones((len(allpred),46))#22+xy
    for i in range(len(allpred)):
        for j in range(len(allpred5[i])):
            positiony=allpred5[i][j]-1
            costij=1-allprob5[i][j]
            if positiony not in special_pos:
                costarray2[i,2*positiony]=costij
                costarray2[i,2*positiony+1]=costij
            if positiony==22:
                costarray2[i,2*positiony]=costij
            if positiony==23:
                costarray2[i,2*positiony-1]=costij

    #print(costarray1[:,0:10])
    rids1, cids1 = solve_dense(costarray1)
    allcost1=0
    for i in range(cids1.shape[0]):
        allcost1+=costarray1[rids1[i],cids1[i]]
    rids2, cids2 = solve_dense(costarray2)
    allcost2=0
    for i in range(cids2.shape[0]):
        allcost2+=costarray2[rids2[i],cids2[i]]
    #print(rids2,cids2)
    print(allcost1,allcost2,len(rids2),len(cids2))
    allpredpost=allpred.copy()
    jumpnum=0
    #if min([allcost1,allcost2])>10: #jia shang hou xiao guo bu hao
    #    return allpredpost, allprob
    if allcost1<allcost2:
        for i in range(len(allpred)):
            if i in rids1:
                allpredpost[i]=cids1[i-jumpnum]//2+1
            else:
                jumpnum=jumpnum+1
    else:
        for i in range(cids2.shape[0]):
            if cids2[i]==45:
                cids2[i]=46
        for i in range(len(allpred)):
            if i in rids2:
                allpredpost[i]=cids2[i-jumpnum]//2+1
            else:
                jumpnum=jumpnum+1
    return allpredpost, allprob

def pola_processing(allcrop,allpred):
    #print(allpred)
    global polachange
    for i in range(len(allpred)):
        if allpred[i]==20:
            polalabel,probs=chrompola20.detect(allcrop[i])
            if polalabel[0]==1:
                allcrop[i] = np.flip(allcrop[i])
                #polachange=polachange+1
        if allpred[i]==18:
            polalabel,probs=chrompola18.detect(allcrop[i])
            if polalabel[0]==1:
                allcrop[i] = np.flip(allcrop[i])
                #polachange=polachange+1
        if allpred[i]==17:
            polalabel,probs=chrompola17.detect(allcrop[i])
            if polalabel[0]==1:
                allcrop[i] = np.flip(allcrop[i])
                #polachange=polachange+1
        if allpred[i]==16:
            polalabel,probs=chrompola16.detect(allcrop[i])
            if polalabel[0]==1:
                allcrop[i] = np.flip(allcrop[i])
                #polachange=polachange+1
        if allpred[i]==15:
            polalabel,probs=chrompola15.detect(allcrop[i])
            if polalabel[0]==1:
                allcrop[i] = np.flip(allcrop[i])
                polachange=polachange+1
        if allpred[i]==14:
            polalabel,probs=chrompola14.detect(allcrop[i])
            if polalabel[0]==1:
                allcrop[i] = np.flip(allcrop[i])
                polachange=polachange+1
        if allpred[i]==1:
            polalabel,probs=chrompola1.detect(allcrop[i])
            if polalabel[0]==1:
                allcrop[i] = np.flip(allcrop[i])
                polachange=polachange+1
        if allpred[i]==3:
            polalabel,probs=chrompola3.detect(allcrop[i])
            if polalabel[0]==1:
                allcrop[i] = np.flip(allcrop[i])
                polachange=polachange+1
        if allpred[i]==4:
            polalabel,probs=chrompola4.detect(allcrop[i])
            if polalabel[0]==1:
                allcrop[i] = np.flip(allcrop[i])
                polachange=polachange+1
    return allcrop

def find_212224angle(img):
    [cx,cy]=[round(img.shape[0]/2),round(img.shape[1]/2)]
    diff=[]
    th=230
    for i in range(15):
        #i=9
        a=math.tan(0.2*i)
        b=cy-a*cx
        part1=[]
        part2=[]
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if y<a*x+b and img[x,y,0]<th:
                    part1.append(img[x,y,0])
                if y>a*x+b and img[x,y,0]<th:
                    part2.append(img[x,y,0])
        part1cut=np.array(pd.cut(part1,range(0,th,2)).value_counts().tolist())
        part2cut=np.array(pd.cut(part2,range(0,th,2)).value_counts().tolist())
        part2cut=part2cut/len(part2)*len(part1)
        c=abs(part1cut-part2cut)
        diff.append(np.sum(c))
    angle=0.2*diff.index(min(diff))
    #print(diff,diff.index(min(diff)))
    a=math.tan(angle)
    b=round(cy-a*cx)
    #cv2.line(img,(cy,cx),(b,0),(250,0,0),1)
    M=cv2.getRotationMatrix2D((cy,cx),0-angle*57.3,1)
    img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]),borderValue=(255,255,255))
    return img

def find_212224angle_regression(image,outputs,outputs2):
    if outputs>1:
        outputs=1.0
    angle=math.acos(abs(outputs))*180/math.pi
    if outputs>0 and outputs2>0:
        finalangle=angle
    if outputs<0 and outputs2>0:
        finalangle=180-angle
    if outputs<0 and outputs2<0:
        finalangle=180+angle
    if outputs>0 and outputs2<0:
        finalangle=360-angle
    #print(img,outputs,outputs2,finalangle)
    #    num=num+1
    if min([360-finalangle,finalangle])>30:
    #        print('rotate')
        [cx,cy]=[round(image.shape[0]/2),round(image.shape[1]/2)]
        M=cv2.getRotationMatrix2D((cy,cx),0-finalangle,1)
        image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]),borderValue=(255,255,255))
    return image

def show_mask_result(img,
                    result,
                    dataset='coco',
                    score_thr=0.7,
                    with_mask=True):
        segm_result=None
        if with_mask:
            bbox_result, segm_result = result
        else:
            bbox_result=result
        if isinstance(dataset, str):#  add own data label to mmdet.core.class_name.py
            class_names = get_classes(dataset)
            # print(class_names)
        elif isinstance(dataset, list):
            class_names = dataset
        else:
            raise TypeError('dataset must be a valid dataset name or a list'
                            ' of class names, not {}'.format(type(dataset)))
        h, w, _ = img.shape
        img_origin=img[:,:,0].copy()
        img_show = img[:h, :w, :]
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)# len(labels)=100
        bboxes = np.vstack(bbox_result)# bboxes.shape=(100, 5)
        
        single_range=int(max(max(bboxes[:,2]-bboxes[:,0]),max(bboxes[:,3]-bboxes[:,1])))+70
        img_seg=np.ones([single_range*4,single_range*14,3])*255
        img_seg_null=np.ones([single_range*4,single_range*14,3])*255
        labelsize=float(img_seg.shape[0]/400)/3
        allimage=[]
        alllabel=[]
        tosave=[]
        #print(img_seg.shape)
        if with_mask:
            segms = mmcv.concat_list(segm_result)# len(segms)=100, each one is an image with single mask
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            print(inds)
            while len(inds)<45 and score_thr>0.5:
                score_thr=score_thr-0.1
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                #print(score_thr)
                if len(inds)>46:
                    score_thr=score_thr+0.1
                    inds = np.where(bboxes[:, -1] > score_thr)[0]
                    break
            print(score_thr,len(inds))
            segnum.append(len(inds))
            img_seg=cv2.putText(img_seg,'All segmentation num: '+str(len(inds)),(0,img_seg.shape[0]),cv2.FONT_HERSHEY_SIMPLEX,2*labelsize,(0,0,0),2)
            num=-1
            allpred=[]
            #allpredgroup3=[]
            #allpredgroup7=[]
            allpred5=[]
            allprob=[]
            allprob5=[]
            allcrop=[]
            mask_img=[]
            mask_bbox=[] 
            sizes=[]
            times=0
            for i in inds:
                num=num+1
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask = segms[i]
                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
                img_single=img_origin*mask
                mask_img.append(img_single)
                mask_bbox.append(bboxes[i,:])
                imgtmp=img_single[int(bboxes[i,1]):int(bboxes[i,3]),int(bboxes[i,0]):int(bboxes[i,2])]
                imgtmp=np.where(imgtmp==0,255,imgtmp)
                row,col=imgtmp.shape
                gray=imgtmp.copy()
                _,bi=cv2.threshold(gray,250,255,cv2.THRESH_BINARY)
                bi_=255-bi
                contours, hierarchy = cv2.findContours(bi_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                areas=[]
                for contour in contours:
                    areas.append(cv2.contourArea(contour))
                rect = cv2.minAreaRect(contours[areas.index(max(areas))])
                angle=rect[2]
                heigh=int(col*fabs(sin(radians(angle)))+row*fabs(cos(radians(angle))))
                width=int(row*fabs(sin(radians(angle)))+col*fabs(cos(radians(angle))))
                M=cv2.getRotationMatrix2D((col/2,row/2),angle,1)
                M[0,2]+=(width-col)/2
                M[1,2]+=(heigh-row)/2
                image1 = cv2.warpAffine(gray,M,(width,heigh),borderValue=(255,255,255))
                _,bi=cv2.threshold(image1,250,255,cv2.THRESH_BINARY)
                bi_=255-bi
                contours, hierarchy = cv2.findContours(bi_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                areas=[]
                for contour in contours:
                    areas.append(cv2.contourArea(contour))
                rect = cv2.minAreaRect(contours[areas.index(max(areas))])
                box = np.int0(cv2.boxPoints(rect))
                image2=image1[min(box[:,1]):max(box[:,1]),min(box[:,0]):max(box[:,0])]
                if min(image2.shape)<10:
                    image2=image1
                if image2.shape[0]<image2.shape[1]:
                    image2=np.rot90(image2)
                imgtmp11=np.ones([image2.shape[0],image2.shape[1],3])
                imgtmp11[:,:,0]=image2.copy()
                imgtmp11[:,:,1]=image2.copy()
                imgtmp11[:,:,2]=image2.copy()
                imgtmp11=myenh(imgtmp11.astype(np.uint8))
                imgtmpsave=imgtmp11.copy()
                imgtmp11 = Image.fromarray(imgtmp11)
                imgtmp11 = transform(imgtmp11)
                imgtmp11 = imgtmp11.unsqueeze(0)
                inputs = Variable(imgtmp11.cuda())
                sizes.append(imgtmpsave.shape[0])
                allcrop.append(imgtmpsave)
            sizesq=max(sizes)*1+1
            allcropsq=[]
            start=time.time()
            for i in allcrop:
                imgtmpsavesq=np.ones([sizesq,sizesq,3])*255
                start0=round((sizesq-i.shape[0])/2)
                start1=round((sizesq-i.shape[1])/2)
                imgtmpsavesq[start0:start0+i.shape[0],start1:start1+i.shape[1],:]=i
                allcropsq.append(imgtmpsavesq.astype(np.uint8))
            label,probs,labelpola=chrom.detectcase(allcropsq)
            for i in range(label.shape[0]):
                #print(labelpola[i,])
                if labelpola[i,0]<labelpola[i,1]:
                    allcrop[i] = np.flip(allcrop[i])
            print('recognition time pass:', time.time()-start, 's')
            print(label.shape,probs.shape,labelpola.shape)
            for i in range(label.shape[0]):
                allpred.append(label[i,0]+1)
                allpred5.append((label[i]+1).tolist())
                allprob.append(probs[i,0])
                allprob5.append(probs[i].tolist())


            num=-1
            last_label='-1'
            putline=0
            putrow=0
            label_pair=[]
            imgtmp_pair=[]
            probtmp_pair=[]
            ###post process# allcrop  allpred
            countid=np.zeros(len(typelist))
            for i in range(len(typelist)):
                countid[i]=allpred.count(int(typelist[i]))
            problemid=[]
            for i in range(len(typelist)-2):
                if countid[i]!=2:
                    problemid.append(i+1)
            if countid[22]+countid[23]!=2:
                problemid.append(23)
                problemid.append(24)


            #postprocess
            allpred_rank=sorted(range(len(allpred)), key=lambda k: allpred[k])
            for i in range(len(typelist)):
                countid[i]=allpred.count(int(typelist[i]))
            for i in range(len(allpred_rank)):
                imgtmp11=allcrop[allpred_rank[i]]
                label=str(allpred[allpred_rank[i]])
                probtmp=str(allprob[allpred_rank[i]])
                if label!=last_label and len(label_pair)>0:
                # draw a pair
                    if int(label_pair[0])<6:
                        putline=0
                    if int(label_pair[0])>=6 and int(label_pair[0])<13:
                        putline=1
                    if int(label_pair[0])>=13 and int(label_pair[0])<19:
                        putline=2
                    if int(label_pair[0])>=19 and int(label_pair[0])<25:
                        putline=3
                    if int(label_pair[0])>=25:
                        putline=4
                    putrow=find_start_row(label_pair[0],single_range)
                    putrowstart=find_start_row(label_pair[0],single_range)
                    chromheight=[]
                    for j in range(len(imgtmp_pair)):
                        num=num+1
                        imgtmp12=imgtmp_pair[j]
                        probtmp2=probtmp_pair[j]
                        img_seg[10+putline*single_range:10+putline*single_range+imgtmp12.shape[0],putrow:putrow+imgtmp12.shape[1],:]=imgtmp12
                        allimage.append(imgtmp12)
                        alllabel.append(label_pair[j])
                        putrow=putrow+imgtmp12.shape[1]+25
                        chromheight.append(imgtmp12.shape[0])
                    if label_pair[j]=='23':
                        label_pair[j]='X'
                    if label_pair[j]=='24':
                        label_pair[j]='Y'
                    img_seg=cv2.putText(img_seg,label_pair[j],(int(0.5*putrow+0.5*putrowstart),45+putline*single_range+max(chromheight)),cv2.FONT_HERSHEY_SIMPLEX,labelsize,(255,0,0),2)
                    label_pair=[]
                    imgtmp_pair=[]
                    probtmp_pair=[]
                    label_pair.append(label)
                    imgtmp_pair.append(imgtmp11)
                    probtmp_pair.append(probtmp)
                else:
                    label_pair.append(label)
                    imgtmp_pair.append(imgtmp11)
                    probtmp_pair.append(probtmp)
                last_label=label

            # draw the last pair
            if len(label_pair)>0:
                putline=3
                putrow=find_start_row(label_pair[0],single_range)
                putrowstart=find_start_row(label_pair[0],single_range)
                chromheight=[]
                for j in range(len(imgtmp_pair)):
                    num=num+1
                    imgtmp12=imgtmp_pair[j]
                    chromheight.append(imgtmp12.shape[0])
                    img_seg[10+putline*single_range:10+putline*single_range+imgtmp12.shape[0],putrow:putrow+imgtmp12.shape[1],:]=imgtmp12
                    allimage.append(imgtmp12)
                    alllabel.append(label_pair[j])
                    putrow=putrow+imgtmp12.shape[1]+25
                if label_pair[j]=='23':
                    label_pair[j]='X'
                if label_pair[j]=='24':
                    label_pair[j]='Y'
                img_seg=cv2.putText(img_seg,label_pair[j],(int(0.5*putrow+0.5*putrowstart),45+putline*single_range+max(chromheight)),cv2.FONT_HERSHEY_SIMPLEX,labelsize,(255,0,0),2)
                label_pair=[]
                imgtmp_pair=[]
            cv2.imwrite(new_path1,img_seg)
            print(alllabel)
            print('see result at:'+new_path1)
            return countid


resize_reso=512
crop_reso=480
transform = transforms.Compose([
        transforms.Resize((resize_reso, resize_reso)),
        transforms.CenterCrop((crop_reso, crop_reso)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])



model = init_detector(config_path, model_path)
chrom = Chrom('model/R_model_resnet50_MFIM_DAM-best.pth')
segnum=[]
typelist=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
titlelist=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']
allcounterr=np.array(np.zeros(len(typelist)))
allcounterr_count=np.array(np.zeros([6,len(typelist)]))
print(allcounterr_count.shape)
polachanges=[]
img_names=[]
for img in img_list:
    print()
    print(img)
    #img='/home/bluecat/project/chromosome/rawdata/20200702/A108/19-7232.0050.A1.png'
    img_name=os.path.basename(img)
    new_path=img.replace("A1.png", "C.png")
    new_path1=img.replace("A1.png", "D.png")
    polachange=0
    startall=time.time()
    start=time.time()
    result=inference_detector(model, img)
    print('segmentation time pass:', time.time()-start, 's')
    countid=show_mask_result(mmcv.imread(img), result,score_thr=0.8,with_mask=True)
    counterr=np.array(countid)
    allcounterr=np.vstack((allcounterr,counterr))
    print(polachange)
    polachanges.append(polachange)
    img_names.append(img_name)
    print('whole pipline time pass:', time.time()-startall, 's')
