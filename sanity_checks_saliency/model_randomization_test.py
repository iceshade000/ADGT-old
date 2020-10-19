from .. import ADGT
import torch
import numpy as np
import cv2
import os

import torchvision

MODEL='inception_v3'
method=['SmoothGrad','InputXGradient','Guided_BackProb','Saliency','DeepLIFT','RectGrad','PatternNet']
layername=['fc','Mixed_7c','Mixed_7b','Mixed_7a','Mixed_6e','Mixed_6d','Mixed_6c','Mixed_6b','Mixed_6a',
           'Mixed_5d','Mixed_5c','Mixed_5b','Conv2d_4a_3x3','Conv2d_3b_1x1','Conv2d_2b_3x3','Conv2d_2a_3x3',
           'Conv2d_1a_3x3']
use_cuda=True
DATASET_NAME='ImageNet'
ROOT='result'
AUG=True

def prepare_model(model_name=MODEL):
    if model_name=='inception_v3':
        model=torchvision.models.inception_v3(pretrained=True)
    elif model_name=='resnet50':
        model=torchvision.models.resnet50(pretrained=True)
    if use_cuda:
        model=model.cuda()
    return model

def prepare_img(path):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    dirs = os.listdir(path)
    imgs = []
    for fn in dirs:
        img_path = path + '/' + fn
        img = cv2.imread(img_path, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        imgs.append(img)
    imgs = np.array(imgs)

    preprocessed_imgs = imgs.copy()[:, :, :, ::-1]
    for i in range(3):
        preprocessed_imgs[:, :, :, i] = preprocessed_imgs[:, :, :, i] - means[i]
        preprocessed_imgs[:, :, :, i] = preprocessed_imgs[:, :, :, i] / stds[i]
    preprocessed_imgs = \
        np.ascontiguousarray(np.transpose(preprocessed_imgs, (0, 3, 1, 2)))
    preprocessed_imgs = torch.from_numpy(preprocessed_imgs)
    return preprocessed_imgs

max_K=5
net=prepare_model()
img=prepare_img(os.path.join('sanity_checks_saliency','data','demo_images'))
img=img[:max_K]
pred=net(img)
_,topklabel=torch.topk(pred,max_K)
_,target=torch.topk(pred,1)

def class_independent(img,target,topklabel,method):
    pth=os.path.join(ROOT,'class_independent')
    adgt=ADGT.ADGT(use_cuda=use_cuda,name=DATASET_NAME,aug=AUG)
    pth_raw=os.path.join(pth,'raw')
    pth_split=os.path.join(pth,'split')
    for m in range(method):
        adgt.explain_all(img, target, logdir=pth_raw, method=m, model=net, random=False, attack=False,suffix='',topklabel=topklabel)
        adgt.explain_split(img, target, logdir=pth_split, method=m, model=net, random=False, attack=False,suffix='',topklabel=topklabel)

def model_independent_cascade(img,target,topklabel,method,layer_names):
    pth=os.path.join(ROOT,'model_independent_cascade')
    adgt=ADGT.ADGT(use_cuda=use_cuda,name=DATASET_NAME,aug=AUG)
    pth_raw=os.path.join(pth,'raw')
    pth_split=os.path.join(pth,'split')
    for i in range(len(layer_names)):
        net=perturbation(net,layer_names[i])
        for m in range(method):
            adgt.explain_all(img, target, logdir=os.path.join(pth_raw,str(i)), method=m, model=net, random=False,
                             attack=False,suffix='',topklabel=topklabel)
            adgt.explain_split(img, target, logdir=os.path.join(pth_split,str(i)), method=m, model=net, random=False,
                               attack=False,suffix='',topklabel=topklabel)
import copy
def model_independent_individual(img,target,topklabel,method,layer_names):
    pth=os.path.join(ROOT,'model_independent_individual')
    adgt=ADGT.ADGT(use_cuda=use_cuda,name=DATASET_NAME,aug=AUG)
    pth_raw=os.path.join(pth,'raw')
    pth_split=os.path.join(pth,'split')

    for i in range(len(layer_names)):
        temp_net=copy.deepcopy(net)
        temp_net=perturbation(temp_net,layer_names[i])
        for m in range(method):
            adgt.explain_all(img, target, logdir=os.path.join(pth_raw,str(i)), method=m, model=temp_net, random=False,
                             attack=False,suffix='',topklabel=topklabel)
            adgt.explain_split(img, target, logdir=os.path.join(pth_split,str(i)), method=m, model=temp_net, random=False,
                               attack=False,suffix='',topklabel=topklabel)
import torch.nn as nn
def perturbation(net,layername):
    def weights_init(m,layername):
        classname = m.__class__.__name__
        if classname.find(layername):
            # print(classname)
            if classname.find('Conv') != -1:
                nn.init.xavier_normal_(m.weight.data)
            elif classname.find('Linear') != -1:
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
    net.apply(weights_init,layername)
    return net

