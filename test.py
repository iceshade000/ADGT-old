import ADGT
import os
from model import resnet,resnet_small
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
import torch
import torchvision
from utils import obtain_transform

use_cuda=True
method=['SmoothGrad','InputXGradient','Guided_BackProb','Saliency','DeepLIFT','RectGrad','IntegratedGradients']
ROOT='/newsd4/zgh/data'
CKPTDIR='/newsd4/zgh/ADGT/CKPT'
gamma=0.2
BATCHSIZE=128
AUG=True
MODEL='vgg'#'resnet'#'linear'#
CKPTDIR=os.path.join(CKPTDIR,MODEL)
DATASET_NAME='C10'#'MNIST'#'Flower102'#'C100'#
PROB=0.1
PLUGIN=1

torch.set_num_threads(4)
seed = 0
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
import random,numpy
random.seed(seed)
numpy.random.seed(seed)

adgt=ADGT.ADGT(use_cuda=use_cuda,name=DATASET_NAME,aug=AUG)
transform_train,transform_test=obtain_transform.obtain_transform(DATASET_NAME)

adgt.prepare_dataset_loader(root=ROOT, train=True, transform=transform_test, batch_size=BATCHSIZE, shuffle=True)
if MODEL=='resnet':
    if adgt.dataset_name == 'MNIST':
        net = resnet_small.resnet18(indim=1, num_class=10)
    elif adgt.dataset_name =='C10':
        net=resnet_small.resnet18(indim=3,num_class=10)
    elif adgt.dataset_name =='C100':
        net=resnet_small.resnet18(indim=3,num_class=100)
    elif adgt.dataset_name =='Flower102':
        net=resnet.resnet50(num_classes=102)
elif MODEL=='linear':
    if adgt.dataset_name == 'MNIST':
        net = torch.nn.Linear(in_features=3,out_features=10)
    elif adgt.dataset_name =='C10':
        net = torch.nn.Linear(in_features=3,out_features=10)
    elif adgt.dataset_name =='C100':
        net = torch.nn.Linear(in_features=3,out_features=100)
    elif adgt.dataset_name =='Flower102':
        net = torch.nn.Linear(in_features=3,out_features=102)

#pth='attack_1_0.0Falseresnet'
pth='attack_1_0.0Falsevgg'
checkpointdir = os.path.join(CKPTDIR, adgt.dataset_name, pth)
adgt.load_gt(checkpointdir)

#checkpointdir = os.path.join(CKPTDIR,  adgt.dataset_name, 'normal'+str(AUG))
#adgt.load_normal(checkpointdir)
#checkpointdir = os.path.join(CKPTDIR, adgt.dataset_name, 'removeSPP'+'_'+str(gamma)+str(AUG))
#adgt.load_improve(checkpointdir)
#checkpointdir = os.path.join(CKPTDIR,  adgt.dataset_name, 'mixup_'+str(1)+str(AUG))
#checkpointdir = os.path.join(CKPTDIR,  adgt.dataset_name, 'adversarial_'+'l2'+'_'+str(0.3)+str(AUG))
#checkpointdir = os.path.join(CKPTDIR,  adgt.dataset_name, 'adversarial_'+'l2'+'_'+str(1.5)+str(AUG))
#adgt.load_normal(checkpointdir)
K=adgt.nclass[DATASET_NAME]
NUMBER=1
iii=0
img=label=None
for data,label in adgt.trainloader:
    if img is None:
        img=data
        target=label
    else:
        img=torch.cat((img,data),0)
        target=torch.cat((target,label),0)
    iii+=1
    if iii>=3:
        break
imgtemp=[]
targettemp=[]
for i in range(K):
    imgtemp.append(img[target==i][0:NUMBER])
    targettemp.append(target[target==i][0:NUMBER])
img=torch.cat(tuple(imgtemp),0)
target=torch.cat(tuple(targettemp),0)

for m in method:
    print(m)
    if m=='IntegratedGradients':
        adgt.parallel()
    adgt.explain_all(img.clone(),target.clone(), 'result',method=m, random=False,attack=True)
    #adgt.explain(data.clone(),label.clone(), 'result',method=m, random=False,attack=False,improve=True,suffix=str(gamma))
    #adgt.explain(data.clone(), label.clone(), 'result', method=m, random=False, attack=True)

'''
prob=[0]
model=[]
for PLUGIN in range(1):
    for p in prob:
        suffix = 'wd' + str(p)
        #suffix = 'flooding'
        
        checkpointdir = os.path.join(CKPTDIR, adgt.dataset_name,
                                 'RPB_' + str(1) + '_' + str(0.2) + '_' + str(1) + str(AUG)+suffix)
        #checkpointdir = os.path.join(CKPTDIR, adgt.dataset_name,
        #                             'RPB_batch' +  '_' + str(10) + str(AUG)+suffix)
        adgt.load_RPB(checkpointdir)
        model.append(adgt.RPB_model)
        
        checkpointdir = os.path.join(CKPTDIR, adgt.dataset_name, 'normal'+str(AUG)+suffix)
        adgt.load_normal(checkpointdir)
        model.append(adgt.normal_model)



for data,label in adgt.trainloader:
        for m in method:
            print(m)
            if m=='IntegratedGradients':
                for i in range(len(prob)):
                        model[i]=torch.nn.DataParallel(model[i])
            for i in range(1):
                for p in prob:
                    suffix = 'wd' + str(p)
                    #suffix = 'flooding'
                    #pth = os.path.join('result', 'RPB_batch' +  '_' + str(10) + str(AUG)+suffix)
                    #pth= os.path.join('result','RPB_' + str(1) + '_' + str(0.2) + '_' + str(1) + str(AUG)+suffix)
                    pth= os.path.join('result','normal'+str(AUG)+suffix)
                    adgt.explain(data.clone(),label.clone(),pth, method=m,
                     model=model[i],random=False,attack=False)
        break
'''


