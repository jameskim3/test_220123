import torch
import numpy as np
import cv2

class ClassificationLoader:
    def __init__(self, image_paths, targets, resize, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        targets = self.targets[item]

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "img":torch.tensor(image, dtype=torch.float),
            "tar":torch.tensor(targets, dtype=torch.long)
        }

from torch.utils.data.dataset import Dataset
from zModels import EffiNet,EffiNet7,Densenet,Resnet50
class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        
    def rand_bbox(self, size, lam):
        if len(size) == 4:
            W = size[2]
            H = size[3]
        elif len(size) == 3:
            W = size[1]
            H = size[2]
        else:
            raise Exception

        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def onehot(self,size, target):
        vec = torch.zeros(size, dtype=torch.float32)
        vec[target] = 1.
        return vec
    
    def __getitem__(self, index):
        data=self.dataset[index]
        img, lb = data["img"],data["tar"]
        lb_onehot = self.onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            data=self.dataset[rand_index]
            img2, lb2 = data["img"],data["tar"]
            lb2_onehot = self.onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return {
            "img":torch.tensor(img, dtype=torch.float),
            "tar":torch.tensor(lb_onehot, dtype=torch.long)
        }

    def __len__(self):
        return len(self.dataset)

import albumentations as A
from albumentations.pytorch import ToTensor
SIZE=224
import random

from albumentations import ( HorizontalFlip, IAAPerspective, ShiftScaleRotate, 
CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, 
MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine, IAASharpen, 
IAAEmboss, Flip, OneOf, Compose, Rotate, Cutout, HorizontalFlip, Normalize ) 

class Utils:
    def __init__():
        pass
    def get_aug(mode="train"):
        if mode=="Nor":
            aug=A.Compose([
                ToTensor(),
            ])
        elif mode =="train":
            print("train aug")
            mean = (0.485,0.456,0.406)
            std = (0.229,0.224,0.225)
            aug=A.Compose([
                A.Flip(),
                A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
                # Pixels
                A.OneOf([
                    A.IAAEmboss(p=1.0),
                    A.IAASharpen(p=1.0),
                    A.Blur(p=1.0),
                ], p=0.5),
                # Affine
                A.OneOf([
                    A.ElasticTransform(p=1.0),
                    A.IAAPiecewiseAffine(p=1.0)
                ], p=0.5),

                A.Normalize(mean=mean,std=std,max_pixel_value=255.0,always_apply=True),
            ])
        else:
            print("valid/test aug")
            mean = (0.485,0.456,0.406)
            std = (0.229,0.224,0.225)
            aug=A.Compose([
                A.Normalize(mean=mean,std=std,max_pixel_value=255.0,always_apply=True),
            ])

        return aug 

class History:
    def __init__(self,model_name):
        self.train_info=dict()
        self.valid_info=dict()
        self.model_name=model_name
    def initial_info(self,sel_pos):
        categories=["Acc","Loss","Labels","Preds"]
        for cat in categories:
            key=f"F{sel_pos}_{cat}"
            self.train_info[key]=[]
            self.valid_info[key]=[]
    def add_train_info(self,sel_pos, train_acc,train_loss,train_labels,train_preds,valid_acc,valid_loss,valid_labels,valid_preds):
        categories=["Acc","Loss","Labels","Preds"]
        key=f"F{sel_pos}_Acc"
        self.train_info[key].append(train_acc)
        key=f"F{sel_pos}_Loss"
        self.train_info[key].append(train_loss)
        key=f"F{sel_pos}_Acc"
        self.valid_info[key].append(valid_acc)
        key=f"F{sel_pos}_Loss"
        self.valid_info[key].append(valid_loss) 
        key=f"F{sel_pos}_Labels"
        self.valid_info[key].append(valid_labels) 
        key=f"F{sel_pos}_Preds"
        self.valid_info[key].append(valid_preds)

import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import itertools
import seaborn as sns
class ConfusionMatrix:
    def __init__(self):
        pass
    def sigmoid(x):
        return 1 / (1 +np.exp(-x))
    def get_cm(y_true,y_pred):
        cm=metrics.confusion_matrix(y_true, sigmoid(y_pred)>0.5)
        return cm
    def plot_roc_curve(y_true,y_pred):
        fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred)
        sns.lineplot(x=fpr,y=tpr)        
    def plot_confusion_matrix(cm,
                            x_target_names = ['Pred False', 'Pred True'],
                            y_target_names = ['Real False', 'Real True'],
                            title='Confusion matrix',
                            cmap=None,
                            normalize=False):
 

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if x_target_names is not None:
            tick_marks = np.arange(len(x_target_names))
            plt.xticks(tick_marks, x_target_names, rotation=45)
        if y_target_names is not None:
            tick_marks = np.arange(len(y_target_names))
            plt.yticks(tick_marks, y_target_names, rotation=45)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel(f'Predicted label\nPrecision={cm[1,1]/cm.sum(0)[1]:0.4f}; Recall={cm[1,1]/cm.sum(1)[1]:0.4f}\n TPR={cm[1,1]/cm.sum(1)[1]:0.4f}; FPR={cm[1,0]/cm.sum(0)[0]:0.4f}')
        plt.show()  
        
