'''
conda create -n tch4exe python=3.7
conda activate tch4exe

pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas
pip install pyinstaller
pip install opencv-python
pip install efficientnet_pytorch
pip install albumentations
pip install scikit-learn

'''


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch.nn as nn
import torch

from tkinter import *
from tkinter import ttk

import warnings
warnings.filterwarnings(action='ignore') 

from PIL import Image
class ClassificationLoader:
    def __init__(self, image_paths, targets, resize, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        targets = self.targets[item]
        if self.augmentations:
            image = self.augmentations(image)
        return {
            "img":torch.tensor(image, dtype=torch.float),
            "tar":torch.tensor(targets, dtype=torch.long)
        }

from efficientnet_pytorch import EfficientNet
def get_model_effi_b4(classes):
    classes=classes
    base_model = EfficientNet.from_pretrained("efficientnet-b4")
    num_ftrs = base_model._fc.in_features
    base_model._fc = nn.Linear(num_ftrs,classes, bias = True)
    return base_model 

import torchvision.transforms as transforms
class Utils:
    def __init__():
        pass
    def get_aug(mode="train"):
        if mode=="Nor":
            aug=transforms.Compose([
                transforms.Normalize(mean=mean,std=std),
            ])
        elif mode =="train":
            print("train aug2")
            mean = (0.0, 0.0, 0.0)
            std = (1., 1., 1.)
            aug=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ])
        else:
            print("valid/test aug2")
            mean = (0.0, 0.0, 0.0)
            std = (1., 1., 1.)
            aug=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ])

        return aug 

class Engine:
    def __init__(self,model,optimizer,device,classes,weights=None):
        self.model=model
        self.optimizer=optimizer
        self.device=device
        self.classes=classes
        self.weights=weights
        
        if weights is None:
            self.criterion=nn.CrossEntropyLoss()
        else:
            class_weights = torch.FloatTensor(weights).cuda()
            self.criterion=nn.CrossEntropyLoss(class_weights)
    
    def loss_fn(self,targets,outputs):
        return self.criterion(outputs,targets)
        
    def get_accuracy(self,labels,preds):
        total=labels.shape[0]
        preds=preds.argmax(1).reshape(-1,1)
        return np.uint8(labels==preds).sum()/total
    
    def train(self,data_loader):
        preds_for_acc = []
        labels_for_acc = []
        self.model.train()
        final_loss=0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs=data["img"].to(self.device)
            targets=data["tar"].to(self.device)
            outputs=self.model(inputs)
            loss=self.loss_fn(targets,outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
            ## accuracy
            labels = targets.cpu().numpy().reshape(-1,1)
            preds = outputs.cpu().detach().numpy()
            if len(labels_for_acc)==0:
                labels_for_acc = labels
                preds_for_acc = preds
            else:
                labels_for_acc=np.vstack((labels_for_acc,labels))
                preds_for_acc=np.vstack((preds_for_acc,preds))
        accuracy = self.get_accuracy(labels_for_acc,preds_for_acc)
        return final_loss/len(data_loader),accuracy,labels_for_acc,preds_for_acc
    
    def validate(self,data_loader):
        preds_for_acc = []
        labels_for_acc = []
        self.model.eval()
        final_loss=0
        for data in data_loader:
            inputs=data["img"].to(self.device)
            targets=data["tar"].to(self.device)
            with torch.no_grad():
                outputs=self.model(inputs)
                loss=self.loss_fn(targets,outputs)
                final_loss += loss.item()
            ## accuracy
            labels = targets.cpu().numpy().reshape(-1,1)
            preds = outputs.cpu().detach().numpy()
            if len(labels_for_acc)==0:
                labels_for_acc = labels
                preds_for_acc = preds
            else:
                labels_for_acc=np.vstack((labels_for_acc,labels))
                preds_for_acc=np.vstack((preds_for_acc,preds))
        accuracy = self.get_accuracy(labels_for_acc,preds_for_acc)
        return final_loss/len(data_loader),accuracy,labels_for_acc,preds_for_acc
    
    def predict(self,data_loader):
        self.model.eval()
        final_predictions = []
        with torch.no_grad():
            for data in data_loader:
                inputs=data["img"].to(self.device)
                predictions = self.model(inputs)
                predictions = predictions.cpu()
                final_predictions.append(predictions.detach().numpy())
        return final_predictions

bs_train=16
bs_valid=12

import datetime
def loop_train(fold, weights,sel_pos):
    train=pd.read_csv("E:/kaggle_imgs/H2/data/train_fold.csv")
    train_df=train[train.fold!=fold].reset_index(drop=True)[:100]
    valid_df=train[train.fold==fold].reset_index(drop=True)[:80]
    train_df["result"]=train_df["category"].apply(lambda x : x>0).astype(np.int)
    valid_df["result"]=valid_df["category"].apply(lambda x : x>0).astype(np.int)

    train_imgs=train_df.tar_path.values.tolist()
    train_aug=Utils.get_aug("train")
    train_tar=train_df.result.values
    train_dataset=ClassificationLoader(
        image_paths=train_imgs,targets=train_tar,resize=None,augmentations=train_aug
    )
    train_loader=torch.utils.data.DataLoader(
        train_dataset,batch_size=bs_train,num_workers=0,shuffle=True
    )
    
    valid_imgs=valid_df.tar_path.values.tolist()
    valid_aug=Utils.get_aug("valid")
    valid_tar=valid_df.result.values
    valid_dataset=ClassificationLoader(
        image_paths=valid_imgs,targets=valid_tar,resize=None,augmentations=valid_aug
    )
    valid_loader=torch.utils.data.DataLoader(
        valid_dataset,batch_size=bs_valid,num_workers=0,shuffle=False
    )
    
    model = get_model_effi_b4(classes=2)
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,patience=3,threshold=1e-5,mode="min",verbose=True
    )

    engine=Engine(model,optimizer,device,classes=1,weights=weights)
    best_loss=np.inf
    early_stopping=3#3
    early_stopping_cnt=0
    EPOCH=3#300
    for epoch in range(EPOCH):
        train_loss,train_acc,train_labels,train_preds=engine.train(train_loader)
        valid_loss,valid_acc,valid_labels,valid_preds=engine.validate(valid_loader)
        scheduler.step(valid_loss)
        
        # Add train Info
        tm=datetime.datetime.now().strftime("%H:%M:%S")
        print(f"{tm}, fold={fold}, epoch={epoch}, train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")
        
        if valid_loss<best_loss :
            best_loss=valid_loss
            tm=datetime.datetime.now().strftime("%m%d")
            torch.save(model.state_dict(),f"E:/kaggle_imgs/H2/saved_models/model_fold_{fold}_{tm}_ex.bin")
            early_stopping_cnt=0
        else:
            early_stopping_cnt+=1
        if early_stopping_cnt>=early_stopping:
            break

    print(f"fold={fold}, best val loss={best_loss}")

def predict(fold):
    train=pd.read_csv("E:/kaggle_imgs/H2/data/train_fold.csv")
    df=train[0:10]
    device="cuda" if torch.cuda.is_available() else "cpu"

    test_imgs=df.tar_path.values.tolist()
    test_aug=Utils.get_aug("test")
    test_tar=np.zeros((len(test_imgs),2))
    test_dataset=ClassificationLoader(
        image_paths=test_imgs,targets=test_tar,resize=None,augmentations=test_aug
    )
    test_loader=torch.utils.data.DataLoader(
        test_dataset,batch_size=bs_valid,num_workers=0,shuffle=False
    )
    model=get_model_effi_b4(2)
    tm=datetime.datetime.now().strftime("%m%d")
    model_save_path=f"E:/kaggle_imgs/H2/saved_models/model_fold_{fold}_{tm}_ex.bin"
    model.load_state_dict(torch.load(model_save_path))
    model=model.to(device)

    engine=Engine(model,None,device,classes=2,weights=None)
    preds=engine.predict(test_loader)
    preds=np.vstack(preds)#.argmax(axis=1)

    #script to c++
    sample=torch.rand(1,3,224,224)
    model.to("cpu")
    model.set_swish(False)
    traced_script_module = torch.jit.trace(model,sample)
    traced_script_module.save(f"E:/kaggle_imgs/H2/saved_models/traced_fold_{fold}_{tm}_ex.pt")
    return preds

if __name__ =="__main__":
    root = Tk()
    root.title("Test")
    root.geometry("320x240")
oloop_train(fold=4,weights=[2,1],sel_pos=4)

        # predict(0)
        p1=predict(1)
        # predict(2)
        # predict(3)
        # predict(4)
        print(p1)
        
        global count
        count+=1
        label.config(text=str(count))
    btn = Button(root, text="OK",command=lambda:myfunc())
    btn.pack()

    root.mainloop()

