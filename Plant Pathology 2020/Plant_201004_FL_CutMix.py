import torch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore') 

train=pd.read_csv("E:/kaggle_imgs/Plant-pathology-2020/Data/train_fold.csv")
# test=pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
# sample_submission=pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")

train_info=dict()
valid_info=dict()
def ready_train_info(fold):
    key=f"Fold[{fold}]_Acc"
    train_info[key]=[]
    key=f"Fold[{fold}]_loss"
    train_info[key]=[]    
    key=f"Fold[{fold}]_Acc"
    valid_info[key]=[]
    key=f"Fold[{fold}]_loss"
    valid_info[key]=[]     
    
def add_train_info(fold, train_acc,train_loss,valid_acc,valid_loss):
    key=f"Fold[{fold}]_Acc"
    train_info[key].append(train_acc)
    key=f"Fold[{fold}]_loss"
    train_info[key].append(train_loss)
    key=f"Fold[{fold}]_Acc"
    valid_info[key].append(valid_acc)
    key=f"Fold[{fold}]_loss"
    valid_info[key].append(valid_loss)     

bs_train=20#38
bs_valid=10#20

from zUtils import CutMix
from zUtils import ClassificationLoader
from zUtils import Utils
from zEngine import Engine

import datetime
def loop_train(fold=0):
    ready_train_info(fold)
    train_df=train[train.fold!=fold].reset_index(drop=True)#[0:32]
    valid_df=train[train.fold==fold].reset_index(drop=True)#[0:32]

    imgs=train_df.image_id.values.tolist()
    path="E:/kaggle_imgs/Plant-pathology-2020/images_224_224/"
    train_imgs=[path+file+".png" for file in imgs]
    train_aug=Utils.get_aug("train")
    #train_tar=train_df[["healthy","multiple_diseases","rust","scab"]].values
    train_tar=train_df.CAT.values
    train_dataset=ClassificationLoader(
        image_paths=train_imgs,targets=train_tar,resize=None,augmentations=train_aug
    )
    CutMix_train_dataloader = CutMix(train_dataset, 
                          num_class=4, 
                          beta=1.0, 
                          prob=0.999, 
                          num_mix=1)
    train_loader=torch.utils.data.DataLoader(
        CutMix_train_dataloader,batch_size=bs_train,num_workers=4,shuffle=True
    )
    
    imgs=valid_df.image_id.values.tolist()
    path="E:/kaggle_imgs/Plant-pathology-2020/images_224_224/"
    valid_imgs=[path+file+".png" for file in imgs]
    valid_aug=Utils.get_aug("valid")
    #valid_tar=valid_df[["healthy","multiple_diseases","rust","scab"]].values
    valid_tar=valid_df.CAT.values
    valid_dataset=ClassificationLoader(
        image_paths=valid_imgs,targets=valid_tar,resize=None,augmentations=valid_aug
    )
    CutMix_valid_dataloader = CutMix(valid_dataset, 
                          num_class=4, 
                          beta=1.0, 
                          prob=0, 
                          num_mix=1)
    valid_loader=torch.utils.data.DataLoader(
        CutMix_valid_dataloader,batch_size=bs_valid,num_workers=4,shuffle=False
    )
    
    # Model,Optimizer, scheduler, engine
    model=Utils.get_model("effinet")
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,patience=3,threshold=1e-5,mode="min",verbose=True
    )

    engine=Engine(model,optimizer,device)
    best_loss=np.inf
    early_stopping=7#3
    early_stopping_cnt=0
    EPOCH=300
    for epoch in range(EPOCH):
        train_loss,train_acc=engine.train(train_loader)
        valid_loss,valid_acc=engine.validate(valid_loader)
        scheduler.step(valid_loss)
        
        # Add train Info
        add_train_info(fold,train_acc,train_loss,valid_acc,valid_loss)
        
        if valid_loss<best_loss :
            best_loss=valid_loss
            torch.save(model.state_dict(),f"model_fold_{fold}.bin")
            tm=datetime.datetime.now().strftime("%H:%M:%S")
            print(f"{tm}, fold={fold}, epoch={epoch}, train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}")    
            early_stopping_cnt=0
        else:
            early_stopping_cnt+=1
        if early_stopping_cnt>early_stopping:
            break

    print(f"fold={fold}, best val loss={best_loss}")

bs_test=12
def predict(fold=0):    
    df=pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
    device="cuda" if torch.cuda.is_available() else "cpu"
    imgs=df.image_id.values.tolist()
    path="../input/plant-images-224-224/"
    test_imgs=[path+file+".png" for file in imgs]
    test_aug=Utils.get_aug("test")
    test_tar=np.zeros((len(imgs),4))
    test_dataset=ClassificationLoader(
        image_paths=test_imgs,targets=test_tar,resize=None,augmentations=test_aug
    )
    test_loader=torch.utils.data.DataLoader(
        test_dataset,batch_size=bs_test,num_workers=0,shuffle=False
    )

    model=get_model(model_name)
    model_save_path=f"./model_fold_{fold}.bin"
    model.load_state_dict(torch.load(model_save_path))
    model=model.to(device)

    engine=Engine(model,None,device)
    preds=engine.predict(test_loader)
    preds=np.vstack(preds)
    return preds 

if __name__ == "__main__":
    loop_train(fold=0)
    # loop_train(fold=1)
    # loop_train(fold=2)
    # loop_train(fold=3)
    # loop_train(fold=4)

    # d1=predict(0)
    # d2=predict(1)
    # d3=predict(2)
    # d4=predict(3)
    # d5=predict(4)

    # p1 = softmax(d1, axis=1)
    # p2 = softmax(d2, axis=1)
    # p3 = softmax(d3, axis=1)
    # p4 = softmax(d4, axis=1)
    # p5 = softmax(d5, axis=1)

    # p=(p1+p2+p3+p4+p5)/5
    # p = softmax(p, axis=1)
    # submission_df = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")
    # submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = p
    # submission_df.to_csv('submission.csv', index = False)