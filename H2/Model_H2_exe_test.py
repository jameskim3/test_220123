import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# import torch.nn as nn
import torch

from tkinter import *
from tkinter import ttk

import warnings
warnings.filterwarnings(action='ignore') 
from zUtils import ClassificationLoader
from zModels import get_model_effi_b4
from zUtils import Utils
from zUtils import History
from zEngine import Engine

bs_train=16
bs_valid=12
history=History(model_name="effinet")

import datetime
def loop_train(fold, weights,sel_pos):
    train=pd.read_csv("E:/kaggle_imgs/H2/data/train_fold.csv")
    history.initial_info(sel_pos)
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
        history.add_train_info(sel_pos,train_acc,train_loss,train_labels,train_preds,valid_acc,valid_loss,valid_labels,valid_preds)
        tm=datetime.datetime.now().strftime("%H:%M:%S")
        print(f"{tm}, fold={fold}, epoch={epoch}, train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")
        
        if valid_loss<best_loss :
            best_loss=valid_loss
            tm=datetime.datetime.now().strftime("%m%d")
            torch.save(model.state_dict(),f"E:/kaggle_imgs/H2/saved_models/model_fold_{fold}_{tm}.bin")
            early_stopping_cnt=0
        else:
            early_stopping_cnt+=1
        if early_stopping_cnt>=early_stopping:
            break

    print(f"fold={fold}, best val loss={best_loss}")

def predict(fold):
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
    model_save_path=f"E:/kaggle_imgs/H2/saved_models/model_fold_{fold}_{tm}.bin"
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
    traced_script_module.save(f"E:/kaggle_imgs/H2/saved_models/traced_{model_name}_fold_{fold}_{tm}.pt")
    return preds
if __name__ =="__main__":
    root = Tk()
    root.title("Test")
    root.geometry("320x240")

    label=Label(root,text="0")
    label.pack()
    count=0
    def myfunc():
        loop_train(fold=0,weights=[2,1],sel_pos=0)
        loop_train(fold=1,weights=[2,1],sel_pos=1)
        loop_train(fold=2,weights=[2,1],sel_pos=2)
        loop_train(fold=3,weights=[2,1],sel_pos=3)
        loop_train(fold=4,weights=[2,1],sel_pos=4)

        predict(0)
        predict(1)
        predict(2)
        predict(3)
        predict(4)
        
        global count
        count+=1
        label.config(text=str(count))
    btn = Button(root, text="OK",command=lambda:myfunc())
    btn.pack()

    root.mainloop()

