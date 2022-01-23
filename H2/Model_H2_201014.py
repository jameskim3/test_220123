import torch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore') 

train=pd.read_csv("E:/kaggle_imgs/Plant-pathology-2020/Data/train_fold.csv")

bs_train=20#38
bs_valid=10#20

from zUtils import CutMix
from zUtils import ClassificationLoader
from zUtils import Utils
from zEngine import Engine
from zUtils import History
from zModels import get_effinet

import datetime
def loop_train(fold, model_name,weights,sel_weight):
    history.initial_info(sel_weight)
    train_df=train[train.fold!=fold].reset_index(drop=True)#[0:65]
    valid_df=train[train.fold==fold].reset_index(drop=True)#[0:64]
    train_df["result"]=train_df["CAT"].apply(lambda x : x>0).astype(np.int)
    valid_df["result"]=valid_df["CAT"].apply(lambda x : x>0).astype(np.int)

    imgs=train_df.image_id.values.tolist()
    path="E:/kaggle_imgs/Plant-pathology-2020/images_224_224/"
    train_imgs=[path+file+".png" for file in imgs]
    train_aug=Utils.get_aug("train")
    train_tar=train_df.result.values
    train_dataset=ClassificationLoader(
        image_paths=train_imgs,targets=train_tar,resize=None,augmentations=train_aug
    )
#     CutMix_train_dataloader = CutMix(train_dataset, 
#                           num_class=4, 
#                           beta=1.0, 
#                           prob=0.999, 
#                           num_mix=1)
    CutMix_train_dataloader=train_dataset
    train_loader=torch.utils.data.DataLoader(
        CutMix_train_dataloader,batch_size=bs_train,num_workers=4,shuffle=True
    )
    
    imgs=valid_df.image_id.values.tolist()
    path="E:/kaggle_imgs/Plant-pathology-2020/images_224_224/"
    valid_imgs=[path+file+".png" for file in imgs]
    valid_aug=Utils.get_aug("valid")
    valid_tar=valid_df.result.values
    valid_dataset=ClassificationLoader(
        image_paths=valid_imgs,targets=valid_tar,resize=None,augmentations=valid_aug
    )
#     CutMix_valid_dataloader = CutMix(valid_dataset, 
#                           num_class=4, 
#                           beta=1.0, 
#                           prob=0, 
#                           num_mix=1)
    CutMix_valid_dataloader=valid_dataset
    valid_loader=torch.utils.data.DataLoader(
        CutMix_valid_dataloader,batch_size=bs_valid,num_workers=4,shuffle=False
    )
    
    # Model,Optimizer, scheduler, engine
    model=get_effinet(classes=2)
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,patience=3,threshold=1e-5,mode="min",verbose=True
    )

    engine=Engine(model,optimizer,device,classes=2,weights=weights)
    best_loss=np.inf
    early_stopping=3#3
    early_stopping_cnt=0
    EPOCH=300
    for epoch in range(EPOCH):
        train_loss,train_acc=engine.train(train_loader)
        valid_loss,valid_acc,valid_labels,valid_preds=engine.validate(valid_loader)
        scheduler.step(valid_loss)
        
        # Add train Info
        history.add_train_info(sel_weight,train_acc,train_loss,valid_acc,valid_loss,valid_labels,valid_preds)
        tm=datetime.datetime.now().strftime("%H:%M:%S")
        print(f"{tm}, fold={fold}, epoch={epoch}, train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")    
        
        if valid_loss<best_loss :
            best_loss=valid_loss
            torch.save(model.state_dict(),f"model_fold_{fold}.bin")
            early_stopping_cnt=0
            history.best_idx[fold]=epoch
        else:
            early_stopping_cnt+=1
        if early_stopping_cnt>early_stopping:
            break

    print(f"fold={fold}, best val loss={best_loss}")

bs_test=12
def predict(fold,model_name):    
    df=pd.read_csv("E:/kaggle_imgs/Plant-pathology-2020/Data/train_fold.csv")[0:10]
    device="cuda" if torch.cuda.is_available() else "cpu"
    imgs=df.image_id.values.tolist()
    path="E:/kaggle_imgs/Plant-pathology-2020/images_224_224/"
    test_imgs=[path+file+".png" for file in imgs]

    utils=Utils(mode="test")
    test_aug=utils.get_aug()
    test_tar=np.zeros((len(imgs),4))
    test_dataset=ClassificationLoader(
        image_paths=test_imgs,targets=test_tar,resize=None,augmentations=test_aug
    )
    test_loader=torch.utils.data.DataLoader(
        test_dataset,batch_size=bs_test,num_workers=4,shuffle=False
    )

    model=get_effinet(classes=2)
    model_save_path=f"./model_fold_{fold}.bin"
    model.load_state_dict(torch.load(model_save_path))
    model=model.to(device)

    engine=Engine(model,None,device,classes=2,weights=None)
    preds=engine.predict(test_loader)
    preds=np.vstack(preds)
    
    #script to c++
    sample=torch.rand(1,3,224,224)
    model.to("cpu")
    traced_script_module = torch.jit.trace(model,sample)
    traced_script_module.save("E:/temp/saved_models/"+f"/traced_1015_fold_{fold}.pt")
    
    return preds

def loop_train_process(model_name):
    loop_train(fold=0,model_name=model_name,weights=[1,1],sel_weight=0)
    # loop_train(fold=0,model_name=model_name,weights=[2,1])
    # loop_train(fold=0,model_name=model_name,weights=[4,1])
    # loop_train(fold=0,model_name=model_name,weights=[8,1])
    # loop_train(fold=0,model_name=model_name,weights=[16,1])
    # loop_train(fold=0,model_name=model_name,weight=[32,1])
    
    
    
#     loop_train(fold=1,model_name=model_name)
#     loop_train(fold=2,model_name=model_name)
#     loop_train(fold=3,model_name=model_name)
#     loop_train(fold=4,model_name=model_name)

    preds=predict(0,model_name)
    print(preds)

if __name__ == "__main__":
    history=History(model_name="effinet")
    loop_train_process(model_name="effinet")