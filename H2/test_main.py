
import pandas
import numpy as np
import torch

if __name__ =="__main__":
    print(654*78)
    print(np.random.randint(0,10,(10,5)))
    print('cuda' if torch.cuda.is_available() else 'cpu')

    