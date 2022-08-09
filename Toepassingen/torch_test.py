import torch
import os
import numpy as np
print(torch.cuda.is_available())

datapath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'

cropped_new = np.load(datapath+'\e_box.npy')

print(cropped_new)