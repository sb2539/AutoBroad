import numpy as np
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def z_score(list):
    nomallized = []
    for value in list:
        nomallized_value = (value-np.mean(list))/np.std(list)
        nomallized.append(nomallized_value)
    return nomallized

def min_max(list):
    nomalized = []
    for value in list:
        nomalized_val = (value-min(list))/(max(list)-min(list))
        nomalized.append(nomalized_val)
    return nomalized

class stacking(nn.Module):
    def __init__(self, img_size, data_num, channel_num, choosed_encod):
        super().__init__()
        self.img_size = img_size
        self.data_num = data_num
        self.channel_num = channel_num
        self.choosed_encod = choosed_encod
    def forward(self, x):
        stack = np.array([]).reshape(-1, 1, self.img_size, self.img_size)
        for i, name in enumerate(self.choosed_encod):
            #x[name] = resize(x[name],((self.data_num, 1,self.img_size, self.img_size)))
            #print(x[name].shape)
            if i == 0 :
                stack = x[name]
            else :
                stack = np.concatenate((stack, x[name]), axis=1)
        return stack