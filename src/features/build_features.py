import torch
import string
import numpy as np

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)

def letterToVector(letter):
    tensor = torch.zeros(n_letters)
    letterIdx = all_letters.find(letter)
    tensor[letterIdx] = 1
    return tensor

def nameToTensor(name):
    tensor = torch.zeros(len(name), n_letters)
    for  idx, c in enumerate(name):
        tensor[idx] = letterToVector(c)
    return tensor.unsqueeze(1)

def build_features(category_lines):  
    X = []
    y = []

    for cat, names in category_lines.items():
        for name in names:
            X.append(name)
            y.append(cat)  

    X = np.array(X)
    y = np.array(y)
    return X, y