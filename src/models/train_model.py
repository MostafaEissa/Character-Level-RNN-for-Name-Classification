import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import numpy as np
from features.build_features import letterToVector, nameToTensor, build_features

def train_test_split(data, labels, test_size=0.2, random_state=0):
    np.random.seed(random_state)
    N = labels.shape[0]
    idx = np.random.permutation(N)
    train_size = int(np.ceil((1-test_size)*N))
    X_train = data[idx[:train_size]]
    y_train = labels[idx[:train_size]]
    X_test = data[idx[train_size:]]
    y_test = labels[idx[train_size:]]
    return X_train, X_test, y_train, y_test

class ListDataset(Dataset):
    
    def __init__(self, *lists):
        self.lists = lists

    def __getitem__(self, index):
        return tuple(lst[index] for lst in self.lists)

    def __len__(self):
        return len(self.lists[0])

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_to_output = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        combined_state = torch.cat((input, hidden), 1)
        hidden = self.input_to_hidden(combined_state)
        output = self.input_to_output(combined_state)
        output = self.softmax(output)
        return output, hidden


def train(model, X, y, categories, num_epochs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=113)
    train_data = ListDataset(X_train, y_train)
    val_data = ListDataset(X_test, y_test)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1)
    val_dataloader = DataLoader(dataset=val_data, batch_size=1)
    dataloaders = {'train': train_dataloader, 'val':val_dataloader}
    dataset_sizes = {'train': len(X_train), 'val': len(X_test)}

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    return train_model(model,categories, criterion, optimizer, dataset_sizes, dataloaders, bs=1, num_epochs=num_epochs)


def train_model(model, categories, criterion, optimizer, dataset_sizes, dataloaders, bs, num_epochs):
    
    since = time.time()
    epoch_loss = []
    epoch_acc = []

    print_every = int((dataset_sizes['train']/bs)/10)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            i = 0
            running_loss = 0.0
            running_corrects = 0
            
            if phase == 'train':
                print(f'\rProgress:',end='')
                    
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                #initialize the hidden state
                hidden = torch.zeros(1, model.n_hidden)
                
                
                input_tensor = nameToTensor(inputs[0])
                label_tensor = torch.tensor([categories.index(labels[0])], dtype=torch.long) 

                # forward
                with torch.set_grad_enabled(phase == 'train'):

                    for ix in range(input_tensor.size()[0]):
                        output, hidden = model(input_tensor[ix], hidden)
                    
                    _, preds = torch.max(output, 1)
                    loss = criterion(output, label_tensor)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == label_tensor.data)

                if phase == 'train' and i % print_every == print_every - 1:
                    print(f"\rProgress: [{'='*((i+1)//print_every)}] ",end='')
                        
        
                i += 1  
                
            epoch_loss.append(running_loss / dataset_sizes[phase])
            epoch_acc.append(running_corrects.numpy() / dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss[-1], epoch_acc[-1]))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return epoch_loss, epoch_acc