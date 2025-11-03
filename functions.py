import torch
import random
import os
import operator

import pandas as pd
import torch.optim as optim

from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import datasets, transforms

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from models import FCVar
from wrapper import create_folder, save_report

# Dataset PyTorch personalizzato
class EmbeddingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

def dataset_operations(dataset_name, train_batch_size, test_batch_size, seed):


    ########################
    ##   TO BE DEFINED    ##
    ########################

    train_set = True
    test_set = True


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

# Funzione di addestramento
def train(learning_rate, dropout, hidden_layers_dim, batch_size_train, batch_size_test, weight_decay, epochs, trial, trial_path=False, verbose=False, seed = False):

    torch.manual_seed(0)
    device = torch.device("cpu")

    # If seed is not defined, generate it
    if not seed:
        seed = random.randint(0,1000)
    
    train_loader, test_loader = dataset_operations('dataset_path', batch_size_train, batch_size_test, seed)

    model = FCVar(784, 10, dropout, hidden_layers_dim)
    model.to(device)

    # Define loss and optimizier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Objective Function Target
    max_test_accuracy = 0
    # Accuracy List for Graphs
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        correct = 0

        #Training
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass e optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Running Metrics
            preds = F.log_softmax(outputs, dim=1).argmax(dim=1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()

            running_loss += loss.item()

        #Training Metrics
        accuracy_train_set = correct / total
        avg_train_loss = running_loss / len(train_loader)
        train_accuracies.append(accuracy_train_set)

        #Reset Metrics for Testing
        total = 0
        correct = 0

        #Testing
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                #Forward Pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

                #Running Metrics
                preds = F.log_softmax(outputs, dim=1).argmax(dim=1)
                total += y_batch.size(0)
                correct += (preds == y_batch).sum().item()

        #Testing Metrics
        accuracy_test_set = correct / total
        test_accuracies.append(accuracy_test_set)

        avg_test_loss = test_loss / len(test_loader)

        if accuracy_test_set > max_test_accuracy:
            max_test_accuracy = accuracy_test_set
            if verbose:
                print(f'New Max Test Accuracy: {max_test_accuracy:2.4%}')
        

        epoch_report = f"Epoch {epoch+1:{len(str(epochs))}d}/{epochs} | TrL: {avg_train_loss:.4f} | TeL: {avg_test_loss:.4f} | AcTr {accuracy_train_set:7.2%} | AcTe: {accuracy_test_set:7.2%}"
        
        if verbose:
            print(epoch_report)

        if trial_path:
            if accuracy_test_set > 0.20:
                str_temp = trial_path+"/" + f'[T{trial}]-[E{epoch+1:{len(str(epochs))}d}]-[ACC-{accuracy_test_set:2.2%}].pt'
                torch.save(model, str_temp)
                print(f'Model saved with Test Accuracy: {accuracy_test_set:2.2%}')

            save_report(trial_path, trial, epoch_report)

    return max_test_accuracy, seed


def objective(trial, exp_name):

    print(exp_name)

    num_layers = trial.suggest_int('num_layers', 2, 8)
    hidden_layers_dim = []

    batch_size_train = 1500
    batch_size_test = 1500
    
    lr = trial.suggest_float('learning_rate', 0.00005, 0.0003)
    weight_decay = trial.suggest_float('weight_decay', 0.001, 0.05)
    dropout = trial.suggest_float('dropout', 0.15, 0.35)

    #step_size = trial.suggest_int('step_size', 50, 100)
    #gamma = trial.suggest_float('gamma', 0.5, 0.9)

    epochs = 3

    for i in range(0, num_layers):
        hidden_layers_dim.append(trial.suggest_int(f'dim_layer_{i}', 25, 350))
        
    print('Starting Trial', trial.number)
    print('Hyperparameters chosen: ')
    print('Learning Rate: ', lr)
    print('Hidden Layers: ', num_layers)
    print('Dimensions : ', hidden_layers_dim)
    print('Dimensions : ', dropout)
    print('Batch Train: ', batch_size_train)
    print('Batch Test: ', batch_size_test)
    print('Weight Decay: ', weight_decay)
    
    trial_path = create_folder(exp_name, trial.number)

    with open(trial_path+"/"+str(trial.number)+"-report.txt", "a") as f:
            f.write(f'Starting Trial {trial.number} \n')
            f.write(f'Hyperparameters chosen: ')
            f.write(f'Learning Rate: {lr} \n')
            f.write(f'Hidden Layers: {num_layers} \n')
            f.write(f'Dimensions : {hidden_layers_dim} \n')
            f.write(f'Dropout :  {dropout} \n')
            f.write(f'Batch Train: {batch_size_train} \n')
            f.write(f'Batch Test: {batch_size_test} \n')
            f.write(f'Weight Decay: {weight_decay} \n')
    f.close()

    of, seed = train(lr, dropout, hidden_layers_dim, batch_size_train, batch_size_test, weight_decay, epochs, trial.number, trial_path, verbose=True)
    trial.set_user_attr("Seed", seed)

    return of