from __future__ import print_function
from re import X

import sys

import torch
sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation

from torch.utils.data import DataLoader, Dataset

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)
    # state = data['state']                           # 96,96,3 x 30000
    # next_state = data['next_state']                 # 96,96,3 x 30000
    # reward = data['reward']                         # 30000
    # action = data['action']
    # terminal = data['terminal']
    # print(state)
    # print(len(next_state))                          # 
    # print(len(reward))
    # print(action[0])                                # {0, 0, 0} x 30000
    # print(len(terminal))
    

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    # print(X_train.shape)            # (27000, 96, 96, 3)
    # print(y_train.shape)            # (27000, 3)
    # print(X_valid.shape)            # (3000, 96, 96, 3)
    # print(y_valid.shape)            # (3000, 3)
    
    X_train = rgb2gray(X_train)
    y_train = np.array(list(map(action_to_id, y_train)))
    X_valid = rgb2gray(X_valid)
    y_valid = np.array(list(map(action_to_id, y_valid)))
    
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, number_of_epochs, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    agent = BCAgent(learning_rate=lr)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[-1], X_train.shape[-1])
    X_valid = X_valid.reshape(X_valid.shape[0], 1, X_valid.shape[-1], X_valid.shape[-1])

    tensorboard_eval = Evaluation(
        tensorboard_dir, 
        name='example', 
        stats=[ 
        "total_train_loss", 
        "total_val_loss"
        ]
        )

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    
    training_data = MyDataset(X_train, y_train)
    val_data = MyDataset(X_valid, y_valid)
    train_data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    # calculate steps per epoch for training and validation set
    train_steps = len(train_data_loader.dataset) // batch_size
    val_steps = len(val_data_loader.dataset) // batch_size
    
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")

    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    # training loop
    for epoch in range(number_of_epochs):
        agent.net.train()

        total_train_loss = 0
        total_val_loss = 0

        for i, (train_features, train_labels) in enumerate(train_data_loader):
            
            batch_train_loss = agent.update(train_features, train_labels)
            total_train_loss += batch_train_loss

            # if i % 10 == 0:
            #     # compute training/ validation accuracy and write it to tensorboard
                # result_dict = {"batch_train_loss": float(batch_train_loss)}
                # tensorboard_eval.write_episode_data(i , result_dict)
        
        #tensorboard_eval.write_episode_data(epoch, {"total_train_loss": float(total_train_loss)})
        
        # train_losses.append(total_train_loss)

        # validation part
        with torch.no_grad():
            agent.net.eval()

            for val_features, val_labels in val_data_loader:
                val_output = agent.predict(val_features)
                total_val_loss += agent.loss(val_output, val_labels).item()
                
                # print(total_val_loss)
            
        # tensorboard_eval.write_episode_data(epoch, {"total_val_loss": float(total_val_loss)})
        
        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps

        tensorboard_eval.write_episode_data(epoch, {"total_train_loss": total_train_loss})
        tensorboard_eval.write_episode_data(epoch, {"total_val_loss": total_val_loss})

        # update our training history
        H["train_loss"].append(avg_train_loss)
        H["val_loss"].append(avg_val_loss)
        
        # TODO: save your agent
        # print(os.path.join(model_dir, "agent.pt"))
        # agent.save(os.path.join(model_dir, "agent.pt"))
        print('Epoch : ',epoch+1, '\t', 'train_loss :', avg_train_loss, '\t', 'val_loss : ', avg_val_loss)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.show()


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.y = y


    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, number_of_epochs=2, batch_size=64, lr=1e-4)
 
