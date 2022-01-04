import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import numpy as np
import os


def mnist():

    # Define data path
    path = os.path.join('..', '..', '..', 'data', 'corruptmnist')

    # Load training data
    train_img = np.load(os.path.join(path, "train_0.npz"))['images']
    train_label = np.load(os.path.join(path, "train_0.npz"))['labels']
    filenames = ["train_1.npz", "train_2.npz", "train_3.npz", "train_4.npz"]
    for i, fn in enumerate(filenames):
        train_img = np.concatenate((train_img, np.load(os.path.join(path, fn))['images']))
        train_label = np.concatenate((train_label, np.load(os.path.join(path, fn))['labels']))

    # load testing data
    test_img = np.load(os.path.join(path, "test.npz"))['images']
    test_label = np.load(os.path.join(path, "test.npz"))['labels']

    # Convert the data into tensors
    train_img = torch.Tensor(train_img)
    train_label = torch.Tensor(train_label).type(torch.LongTensor)
    test_img = torch.Tensor(test_img)
    test_label = torch.Tensor(test_label).type(torch.LongTensor)

    # transform into dataset
    train_set = TensorDataset(train_img, train_label)
    test_set = TensorDataset(test_img, test_label)

    # transform into data loader
    train = DataLoader(train_set, batch_size=64, shuffle=True)
    test = DataLoader(test_set, batch_size=test_label.shape[0], shuffle=False)

    return train, test
#%%
# train_set, _ = mnist()
# for images, labels in train_set:
#     print(type(labels))