import argparse
import sys

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        # define criterion
        criterion = nn.NLLLoss()
        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        # Bookkeeping for loss and steps
        steps = 0
        loss_list = []
        # define nr of epochs
        epochs = 5
        # loop over epochs
        for e in range(epochs):
            # loop over batches
            running_loss = 0
            for images, labels in train_set:
                steps += 1
                # flatten images
                images = images.view(images.shape[0], -1)
                # set optimizer gradient to 0
                optimizer.zero_grad()
                # Evaluate model: forward pass
                output = model(images)
                # calculate loss
                loss = criterion(output, labels)
                # Calculate gradients
                loss.backward()
                # update the weights
                optimizer.step()
                # calculate loss
                running_loss += loss.item()
            else:
                print(f"The current training loss is: {running_loss / len(train_set)}")
            loss_list.append(running_loss)
        # saving the model
        torch.save(model, 'checkpoint.pth')
        # plotting the learning curve
        plt.plot(np.arange(epochs), np.array(loss_list))
        plt.title('Learning Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Training loss')
        plt.show()





        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, test_set = mnist()
        accuracy = 0
        for images, labels in test_set:
            images = images.view(images.shape[0], -1)
            output = model(images)
            ps = torch.exp(output)
            equal = (labels.data == ps.max(1)[1])
            accuracy = equal.type_as(torch.FloatTensor()).mean()
            print(f'Model accuracy on test set is {accuracy*100}%')


if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    