import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os

import resnet
from loader import get_data_loader



if __name__ == '__main__':
    train_loader = get_data_loader()
    model = torchvision.models.resnet50(pretrained=False)
    device = 'cuda'
    learning_rate = 0.1
    num_epochs = 3
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model...

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}... {len(train_loader)} batches total')
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            #print(np.shape(inputs))
            #print(np.shape(labels))
            #print(labels)
            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % 100 == 0:
                print(f'  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # Print the loss for every epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    print(f'Finished Training, Loss: {loss.item():.4f}')
