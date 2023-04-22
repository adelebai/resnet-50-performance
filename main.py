import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import time

import resnet
from loader import get_data_loader

import config



if __name__ == '__main__':
    train_loader = get_data_loader()
    model = torchvision.models.resnet50(pretrained=False)
    model = model.to(config.device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train the model...

    time_checkpoint = time.perf_counter()
    time_dataload_total = 0
    time_training_total = 0
    time_itcount = 0
    for epoch in range(config.num_epochs):
        print(f'Starting epoch {epoch+1}... {len(train_loader)} batches total')
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            time_itcount += 1
            time_checkpoint_current = time.perf_counter()
            time_dataload_total += time_checkpoint_current-time_checkpoint

            # Move input and label tensors to the device
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            time_checkpoint = time.perf_counter()
            time_training_total += time_checkpoint - time_checkpoint_current

            if (batch_idx+1) % config.print_every_n_batches == 0:
                print(f'  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
                print(f'    Avg batch data load: {time_dataload_total/time_itcount:.2f}, Avg batch training: {time_training_total/time_itcount:.2f}')

        # Print the loss for every epoch
        print(f'Finished epoch {epoch+1}/{config.num_epochs}, Loss: {loss.item():.4f}')
        print(f'  Avg epoch data load: {time_dataload_total/time_itcount:.2f}, Avg epoch training: {time_training_total/time_itcount:.2f}')
        print(f'  Total epoch data load: {time_dataload_total:.2f}, Total epoch training: {time_training_total:.2f}')
        print()
        time_checkpoint = time.perf_counter()
        time_dataload_total = 0
        time_training_total = 0
        time_itcount = 0

    print(f'Finished Training, Loss: {loss.item():.4f}')
