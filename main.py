import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import time

import resnet
import resnet_adjusted
import revnet
from loader import get_data_loader

import config


if __name__ == '__main__':
    train_loader = get_data_loader()
    use_revnet = config.use_revnet
    if use_revnet:
        # Chosen this configuration to match the number of params in resnet18 adjusted
        # which has around 746k parameters.
        # revnet 5-5-5 has around the same (790k)
        model = revnet.revnet_custom([5,5,5])
    else:
        model = resnet_adjusted.ResNet18()
        #model = torchvision.models.resnet18(pretrained=False)

    device = 'cuda'
    learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    model = model.to(config.device)
    momentum = config.momentum
    decay = config.decay

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=momentum, weight_decay=decay)

    # Train the model...
    # Add profiler to print profiler stats after training. Note - profiler overhead is large.
    # If you are runnning it, reduce the batches to like 10 or less.
    # since this is slow, run ~ 300 batches only.
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

            # free activations, clip gradients
            if use_revnet:
                model.free()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            optimizer.step()

            time_checkpoint = time.perf_counter()
            time_training_total += time_checkpoint - time_checkpoint_current

            if (batch_idx+1) % config.print_every_n_batches == 0:
                print(f'  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
                print(f'    Avg batch data load: {time_dataload_total/time_itcount:.2f}, Avg batch training: {time_training_total/time_itcount:.2f}')

            # if (batch_idx+1) > 300:
            #     break

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
    