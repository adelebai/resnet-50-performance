import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import time

import resnet_adjusted
import revnet
from loader import get_data_loader

import config


def print_mem_stats():
    # note - not using torch.cuda.mem_get_info() since it includes the overhead memory
    # mems = torch.cuda.mem_get_info()
    # the below measurements seem more reasonable for cuda mem during training.
    alloc = torch.cuda.memory_allocated(0)/1024/1024/1024
    reserved = torch.cuda.memory_reserved(0)/1024/1024/1024
    max_reserved = torch.cuda.max_memory_reserved(0)/1024/1024/1024
    print(f"memory_allocated: {alloc}GB memory_reserved: {reserved}GB max_memory_reserved: {max_reserved}GB")


if __name__ == '__main__':
    train_loader = get_data_loader()
    model = revnet.revnet3_3_3()
    #model = torchvision.models.resnet18(pretrained=False)
    device = 'cuda'
    learning_rate = 0.1
    num_epochs = 1
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train the model...
    # Add profiler to print profiler stats after training. Note - profiler overhead is large.
    # If you are runnning it, reduce the batches to like 10 or less.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/revnet-3-3-3')) as prof:
        for epoch in range(config.num_epochs):
            print(f'Starting epoch {epoch+1}... {len(train_loader)} batches total')
            for batch_idx, (inputs, labels) in enumerate(train_loader):

                # Move input and label tensors to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero out the optimizer
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # free activations, clip gradients
                # model.free()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

                if (batch_idx+1) % 100 == 0:
                    print_mem_stats()

                optimizer.step()

                if (batch_idx+1) > 10:
                    print_mem_stats()
                    break

        # Print the loss for every epoch
        print(f'Finished epoch {epoch+1}/{config.num_epochs}, Loss: {loss.item():.4f}')

    print(f'Finished Training, Loss: {loss.item():.4f}')
    
    #print(torch.cuda.memory_stats(device=device))
    #print profiler results
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))