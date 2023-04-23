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

import resnet
import revnet
from loader import get_data_loader


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
    # model = torchvision.models.resnet50(pretrained=False)
    # is_revnet = False
    # TODO - this is just a temporary model, need to adjust it to "match" resnet50
    model = revnet.revnet38()
    is_revnet = True
    device = 'cuda'
    learning_rate = 0.1
    num_epochs = 1
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model...
    # Add profiler to print profiler stats after training. Note - profiler overhead is large.
    # If you are runnning it, reduce the batches to like 10 or less.
    # since this is slow, run ~ 300 batches only.
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     profile_memory=True) as prof:
    for epoch in range(num_epochs):
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
            optimizer.step()

            # free activations, clip gradients
            if is_revnet:
                model.free()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            if (batch_idx+1) % 100 == 0:
                print_mem_stats()
            if (batch_idx+1) % 100 == 0:
                print(f'  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

            if (batch_idx+1) > 300:
                break

        # Print the loss for every epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    print(f'Finished Training, Loss: {loss.item():.4f}')
    
    #print(torch.cuda.memory_stats(device=device))
    # print profiler results
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))