import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data._utils.worker

import os

from get_dataset import DATA_DIR
from loader_worker import custom_worker
import config

def get_data_loader(train=True):

    t = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if train:
        data = torchvision.datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train.comb'), transform=t['train'])
    else:
        data = torchvision.datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val.X'), transform=t['test'])

    # monkey patch our custom, parallelized worker loop
    if config.use_threaded_loader_worker:
        torch.utils.data._utils.worker._worker_loop = custom_worker

    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_dataloader_workers)
    return data_loader
