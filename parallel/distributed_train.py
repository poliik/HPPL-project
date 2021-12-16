
import os
import argparse
from argparse import Namespace

import numpy as np

import copy
# from datasets import load_dataset

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


import torchvision
import torchvision.transforms as transforms
       
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpu per node')
    parser.add_argument('-ep', '--epochs', default=1, type=int, help='number of total epoch to run')
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    
    args = parser.parse_args()
    args.world_size = args.gpus
    mp.spawn(train, nprocs=args.gpus, args=(args,)) #create a prosesses one one node

def train(gpu, args):
    node_rank = 0
    
    rank = node_rank * args.gpus + gpu
    
    print(f'rank is {rank}. Preparing to configure environment')
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', #protocol - how to communicate
                            rank=rank, 
                            world_size=args.world_size)
    
    print('Init success')
    
    torch.manual_seed(0)
    device = torch.device(f'cuda:{gpu}')
    

    criterion = torch.nn.CrossEntropyLoss()
    # Dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    eval_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # model
    model = torchvision.models.resnet34()
    model.to(device)
    
    ddp_model = DDP(model, device_ids=[gpu]) #wrap the model for further gradient communication
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9)
    
#     max_train_steps = 100
#     max_train_epochs = 2
    
    # different batches on different GPU
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    
    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
 
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=False,
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        sampler=eval_sampler,
        pin_memory=True
    )
    
#     completed_steps = 0
    
    N_EPOCHS = 20
    print('Start training')
    for epoch in range(N_EPOCHS):
        train_dataloader.sampler.set_epoch(epoch) 
        ddp_model.train()
        losses = []
        for step, (input, labels) in enumerate(train_dataloader):
            input = input.to(device)
            labels = labels.to(device)
            
            # batch['input_ids'] = batch['input_ids'].to(device)    
            # batch['labels'] = batch['labels'].to(device)
            # input, labels = batch[0].to(device), batch[1].to(device)
            
            if rank == 0 and step % 20 == 0:
                print(step, '/', len(train_dataloader))
                
            optimizer.zero_grad()
            # outputs = ddp_model(input_ids=batch['input_ids'], labels=batch['labels'])
            outputs = ddp_model(input)
            
            # print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
                
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()

        if rank == 0:
            loss_mean = np.mean(losses)
            print(f'Epoch {epoch}/{N_EPOCHS}\tloss: {loss_mean}')
            
    
    # print('Start evaluating')
    # ddp_model.eval()
    # losses = []
    
    # with torch.no_grad():
    #     for step, batch in enumerate(eval_dataloader):
    #         if step == 10:
    #             break

    #         batch['input_ids'] = batch['input_ids'].to(device)
    #         batch['labels'] = batch['labels'].to(device)

    #         if rank == 0 and step % 20 == 0:
    #             print(step, '/', len(train_dataloader))

    #         optimizer.zero_grad()
    #         outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])

    #         loss = criterion(outputs, labels)
    #         losses.append(loss.detach().cpu().numpy())

    # loss_mean = np.mean(losses)
    # if rank == 0:
    #     print(f'Epoch {epoch}/{N_EPOCHS}\tloss: {loss_mean}')
            
    
    dist.destroy_process_group()
    
if __name__ == '__main__':
    print('Entering main function...')
    main()
    exit(0)
