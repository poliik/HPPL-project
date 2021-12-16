import torch
import torchvision

import torchvision.transforms as transforms

from Shampoo_parallel import Shampoo

import multiprocessing as mp

from time import time

def precond_grad(rank, q_data, q_precond):
    while True:
        if q_data.empty() is False:
            grad, precond, order, dim_id = q_data.get()

            dim = grad.size()[dim_id]
            grad = grad.transpose_(0, dim_id).contiguous()
            grad = grad.view(dim, -1)

            grad_t = grad.t()
            precond += grad @ grad_t
            inv_precond = _matrix_power(precond, -1 / order)
            q_precond.put((dim_id, (precond, inv_precond)))

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    # use CPU for svd for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)


def train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = torch.nn.CrossEntropyLoss()
    net = torchvision.models.resnet34()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    optimizer = Shampoo(net.parameters(), lr=0.5, momentum=0.9)
    
    q_data = mp.Queue()
    q_precond = mp.Queue()
    
#     print('Starting spawn...')
#     mp.spawn(fn=precond_grad, nprocs=4, args=(q_data, q_result))

    processes = []
    for rank in range(10):
        p = mp.Process(target=precond_grad, args=(rank, q_data, q_precond))
        processes.append(p)
    
    [x.start() for x in processes]
    
    print('Starting training...')
    
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        # with tqdm(total=len(trainloader.dataset)) as pb:
        for i, data in enumerate(trainloader, 0):
              # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            start = time()
            optimizer.step(q_data, q_precond)
            end = time()
            print('i:', i, 'time:', end-start)

            # print statistics
            running_loss += loss.item() * len(labels)
            print('i:', i, 'loss;', loss.item())

              # pb.update(len(labels))
        print(running_loss / len(trainloader.dataset))


    print('Finished Training')

if __name__ == '__main__':
    print('Entering main function...')
    train()
    exit(0)
