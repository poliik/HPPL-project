import torch
import torchvision

import torchvision.transforms as transforms

from Shampoo_distributed import Shampoo


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
            optimizer.step()

            # print statistics
            running_loss += loss.item() * len(labels)
            print(loss.item())

              # pb.update(len(labels))
        print(running_loss / len(trainloader.dataset))


    print('Finished Training')

if __name__ == '__main__':
    print('Entering main function...')
    train()
    exit(0)