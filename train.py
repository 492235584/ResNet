import torch, copy
import torchvision as tv
import torchvision.transforms as transforms
from torch import optim
import torch.nn as nn
import ResNet
import argparse
import os
from torchsummary import summary

input_size = 224
epoch = 30
learning_rate = 0.01
batch_szie = 128


def load_data():
    '''
    get data loader
    :return: data loader
    '''
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(input_size, scale=(0.8, 1)),
            transforms.ToTensor()
        ]),
        'validate': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()]
        )
    }

    train_dataset = tv.datasets.CIFAR10(
        root='./data/', train=True, download=True, transform=data_transforms['train'])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_szie, shuffle=True, num_workers=4
    )

    test_dataset = tv.datasets.CIFAR10(
        root='./data/', train=False, download=True, transform=data_transforms['validate'])
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_szie, shuffle=False,
    )
    return {'train': train_dataloader, 'validate': test_dataloader}


def train(optimizer, model, dataloaders, criterion, lr_scheduler, max_epoch=20):
    '''
    train model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model_weights = None
    best_acc = 0.0

    epoch_list = []
    acc_list = []
    val_acc_list = []
    print('start train')
    for epoch in range(1, max_epoch + 1):
        epoch_list.append(epoch)
        lr_scheduler.step()
        print('------epoch : %d  lr:%.5f------' % (epoch, optimizer.param_groups[0]['lr']))
        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loos = 0.0
            epoch_acc = 0.0

            # show progress
            for step, (X, Y) in enumerate(dataloaders[phase]):
                # if (step - 1) % 20 == 0:
                #     print('step %d' % step)
                X = X.to(device)
                Y = Y.to(device)
                # compute output
                optimizer.zero_grad()
                Y_hat = model(X)
                loss = criterion(Y_hat, Y)
                _, preds = torch.max(Y_hat, 1)
                # backprop
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # get loss and accuracy
                epoch_loos += loss.item() * X.size(0)
                epoch_acc += torch.sum(preds == Y.data)

            epoch_loos /= len(dataloaders[phase].dataset)
            epoch_acc = epoch_acc.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                acc_list.append(epoch_acc)
            else:
                val_acc_list.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loos, epoch_acc))
            # save best model
            if phase == 'validate' and epoch_acc > best_acc:
                best_model_weights = copy.deepcopy(model.state_dict())
                best_acc = epoch_acc
            # save history model
            torch.save(model, 'checkpoint/epoch_%s.pth' % epoch)

    # save best model
    model.load_state_dict(best_model_weights)
    torch.save(model, 'checkpoint/best.pth')
    return model

def inference(model, dataloaders):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epoch_acc = 0.0
    # show progress
    for step, (X, Y) in enumerate(dataloaders['validate']):
        # if (step - 1) % 20 == 0:
        #     print('step %d' % step)
        X = X.to(device)
        Y = Y.to(device)
        # compute output
        optimizer.zero_grad()
        Y_hat = model(X)
        loss = criterion(Y_hat, Y)
        _, preds = torch.max(Y_hat, 1)
        # backprop
        # get loss and accuracy
        epoch_acc += torch.sum(preds == Y.data)

    epoch_acc = epoch_acc.double() / len(dataloaders['validate'].dataset)
    print('test Acc: {:.4f}'.format(epoch_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='use (--mode train) if you want train model')
    parser.add_argument('--model', default='checkpoint/best.pth', help='choose your model path when you test')
    opt = parser.parse_args()

    model = ResNet.ResNet50()
    summary(model, (3, 224, 224))
    print(model.state_dict().keys())

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)
    criterion = nn.CrossEntropyLoss()
    data_loaders = load_data()

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if opt.mode == 'train':
        train(optimizer, model, data_loaders, criterion, lr_scheduler, max_epoch=epoch)
    else:
        model = torch.load(opt.model)
        inference(model, data_loaders)