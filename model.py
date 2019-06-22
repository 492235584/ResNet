import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.feat = nn.Sequential(
            nn.Conv2d(3, 96, 7, 4, 2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, 1, 2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classfier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048), nn.ReLU(),
            nn.Linear(2048, 10)
        )


    def forward(self, X):
        X = self.feat(X)
        X = X.view(X.size()[0], -1)
        X = self.classfier(X)
        return X
