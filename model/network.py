import torchvision.models as models
import torch.nn as nn
import torch

PATH = 'experiments/restnet18.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 6

hyper_param = {
    'lr':0.001,
    'momentum':0.9,
    'weight_decay':0.001,
    'n_epochs':20
}

model = models.resnet18(pretrained=True)

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=hyper_param['lr'], momentum=hyper_param['momentum'],weight_decay=hyper_param['weight_decay'])
