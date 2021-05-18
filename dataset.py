import os
import pandas as pd
from utils import get_mean_std, plot_images
from skimage import io
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

RANDOM_STATE= 42
BATCH_SIZE = 32
IMG_RESIZE = (224,224)
torch.manual_seed(RANDOM_STATE)

train_file = 'data/one-indexed-files-notrash_train.txt'
valid_file = 'data/one-indexed-files-notrash_val.txt'
test_file = 'data/one-indexed-files-notrash_test.txt'
img_dir = 'data/image_dir'

label_map = {
    0:'glass',
    1:'paper',
    2:'cardboard',
    3:'plastic',
    4:'metal',
    5:'trash'
}

class GarbageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, label_map=None):
        self.img_labels = pd.read_csv(annotations_file, header=None, sep=" ")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_map = label_map

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0].split('.')[0].rstrip('0123456789'), self.img_labels.iloc[idx, 0])
        image = io.imread(img_path)
        label = self.img_labels.iloc[idx, 1] - 1
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return (image, label)

    def class_distribution(self):
        return (self.img_labels.iloc[:,1]-1).map(label_map).value_counts(normalize=True).to_dict()

    @property
    def targets(self):
        return torch.Tensor(self.img_labels.iloc[:, 1] - 1)

# Get train dataset mean and std
dataset = GarbageDataset(train_file, img_dir, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(IMG_RESIZE)]))

train = DataLoader(dataset, batch_size=BATCH_SIZE)

mean, std = get_mean_std(train)

# Define image transformations
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_RESIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(contrast=0.25,saturation=2),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])
valid_test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_RESIZE),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# Train, Validation and Test datasets
train_dataset = GarbageDataset(train_file, img_dir, transform=train_transforms)
valid_dataset = GarbageDataset(valid_file, img_dir, transform=valid_test_transforms)
test_dataset = GarbageDataset(test_file, img_dir, transform=valid_test_transforms)


# plot_images(train_dataset, 4,4, label_map)

class_distribution = train_dataset.class_distribution()

class_weights = list((train_dataset.img_labels.iloc[:, 1] - 1).map(label_map).map(class_distribution))

# Create dataloaders

# As there is class imbalance - WeightedRandomSampler should help to ensure each mini batch represents all classes

weighted_sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=False)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          sampler=weighted_sampler
)
val_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False
)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False
)






