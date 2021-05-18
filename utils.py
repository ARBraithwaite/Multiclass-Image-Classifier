import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import pandas as pd
import dataframe_image

def plot_images(dataset, cols:int = 3, rows:int = 3, label_map: dict=None):
    torch.seed()
    figure = plt.figure(figsize=(8, 8))
    cols, rows = cols, rows
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(1, len(dataset), (1,)).item()
        img, label = dataset[sample_idx]
        if label_map is not None:
            label = label_map[label]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()
    return None


def get_mean_std(loader):
    mean = 0.
    var = 0.
    batch_samples = 0
    for data, _ in loader:
        data = data.float()
        batch = data.size(0)
        data = data.view(batch, data.size(1), -1)
        mean += data.mean(2).sum(0)
        var += data.var(2).sum(0)
        batch_samples += batch
    mean /= batch_samples
    var /= batch_samples
    std = torch.sqrt(var)

    return mean, std

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.YlGnBu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('experiments/confusion_matrix.png')
    plt.show()
    return None

def plot_incorrect_classified(dataset, targets, predictions, label_map=None):
    mis_class = []
    torch.seed()
    predictions = predictions.argmax(dim=1).cpu().int()
    for i, val in enumerate((targets == predictions)):
        if val == 0:
            mis_class.append(i)
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 4
    for i in range(1, cols * rows + 1):
        sample_idx = mis_class[torch.randint(1, len(mis_class), (1,)).item()]
        img, label = dataset[sample_idx]
        pred_label = predictions.numpy()[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f'Actual: {label_map[label]}, Predicted: {label_map[pred_label]}')
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()

    return None

def classification_report_(targets, predictions, labels):
    return classification_report(targets, predictions, target_names = labels)

def train_val_plot(train_acc, train_loss, val_acc, val_loss):

    train_acc = pd.DataFrame(train_acc, columns=['Train Accuracy'])
    val_acc = pd.DataFrame(val_acc, columns=['Val Accuracy'])
    train_loss = pd.DataFrame(train_loss, columns=['Train Loss'])
    val_loss = pd.DataFrame(val_loss, columns=['Val Loss'])

    fig, axes = plt.subplots(1, 2, figsize=(15,8))
    train_acc.plot(y='Train Accuracy',xlabel='Epochs', ylabel='Accuracy', title='Train-Val Accuracy/Epoch', ax=axes[0])
    val_acc.plot(y='Val Accuracy', ax=axes[0])
    train_loss.plot(y='Train Loss',xlabel='Epochs',  ylabel='Loss', title='Train-Val Loss/Epoch', ax=axes[1])
    val_loss.plot(y='Val Loss', ax=axes[1])
    plt.savefig('experiments/train_val_acc_loss.png')
    plt.show()


def plot_image_tensor(image, label):
    plt.imshow(image)
    plt.title(f'{label}')
    plt.axis('off')
    plt.show()
