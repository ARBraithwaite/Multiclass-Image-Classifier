import torch
from model.network import *
from evaluate import *
from dataset import train_loader, val_loader
from utils import *

accuracy_stats = {
    'train': [],
}
loss_stats = {
    'train': [],
}

model.to(DEVICE)

def train(model, train, test, criterion, optimizer, n_epochs):
    for epoch in range(n_epochs):
        print(f'Epoch number - {epoch}')
        running_loss = 0
        running_accuracy = 0
        total = 0
        # Training the model
        model.train()
        for images, labels in train:
            # Move to device
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            total += labels.size(0)
            # Clear optimizers
            optimizer.zero_grad()
            # Forward pass
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            # Loss
            loss = criterion(output, labels)
            # Calculate gradients (backpropogation)
            loss.backward()
            # Adjust parameters based on gradients
            optimizer.step()
            # Add the loss to the training set's rnning loss
            running_loss += loss.item()
            running_accuracy += (labels==predicted).sum().item()
        epoch_loss = running_loss / len(train)
        epoch_acc = (running_accuracy / total)
        accuracy_stats['train'].append(epoch_acc)
        loss_stats['train'].append(epoch_loss)
        print(f'Model - Training Dataset got {running_accuracy} out of {total} images correctly classified {epoch_acc * 100.00:.2f}%. Epoch loss: {epoch_loss}')

        _, val_acc, loss_acc = model_evaluate(model, test, 'validation')

    print('Training Finished')
        
    return model, accuracy_stats['train'], loss_stats['train'], val_acc, loss_acc

trained_model, train_acc, train_loss, val_acc, val_loss = train(model, train_loader, val_loader, criterion, optimizer, hyper_param['n_epochs'])

train_val_plot(train_acc, train_loss, val_acc, val_loss)

torch.save(trained_model, PATH)

