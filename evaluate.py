import torch
from model.network import *
from skimage import io
from utils import *

accuracy_stats = {
    'val': [],
}
loss_stats = {
    'val': [],
}

def model_evaluate(model, loader, name='Test/Valid'):
    model.eval()
    running_accuracy = 0
    total = 0
    running_loss = 0
    all_preds = torch.tensor([]).to(DEVICE)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            total += labels.size(0)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, labels)
            running_accuracy += (labels==predicted).sum().item()
            running_loss += loss.item()
            all_preds = torch.cat((all_preds, output),dim=0)
    epoch_loss = running_loss / len(loader)
    epoch_acc = (running_accuracy / total)
    accuracy_stats['val'].append(epoch_acc)
    loss_stats['val'].append(epoch_loss)
    print(f'Model - {name} Dataset got {running_accuracy} out of {total} images correctly classified {epoch_acc * 100:.2f}%. Epoch loss: {epoch_loss}')
    return all_preds, accuracy_stats['val'], loss_stats['val']

def predict_image(model, transformation, path, labels):
    model = model.eval()
    image_ = io.imread(path)
    image = transformation(image_).float()
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    output = model(image)
    _, predicted = torch.max(output.data, 1)
    plot_image_tensor(image_, labels[predicted.item()])
    return labels[predicted.item()]





