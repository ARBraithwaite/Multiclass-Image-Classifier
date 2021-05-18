import torch
from model.network import *
from evaluate import *
from dataset import test_loader, test_dataset, label_map, valid_test_transforms
from utils import *

#Load model:
model = torch.load(PATH)

predictions, _, _ = model_evaluate(model, test_loader, name='Test')
predictions_ = predictions.argmax(dim=1).cpu()


#Plot incorrectly classified examples
plot_incorrect_classified(test_dataset, test_dataset.targets, predictions, label_map)


# Plot confusion matrix/report
classes = [names for names in label_map.values()]

with open("experiments/classification_report.txt", "w") as report:
    report.write(classification_report_(test_dataset.targets, predictions_, classes))

cm = confusion_matrix(test_dataset.targets, predictions_)
plot_confusion_matrix(cm, classes)

predict_image(model, valid_test_transforms, 'data/[].jpg', classes)

