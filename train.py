import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from model import MelSpectrogramCNN, CovidNet

ROOT_PATH = './dataset_baseline'
    
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Load the dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset using ImageFolder
train_dir = os.path.join(ROOT_PATH, 'train')
val_dir = os.path.join(ROOT_PATH, 'val')
test_dir = os.path.join(ROOT_PATH, 'test')

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
print("Train dataset:", len(train_dataset))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
print("Val dataset:", len(val_dataset))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
print("Test dataset:", len(test_dataset))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

covidNet = MelSpectrogramCNN() # Could be change to different network

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(covidNet.parameters(), lr=LEARNING_RATE)

def saveModel():
    path = "./covidNet.pth"
    torch.save(covidNet.state_dict(), path)

def valAccuracy():
    covidNet.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = covidNet(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)

def train():
    best_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            images, labels = data
            optimizer.zero_grad()
            outputs = covidNet(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_dataloader)))
        accuracy = valAccuracy()
        print('For epoch', epoch+1,'the validation accuracy is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

def test():
    covidNet.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = covidNet(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    print(f'Test Accuracy: {accuracy:.2f}%')
    return(accuracy)

train()
test()