import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from model import MelSpectrogramCNN, CovidNet
from sklearn.metrics import classification_report, precision_recall_fscore_support

ROOT_PATH = './dataset_baseline_chunk'

# 1st time 
# BATCH_SIZE = 32
# LEARNING_RATE = 0.00005
# NUM_EPOCHS = 10

# 2nd time: lower the learning rate
# BATCH_SIZE = 32
# LEARNING_RATE = 0.00001
# NUM_EPOCHS = 10

# 3rd time
BATCH_SIZE = 32
LEARNING_RATE = 0.000001
NUM_EPOCHS = 10

# Load the dataset
transform = transforms.Compose([
    transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset using ImageFolder
train_dir = os.path.join(ROOT_PATH, 'train')
val_dir = os.path.join(ROOT_PATH, 'val') 
test_dir = os.path.join(ROOT_PATH, 'test')
test_male_dir = os.path.join(ROOT_PATH, 'test-male')
test_female_dir = os.path.join(ROOT_PATH, 'test-female')

test_under20_dir = os.path.join(ROOT_PATH, 'test-under20')
test_20to40_dir = os.path.join(ROOT_PATH, 'test-20to40')
test_40to60_dir = os.path.join(ROOT_PATH, 'test-40to60')
test_over60_dir = os.path.join(ROOT_PATH, 'test-over60')

test_under40_dir = os.path.join(ROOT_PATH, 'test-under40')
test_over40_dir = os.path.join(ROOT_PATH, 'test-over40')

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
print("Train dataset:", len(train_dataset))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
print("Val dataset:", len(val_dataset))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
print("Test dataset:", len(test_dataset))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_male_dataset = datasets.ImageFolder(root=test_male_dir, transform=transform)
print("Test (Male) dataset:", len(test_male_dataset))
test_male_dataloader = torch.utils.data.DataLoader(test_male_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_female_dataset = datasets.ImageFolder(root=test_female_dir, transform=transform)
print("Test (Female) dataset:", len(test_female_dataset))
test_female_dataloader = torch.utils.data.DataLoader(test_female_dataset, batch_size=BATCH_SIZE, shuffle=False)

# test_under20_dataset = datasets.ImageFolder(root=test_under20_dir, transform=transform)
# print("Test (<=20) dataset:", len(test_under20_dataset))
# test_under20_dataloader = torch.utils.data.DataLoader(test_under20_dataset, batch_size=BATCH_SIZE, shuffle=False)

# test_20to40_dataset = datasets.ImageFolder(root=test_20to40_dir, transform=transform)
# print("Test (20-40) dataset:", len(test_20to40_dataset))
# test_20to40_dataloader = torch.utils.data.DataLoader(test_20to40_dataset, batch_size=BATCH_SIZE, shuffle=False)

# test_40to60_dataset = datasets.ImageFolder(root=test_40to60_dir, transform=transform)
# print("Test (40-60) dataset:", len(test_40to60_dataset))
# test_40to60_dataloader = torch.utils.data.DataLoader(test_40to60_dataset, batch_size=BATCH_SIZE, shuffle=False)

# test_over60_dataset = datasets.ImageFolder(root=test_over60_dir, transform=transform)
# print("Test (>60) dataset:", len(test_over60_dataset))
# test_over60_dataloader = torch.utils.data.DataLoader(test_over60_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_under40_dataset = datasets.ImageFolder(root=test_under40_dir, transform=transform)
print("Test under40 dataset:", len(test_under40_dataset))
test_under40_dataloader = torch.utils.data.DataLoader(test_under40_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_over40_dataset = datasets.ImageFolder(root=test_over40_dir, transform=transform)
print("Test (>60) dataset:", len(test_over40_dataset))
test_over40_dataloader = torch.utils.data.DataLoader(test_over40_dataset, batch_size=BATCH_SIZE, shuffle=False)

# covidNet = MelSpectrogramCNN() # Could be changed to different network

covidNet = MelSpectrogramCNN()
covidNet.load_state_dict(torch.load("./covidNet_1st.pth"))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(covidNet.parameters(), lr=LEARNING_RATE)

def saveModel():
    path = "./covidNet_mul.pth"
    torch.save(covidNet.state_dict(), path)

def valAccuracy():
    covidNet.eval()
    accuracy = 0.0
    total = 0.0
    labelsList = []
    predsList = []
    
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            labelsList += labels.tolist()
            # run the model on the test set to predict labels
            outputs = covidNet(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            predsList += predicted.tolist()
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    print(classification_report(labelsList, predsList))
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)

def train():
    best_accuracy = 37.0
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
            if i % 10 == 0:
                print('Batch: {} \tTraining Loss: {:.6f}'.format(i, loss.item()))
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_dataloader)))
        accuracy = valAccuracy()
        print('For epoch', epoch+1,'the validation accuracy is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

def test_tool(model, dataloader, groupLabel):
    accuracy = 0.0
    total = 0.0
    labelsList = []
    predsList = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            labelsList += labels.tolist()
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            predsList += predicted.tolist()
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    print('Evaluation report on ', groupLabel)
    precision, recall, f1, support = precision_recall_fscore_support(labelsList, predsList)
    accuracy = (100 * accuracy / total)
    print(f'Test Accuracy: {accuracy:.2f}%')
    print('Test Precision:', precision)
    print('Test Recall:', recall)
    print('Test F1-score:', f1)
    print(classification_report(labelsList, predsList))
    return labelsList, predsList

def stat_parity(preds, sens):
    '''
    :preds: numpy array of the model predictions. Consisting of 0s and 1s
    :sens: numpy array of the sensitive features. Consisting of 0s and 1s
    :return: the statistical parity. no need to take the absolute value
    '''
    P_Y_A0 = 0
    P_A0 = 0
    P_Y_A1 = 0
    P_A1 = 0
    for i in range(len(preds)):
        if sens[i] == 0:
            P_A0 += 1
            if preds[i] == 1:
                P_Y_A0 += 1
        else:
            P_A1 += 1
            if preds[i] == 1:
                P_Y_A1 += 1
    first = 0
    if P_A0 != 0:
        first = P_Y_A0/P_A0
    second = 0
    if P_A1 != 0:
        second = P_Y_A1/P_A1
    return (first - second)
#     return 0


def eq_oppo(preds, sens, labels):
    '''
    :preds: numpy array of the model predictions. Consisting of 0s and 1s
    :sens: numpy array of the sensitive features. Consisting of 0s and 1s
    :labels: numpy array of the ground truth labels of the outcome. Consisting of 0s and 1s
    :return: the statistical parity. no need to take the absolute value
    '''
    P_Y_A0_Y1 = 0
    P_A0_Y1 = 0
    P_Y_A1_Y1 = 0
    P_A1_Y1 = 0
    for i in range(len(preds)):
        if sens[i] == 0:
            if labels[i] == 1:
                P_A0_Y1 += 1
                if preds[i] == 1:
                    P_Y_A0_Y1 += 1
        else:
            if labels[i] == 1:
                P_A1_Y1 += 1
                if preds[i] == 1:
                    P_Y_A1_Y1 += 1
    first = 0
    if P_A0_Y1 != 0:
        first = P_Y_A0_Y1/P_A0_Y1
    second = 0
    if P_A1_Y1 != 0:
        second = P_Y_A1_Y1/P_A1_Y1
    return (first - second)

def test():
    covidNet.eval()
    # test_tool(covidNet, test_dataloader, 'the whole test set')
    # test_tool(covidNet, val_dataloader, 'the whole validation set')
    # test_tool(covidNet, test_male_dataloader, 'the test set (male)')
    # test_tool(covidNet, test_female_dataloader, 'the test set (female)')
    # test_tool(covidNet, test_under20_dataloader, 'the test & val set (under 20)')
    # test_tool(covidNet, test_20to40_dataloader, 'the test & val set (20-40)')
    # test_tool(covidNet, test_40to60_dataloader, 'the test & val set (40-60)')
    # test_tool(covidNet, test_over60_dataloader, 'the test & val set (over 60)')
    sens = []
    labels_under_40, preds_under_40 = test_tool(covidNet, test_under40_dataloader, 'the test set (under 40)')
    sens += [0] * len(labels_under_40)
    labels_over_40, preds_over_40 = test_tool(covidNet, test_over40_dataloader, 'the test set (over 40)')
    sens += [1] * len(labels_over_40)
    preds = preds_under_40 + preds_over_40
    labels = labels_under_40 + labels_over_40
    print(eq_oppo(preds, sens, labels), stat_parity(preds, sens))

# train()
print('--------Model--------')
test()
# print('--------Model (1st)--------')
# covidNet = MelSpectrogramCNN()
# covidNet.load_state_dict(torch.load("./covidNet_best.pth"))
# test()