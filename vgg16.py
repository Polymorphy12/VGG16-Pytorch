# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from custom_dataset import BirdsDataset


#create vgg16
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.fc1 = nn.Linear(in_features=25088,out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=325)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.max_pool(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = self.max_pool(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))

        x = self.max_pool(x)

        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))

        x = self.max_pool(x)

        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))

        x = self.max_pool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 1e-3
batch_size = 1
num_epochs = 1

#load data
train_dataset = BirdsDataset(csv_file="train.csv",
                             root_dir="./",
                             transform=transforms.ToTensor())
valid_dataset = BirdsDataset(csv_file="valid.csv",
                             root_dir="./",
                             transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,shuffle=True)


# Model
model = VGG16()
model.to(device)
# x = torch.randn(16,3,224,224)
# x = x.to(device)
# print(model(x).shape)
# exit()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr= learning_rate)

# Train Network
for epoch in range(num_epochs):
    print("train epoch ", epoch)
    for idx, (data,targets) in enumerate(train_loader):
        if idx % 50 == 0:
            print("iter ", idx)
        #Get data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        print(data.shape)
        # exit()

        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent
        optimizer.step()

#Check accuracy on training & test set to see how good our model
def check_accuracy(loader, model, is_train):
    if is_train:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += predictions==y.sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    model.train()


check_accuracy(train_loader, model, True)
check_accuracy(valid_loader, model, False)