import torch
from torch.nn import parallel
import torch.nn as nn
import torch.nn.functional as F
from build_dataloader import my_train_dataloader, my_val_dataloader
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print('cuda gpu available')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if torch.torch.backends.mps.is_available():
    device = torch.device("mps")

class HW4Net(nn.Module):
    def __init__(self):
        super(HW4Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(6272,64)
        self.fc2 = nn.Linear(64, 5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_padded(nn.Module):
    def __init__(self):
        super(CNN_padded, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(8192,64)
        self.fc2 = nn.Linear(64, 5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv9 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv11 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, 3, padding=1)
        
        self.fc1 = nn.Linear(2048,64)
        self.fc2 = nn.Linear(64, 5)
    
    def forward(self, x): #we are passing in a torch.float32 into the network with a shape 12, 3, 64, 64
        
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.pool(F.relu(self.conv3(x)))
        
        x = F.relu(self.conv4(x))
        
        x = F.relu(self.conv5(x))
        
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))            
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
def train_model(model):
    
    model = model.to(device)
    loss_running_list_net1 = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, betas = (0.9, 0.99))
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(my_train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() #Sets gradients of all model parameters to zero. We want to compute fresh gradients
            #based on the new forward run. 
            outputs = model(inputs)
            loss = criterion(outputs, labels) #compute cross-entropy loss
            loss.backward() #compute derivative of loss wrt each gradient. 
            optimizer.step() #takes a step on hyperplane based on derivatives
            running_loss += loss.item() 
            if (i+1) % 100 == 0:
                print("[epoch: %d, batch: %5d] loss: %3f" % (epoch + 1, i + 1, running_loss / 100))
                loss_running_list_net1.append(running_loss/100)
                running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for n, data in enumerate(my_val_dataloader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0) #add to total's total
                for n, i in enumerate(labels):
                    temp = np.array(i.cpu()) #temp holds the one hot encoded label
                    idx = np.argmax(temp) #get the argmax of the encoded label - will be a value between 0 and 4.
                    #print(idx)
                    if idx == predicted[n]: #if the predicted value and label match
                        correct = correct + 1 #add to correct total

        print('Accuracy of the network on the val images: %d %%' % (
            100 * correct / total))
    return model  # trained model

def test_model(trained_model):
    ### Test performance of CNN 1 on val data ###
    correct = 0
    total = 0
    y_pred = []
    y_label = []
    mapping = { 0: 'airplane',
                1: 'bus',
                2: 'cat',
                3: 'dog',
                4: 'pizza'}


    with torch.no_grad():
        for n, data in enumerate(my_val_dataloader):
            images, labels = data

            outputs = trained_model(images)

            _, predicted = torch.max(outputs.data, 1) 

            total += labels.size(0) #add to total count of ground truth images so we can calculate total accuracy
            #print("total images in val set", total)
            for n, i in enumerate(labels):
                temp = np.array(i) #arrays are one hot encoded, we need to convert it into a human readable label for
                #display in the confusion matrix
                label_arg = np.argmax(temp) #get the argument of the one hot encoding
                y_label.append(mapping[label_arg]) #apply the argument to the mapping dictionary above. For example
                # if the argument is 3, then, that corresponds to a label of dog in the mapping dictionary
                t = int(np.array(predicted[n])) #get integer representation of prediction from network (will 
                #be an int from 0 to 4. 
                y_pred.append(mapping[t]) #append the predicted output of this label to the prediction list, but, 
                #via the mapping dictionary definition so that the y_pred list is human readable. 

                if label_arg == predicted[n]:
                    correct = correct + 1 #add to total count of correct predictions so we can calculate total accuracy
                

    print('Accuracy of the network on the val images: %d %%' % (
        100 * correct / total))
    from sklearn.metrics import confusion_matrix

    y_true = y_label
    y_pred = y_pred
    confusion_matrix=confusion_matrix(y_true, y_pred, labels = [ "airplane", "bus", "cat", "dog", "pizza"])
    disp = ConfusionMatrixDisplay(confusion_matrix, display_labels = [ "airplane", "bus", "cat", "dog", "pizza"])
    disp.plot()
    disp.ax_.set_title("Confusion Matrix for CNN 1")
    plt.show()
    plt.savefig('CM_CNN1')

if __name__ == "__main__":
    cnn_model_init = ResCNN()
    trained_model = train_model(cnn_model_init)
    trained_model = trained_model.to("cpu")
    test_model(trained_model)

    # cnn_padded_model_init = CNN_padded()
    # trained_model = train_model(cnn_padded_model_init)
    # trained_model = trained_model.to("cpu")
    # test_model(trained_model)
    
