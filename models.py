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
    
def train_HW4Net():
    net1 = HW4Net()
    net1 = net1.to(device)
    loss_running_list_net1 = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net1.parameters(), lr = 1e-3, betas = (0.9, 0.99))
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(my_train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() #Sets gradients of all model parameters to zero. We want to compute fresh gradients
            #based on the new forward run. 
            outputs = net1(inputs)
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
                outputs = net1(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0) #add to total's total
                for n, i in enumerate(labels):
                    temp = np.array(i) #temp holds the one hot encoded label
                    idx = np.argmax(temp) #get the argmax of the encoded label - will be a value between 0 and 4.
                    #print(idx)
                    if idx == predicted[n]: #if the predicted value and label match
                        correct = correct + 1 #add to correct total

        print('Accuracy of the network on the val images: %d %%' % (
            100 * correct / total))
        

def test_HW4Net():
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

            outputs = net1(images)

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
    train_HW4Net()
    test_HW4Net()
    
