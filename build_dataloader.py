import torch
import os
import torchvision
from torchvision import transforms as tvt
import numpy as np
from PIL import Image
import random

### Create data_loader ###


root_train = 'train/'
root_val = 'val/'
catNms=['airplane','bus','cat', 'dog', 'pizza']

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, catNms, val : bool):
        super(MyDataset).__init__()
        self.val = val
        self.root = {} #dictionary for main directory which holds all the images of a category
        self.filenames = {} #dictionary for filenames of a given category
        for cat in catNms:
            self.root[cat] = root + cat + '/'
        for cat in catNms:
        #create list of image files in each category that can be opened by __getitem__
            self.filenames[cat] = os.listdir(self.root[cat])
        
        self.rand_max = len(os.listdir(self.root[catNms[0]])) - 1 #number of files in directory

        self.mapping = {0 : 'airplane',
                        1: 'bus',
                        2: 'cat',
                        3: 'dog',
                        4: 'pizza'} #makes it easy to convert between index and name of a category.
        
        self.one_hot_encoding = {0: torch.tensor(np.array([1, 0, 0, 0, 0])),
                                1: torch.tensor(np.array([0, 1, 0, 0, 0])),
                                2: torch.tensor(np.array([0, 0, 1, 0, 0])),
                                3: torch.tensor(np.array([0, 0, 0, 1, 0])),
                                4: torch.tensor(np.array([0, 0, 0, 0, 1]))} #one hot encode each category. 

        if self.val == False: # we are training, then we want to do data augmentation
            self.to_Tensor_and_Norm = tvt.Compose([tvt.ToTensor(),tvt.Resize((64,64)) , 
                                            tvt.Normalize([0], [1]) ,  tvt.ColorJitter(0.75, 0.75) , 
                                            tvt.RandomHorizontalFlip( p = 0.75), 
                                            tvt.RandomRotation(degrees = 45)]) #normalize and resize in case the resize op 
    #         wasn't done. Note that resizing here may not have any impact as the resizing was done previously.  
        if self.val == True: # do not do augmentation in validation.
            self.to_Tensor_and_Norm = tvt.Compose([tvt.ToTensor(),tvt.Resize((64,64)), tvt.Normalize([0], [1])]) 

    def __len__(self):
        count = 0
        for cat in catNms:
            temp_num = os.listdir(self.root[cat])
            count = count + len(temp_num)
        return count #return count. Will be 2500 if the root=val/ and 7500 if root=train/

    def __getitem__(self, index):
        file_index = index % self.rand_max + 1
        class_index = index % 5
 
        img_file = self.filenames[self.mapping[class_index]]
        
        try:
            item = Image.open(self.root[self.mapping[class_index]] + img_file[file_index])
        except IndexError: #for debugging
            print('these are the indices for the line above when shape is correct', class_index , file_index)
            
        np_img = np.array(item)
        shape = np_img.shape
        while shape != (64, 64 ,3): #handle if the image from COCO is grayscale. 
            #print('found a grayscale image, fetching an RGB!')
            another_rand = random.randint(0,self.rand_max)  #generate another rand num
            #print('another_rand is', another_rand)
            try:
                item = Image.open(self.root[self.mapping[class_index]] + img_file[another_rand])
            except IndexError: #for debugging
                print('these are the indices for the line above when shape is incorrect', another_rand , class_index)            
            np_img = np.array(item)
            shape = np_img.shape

        img = self.to_Tensor_and_Norm(item)
        class_label = self.one_hot_encoding[class_index].type(torch.FloatTensor) #convert to Float 
        return img, class_label
    
#instantiate objects for train and val datasets.  
my_train_dataset = MyDataset(root_train, catNms, val=False)
print(len(my_train_dataset))
index = 3
print(my_train_dataset[index][0].shape, my_train_dataset[index][1])
my_val_dataset = MyDataset(root_val, catNms, val = True)
print(len(my_val_dataset))
print(my_val_dataset[index][0].shape, my_val_dataset[index][1])

# Use MyDataset class in PyTorches DataLoader functionality
my_train_dataloader = torch.utils.data.DataLoader(my_train_dataset, batch_size=32, num_workers = 32, drop_last=True)
my_val_dataloader = torch.utils.data.DataLoader(my_val_dataset, batch_size = 32, num_workers = 32, drop_last = True)
# for n, batch in enumerate(my_train_dataloader):
# #     #Note: each batch is a list of length 2. The first is a pytorch tensor B x C x H x W and the 
# #     #second is a pytorch tensor of length B with the associated class labels of each image in the 
# #     #first item of the list!
#     print('batch is', n)
