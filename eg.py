import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar
import os

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, Subset, DataLoader, random_split

# TODO: Construct your data in the following baseline structure: 1) ./Dataset/Train/image/, 2) ./Dataset/Train/label, 3) ./Dataset/Test/image, and 4) ./Dataset/Test/label
class LungDataset(Dataset):
    def __init__(self, root, transform):        
        self.root = root
        self.transform = transform

    def __len__(self):
        # Return number of points in the dataset
        imgs_path = os.path.join(self.root, 'image')
        return len(os.listdir(imgs_path))

    def __getitem__(self, idx):
        # Here we have to return the item requested by `idx`. The PyTorch DataLoader class will use this method to make an iterable for training/validation loop.
        img_path = os.path.join(self.root, 'image', f'{str(idx)}.png')
        label_path = os.path.join(self.root, 'label', f'{str(idx)}.txt')

        # Import image
        image = torch.tensor(plt.imread(img_path))

        # Get label of corresponding image
        l = open(label_path, 'r')
        label = int(l.read())

        return image, label


# Load the dataset and train and test splits
print("Loading datasets...")

# Data path


# Data normalization
MyTransform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Convert image to grayscale
    transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
    transforms.Normalize([0.5], [0.5]) # TODO: Normalize to zero mean and unit variance with appropriate parameters
])

DATA_train_path = LungDataset('./Dataset/Train', MyTransform)
DATA_test_path = LungDataset('./Dataset/Test', MyTransform)

# DATA_train = datasets.ImageFolder(root=DATA_train_path, transform=MyTransform)
# DATA_test = datasets.ImageFolder(root=DATA_test_path, transform=MyTransform)

print("Done!")

# Create dataloaders
# TODO: Experiment with different batch sizes
trainloader = DataLoader(DATA_train_path, batch_size=32, shuffle=True)
testloader = DataLoader(DATA_test_path, batch_size=32, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: [Transfer learning with pre-trained ResNet-50] 1) Define how many first layers of convolutoinal neural network (CNN) feature extractor in ResNet-50 to be "frozen" and 2) design your own fully-connected network (FCN) classifier.
        # 1) You will only refine last several layers of CNN feature extractor in ResNet-50 that mainly relate to high-level vision task. Determine how many first layers of ResNet-50 should be frozen to achieve best performances. Commented codes below will help you understand the architecture, i.e., "children", of ResNet-50.
        # 2) Design your own FCN classifier. Here I provide a sample of two-layer FCN.
        # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
        # Some common Choices are: Linear, ReLU, Dropout, MaxPool2d, AvgPool2d
        # If you have many layers, consider using nn.Sequential() to simplify your code
        
        # Load pretrained ResNet-50
        self.model_resnet = models.resnet50(pretrained=True)
        
        # The code below can show children of ResNet-50
        #child_counter = 0
        #for child in model.children():
        #    print(" child", child_counter, "is -")
        #    print(child)
        #    child_counter += 1
        
        # TODO: Determine how many first layers of ResNet-50 to freeze
        child_counter = 0
        for child in self.model_resnet.children():
            if child_counter < 47:
                for param in child.parameters():
                    param.requires_grad = False
            elif child_counter == 47:
                children_of_child_counter = 0
                for children_of_child in child.children():
                    if children_of_child_counter < 10:
                        for param in children_of_child.parameters():
                            param.requires_grad = False
                    else:
                        children_of_child_counter += 1
            else:
                print("child ",child_counter," was not frozen")
            child_counter += 1
        
        # Set ResNet-50's FCN as an identity mapping
        num_fc_in = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        
        # TODO: Design your own FCN
        self.fc1 = nn.Linear(num_fc_in, 64, bias = 0) # from input of size num_fc_in to output of size ?
            #eh: maybe its a 0? 
            #eh: nn.Linar(in_features~int,out_features~int,bias~bool) 
            #eh: if bias false, layer will not learn additive bias
            #eh: in_features: size of intput sample
            #eh: out_features: size of output sample
        self.fc2 = nn.Linear(64, 3, bias = 0 ) # from hidden layer to 3 class scores
            #eh: input feature = output feature above? 
            #eh: out_features: size 3

    def forward(self,x):
        # TODO: Design your own network, implement forward pass here
        
        relu = nn.ReLU() # No need to define self.relu because it contains no parameters
        with torch.no_grad():
            features = self.model_resnet(x)
            
        x = self.fc1(features) # Activation are flattened before being passed to the fully connected layers
            #eh: applies linear transformation to the datay = xA^T+b
        x = relu(x)
            #eh: applies rectified linear unit function element-wise
        x = self.fc2(x) 
            #eh: applies linear transform from hidden layers to x
        x = F.log_softmax(x) 
            #eh: outputs are confidence score

        # The loss layer will be applied outside Network class
        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = Network().to(device)
criterion = nn.CrossEntropyLoss() # Specify the loss layer (note: CrossEntropyLoss already includes LogSoftMax())
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=0.00001) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength (default: lr=1e-2, weight_decay=1e-4)
num_epochs = 4 # TOO: Choose an appropriate number of training epochs
    #eh: epoch counted as each full pass through data set (range 3-10) 
    #eh: too small~ model may not learn everything it could have
    #eh: chose 4 for now, but we can change it
def train(model, loader, num_epoch = num_epochs): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode

    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    print("Done!")

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc
    
train(model, trainloader, num_epochs)
print("Evaluate on test set")
evaluate(model, testloader)