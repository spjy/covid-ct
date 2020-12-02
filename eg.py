import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import pprint

pp = pprint.PrettyPrinter(indent=4)

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, Subset, DataLoader, random_split, ConcatDataset

# Parameters for easy printing
parameters = {
    'batch_size': 32,
    'child_counter': 5,
    'children_of_child_counter': 1,
    'neurons': 128,
    'bias': 1,
    'activation': 'softmax',
    'loss': 'cross entropy loss',
    'optimizer': 'adamw',
    'layers': 3,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'epochs': 30,
    'mean': (0.6495729088783264,0.6495729088783264,0.6495729088783264),
    'std': (0.2604725658893585,0.2604725658893585,0.2604725658893585),
    'training_loss': [],
    'testing_loss': [],
    'evaluation_accuracy': [],
    'total_accuracy': None
}

# TODO: Construct your data in the following baseline structure: 1) ./Dataset/Train/image/, 2) ./Dataset/Train/label, 3) ./Dataset/Test/image, and 4) ./Dataset/Test/label
class LungDataset(Dataset):
    def __init__(self, root):        
        self.root = root

    def __len__(self):
        # Return number of points in the dataset based on root path
        imgs_path = os.path.join(self.root, 'image')
        return len(os.listdir(imgs_path))

    def __getitem__(self, idx):
        # Here we have to return the item requested by `idx`. The PyTorch DataLoader class will use this method to make an iterable for training/validation loop.
        # File names are based on idx.
        img_path = os.path.join(self.root, 'image', f'{str(idx)}.png')
        label_path = os.path.join(self.root, 'label', f'{str(idx)}.txt')

        # Import image
        # Transpose to be 3x244x244
        image = np.transpose(torch.tensor(plt.imread(img_path)), (2, 0, 1))

        # Normalize image to reduce computation
        image = transforms.Normalize(parameters['mean'], parameters['std']).forward(image)
        
        # Greyscale image
        image = transforms.Grayscale(num_output_channels=3).forward(image)
        
        # Get label of corresponding image
        l = open(label_path, 'r')
        label = int(l.read())

        # Return manipulated image and label
        return image, label


# Load the dataset and train and test splits
print("Loading datasets...")

# Store dataset in folder based on fold
fold = [
  LungDataset('./Dataset/fold0'),
  LungDataset('./Dataset/fold1'),
  LungDataset('./Dataset/fold2'),
  LungDataset('./Dataset/fold3'),
  LungDataset('./Dataset/fold4')
]

print("Done!")

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
        # child_counter = 0
        # for child in self.model_resnet.children():
        #    print(" child", child_counter, "is -")
        #    print(child)
        #    child_counter += 1
        
        # TODO: Determine how many first layers of ResNet-50 to freeze - 5
        child_counter = 0
        for child in self.model_resnet.children():
            if child_counter < parameters['child_counter']:
                for param in child.parameters():
                    param.requires_grad = False
            elif child_counter == parameters['child_counter']:
                children_of_child_counter = 0
                for children_of_child in child.children():
                    if children_of_child_counter < parameters['children_of_child_counter']:
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

        # Input layer
        self.fc1 = nn.Linear(num_fc_in, 64, bias = parameters['bias']) # from input of size num_fc_in to output of size ?
        
        # Inner layers. Raise then reduce dimensionality
        self.inner1 = nn.Linear(64, parameters['neurons'], bias=parameters['bias'])
        self.inner2 = nn.Linear(parameters['neurons'], 16, bias=parameters['bias'])

        self.fc2 = nn.Linear(16, 3, bias = parameters['bias']) # from hidden layer to 3 class scores
            #eh: input feature = output feature above? 
            #eh: out_features: size 3

        # Pool to 224 * (3/4), stride of 1
        self.pool = nn.MaxPool2d(168, stride=1)

    def forward(self,x):
        # TODO: Design your own network, implement forward pass here
        
        # Use softplus before each layer as activation function
        af = nn.Softplus()
        with torch.no_grad():
            features = self.model_resnet(x)

        # Pooling
        x = af(x)
        x = self.pool(x)
        
        # Linear layers
        x = self.fc1(features) # Activation are flattened before being passed to the fully connected layers

        x = af(x)
        x = self.inner1(x)

        x = af(x)
        x = self.inner2(x)

        x = af(x)
        x = self.fc2(x)

        # The loss layer will be applied outside Network class
        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = Network().to(device)
criterion = nn.CrossEntropyLoss() # Specify the loss layer (note: CrossEntropyLoss already includes LogSoftMax())
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters['learning_rate'], weight_decay=parameters['weight_decay']) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength (default: lr=1e-2, weight_decay=1e-4)
num_epochs = parameters['epochs'] # TOO: Choose an appropriate number of training epochs

def train(model, loader, testloader, num_epoch = num_epochs): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode

    # Store testing/training loss per epoch
    l = []
    t = []

    for i in range(num_epoch):
        running_loss = []
        test_loss = []

        # Train set
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights

        l.append(np.mean(running_loss))

        # Test set
        with torch.no_grad():
          for batch, label in tqdm(testloader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            test_loss.append(loss.item())

        t.append(np.mean(test_loss))

        print("Epoch {} train loss:{}, test loss:{}".format(i+1,np.mean(running_loss), np.mean(test_loss))) # Print the average loss for this epoch
    print("Done!")

    return l, t

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

# 5 Fold Cross Validation
for i in range(5):
  # Copy fold array
  training = fold.copy()

  testing = DataLoader(training[i], batch_size=32, shuffle=True)

  del training[i]

  training_datasets = DataLoader(ConcatDataset(training), batch_size=32, shuffle=True)
  training_loss, testing_loss = train(model, training_datasets, testing, num_epochs)
  parameters['training_loss'].append(training_loss)
  parameters['testing_loss'].append(testing_loss)

  print(f'Fold: {i} / training loss: {training_loss}, testing loss: {testing_loss}')
  
  print("Evaluate on test set")
  evaluation_accuracy = evaluate(model, testing)
  parameters['evaluation_accuracy'].append(evaluation_accuracy)

  for layer in model.children():
   if hasattr(layer, 'reset_parameters'):
       layer.reset_parameters()

# Plot training loss/testing loss, parameters
fig, axs = plt.subplots(5, sharey=True)

fig.suptitle('Loss / epoch')
fig.set_figheight(15)
fig.set_figwidth(8)

for i in range(len(parameters['training_loss'])):
  x = range(len(parameters['training_loss'][i]))
  axs[i].plot(x, parameters['training_loss'][i], '-o')
  axs[i].plot(x, parameters['testing_loss'][i], '-x')

x = range(len(parameters['testing_loss']))

plt.xticks(range(len(parameters['training_loss'][i])))

plt.ylabel('Loss')
plt.xlabel('Epoch')

parameters['total_accuracy'] = np.mean(parameters['evaluation_accuracy'])
# Parameters
plt.figtext(0, 1, f'Total accuracy: {parameters["total_accuracy"]}', wrap=True)
plt.figtext(0, 1.01, 'Blue=training loss, Orange=testing loss', wrap=True)
  
plt.show()

print(parameters)

print(f"Total accuracy: {parameters['total_accuracy']}")
