import torch

import classes as cs
import arguments as arg
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the training and testing dataset
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5]),
     transforms.Lambda(lambda x: torch.flatten(x))])
# train_loader = cs.Dataset(arg.address1).loadData()
# test_loader = cs.Dataset(arg.address2).loadData()
train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=data_tf, download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=arg.batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./MNIST', train=False, transform=data_tf)
test_loader = DataLoader(dataset=test_dataset, batch_size=arg.batch_size, shuffle=False)

# Initialize the neurons
model = cs.Model(train_loader)

# Train the model using gradient descent
model.train()

# Make predictions on the testing set
model.test(test_loader)

# Save model and plot loss-curve
cs.plot_loss_curve(model)

# test = cs.Test(model, testingSet)
