import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from CNN import CNN
import time

# Configuration variables
device = torch.device("cpu")
batch_size = 128
epochs = 70
model_path = "model.pth"

# Defining the training transform
train_transform = transforms.Compose([
    transforms.ColorJitter(0.2,0.2,0.2,0.1),    #Adding colour augmentation
    transforms.RandomRotation(10),  # Adding random rotation
    transforms.RandomHorizontalFlip(),  # Adding random horiziontal flip
    transforms.RandomCrop(32, padding=4),   # Adding random crop
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform)
val_dataset.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initializing our model
model = CNN().to(device)

# Defining our loss function
loss_function = nn.CrossEntropyLoss()

# Defining our optimizer function
optimizer_function = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#optimizer_function = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Defining our schedular function
#scheduler_function = scheduler.StepLR(step_size=30, gamma=0.1, optimizer=optimizer_function)

scheduler_function = scheduler.ReduceLROnPlateau(
    optimizer=optimizer_function,
    mode='max',              
    factor=0.1,              
    patience=4,             
    threshold=0.002,        # minimum change to qualify as improvement
    cooldown=0,              
    min_lr=1e-6,            
)

def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total


# Used to train the data
def train():
    training_time = time.time()
    for epoch in range(epochs): # Iterate through each epoch

        start_time = time.time()    # Used to keep track of time

        # Used to calculate epoch accuracy
        total = 0
        correct = 0

        total_loss = 0.0    # Used to keep track of the loss for the epoch

        for images, labels in train_loader: # Iterate through each image and corresponding label

            images, labels = images.to(device), labels.to(device)

            optimizer_function.zero_grad()  # Zero the gradient of the optimizer function

            outputs = model(images) # Get the outputs caluclated 

            loss = loss_function(outputs, labels)   # Caluclate the loss of the epoch

            loss.backward() #Backpropagate 

            optimizer_function.step()   # Update the weights 

            total_loss += loss.item()   # Add to the total loss

            # To compute the accuracy at each epoch
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total * 100
        epoch_time = time.time() - start_time
        val_accuracy = evaluate(val_loader)

        print(f"Epoch - {epoch+1}/{epochs}\nLoss - {total_loss/len(train_loader):.6f}\nAccuracy - {accuracy:.2f}%\nValidation Accuracy - {val_accuracy:.2f}%\nTime - {epoch_time:.2f}s\n")

        scheduler_function.step(val_accuracy)   # Step the schedular function

    total_time = time.time() - training_time
    print(f"Training Time: {total_time/60:.2f}m")

if __name__ == "__main__":
    train() 
    torch.save(model.state_dict(), model_path)
