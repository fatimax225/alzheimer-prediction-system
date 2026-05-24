import os
import torch
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# Get the current project directory
# __file__ represents the current Python file
# abspath converts it into a full absolute path
# dirname extracts only the folder location

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the path to the dataset folder
# The dataset folder is inside:
# static/dataset

DATA_DIR = os.path.join(BASE_DIR, "static", "dataset")

# Check if the dataset folder exists
# If the folder is missing, the program will stop
# and show a clear error message to help the user fix the issue

if not os.path.exists(DATA_DIR):
    raise Exception(
        f"️ Dataset folder not found at path: {DATA_DIR}\n"
        f"Please make sure the four image folders are inside it."
    )

# Image preprocessing step
# Before sending images to the neural network,
# all images need to be prepared in the same format

data_transform = transforms.Compose([

    # Resize all images to 128x128 pixels
    # This ensures every image has the same dimensions
    # which is required for deep learning models

    transforms.Resize((128, 128)),

    # Convert image into Tensor format
    # PyTorch models only work with tensors

    transforms.ToTensor(),

    # Normalize image values
    # This improves training stability and performance
    # It also helps the model learn faster

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# Print loading message

print("\n" + "=" * 50)
print(" Loading Dataset...")

# Load dataset using ImageFolder
# Each folder name represents a class label
# Example:
# MildDemented
# ModerateDemented
# NonDemented
# VeryMildDemented

dataset = ImageFolder(
    DATA_DIR,
    transform=data_transform
)

# Total number of images

n = len(dataset)

# Allocate 20% of data for testing

n_test = int(0.2 * n)

# Remaining 80% will be used for training

n_train = n - n_test

# Split the dataset into training and testing sets
# manual_seed(42) ensures the split is reproducible
# meaning we get the same split every run

train_data, test_data = random_split(
    dataset,
    [n_train, n_test],
    torch.Generator().manual_seed(42)
)

# DataLoader loads images in batches
# batch_size=32 means:
# 32 images are processed together at one time

# shuffle=True randomly mixes training images
# which improves learning quality

trainloader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

# Testing data is not shuffled
# because order is not important during evaluation

testloader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)

# Display dataset information

print(
    f" Found {n} images belonging to "
    f"{len(dataset.classes)} classes."
)

# Display dictionary of classes and their index numbers

print(f"Classes dictionary: {dataset.class_to_idx}")

print("=" * 50 + "\n")


# Create Convolutional Neural Network (CNN)
# CNN is commonly used for image classification tasks
# because it can automatically learn visual features

class ConvNet(torch.nn.Module):

    # Constructor function
    # Builds all layers of the neural network

    def __init__(self, num_classes=4):

        # Initialize parent class

        super(ConvNet, self).__init__()

        # First convolution block
        # Conv2d extracts image features
        # ReLU adds non-linearity
        # MaxPool reduces image size

        self.conv1 = torch.nn.Sequential(

            torch.nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        # Second convolution block

        self.conv2 = torch.nn.Sequential(

            torch.nn.Conv2d(
                16,
                32,
                3,
                padding=1
            ),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(
                2,
                2
            )
        )

        # Third convolution block

        self.conv3 = torch.nn.Sequential(

            torch.nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(
                2,
                2
            )
        )

        # Fourth convolution block

        self.conv4 = torch.nn.Sequential(

            torch.nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(
                2,
                2
            )
        )

        # Dropout layer
        # Helps reduce overfitting
        # Prevents the model from memorizing training data

        self.dp = torch.nn.Dropout(p=0.5)

        # Fully Connected Layer
        # Converts extracted features into predictions

        self.fc1 = torch.nn.Sequential(

            torch.nn.Linear(
                128 * 8 * 8,
                32
            ),

            torch.nn.ReLU()
        )

        # Final output layer
        # Produces prediction scores for each class

        self.fc2 = torch.nn.Linear(
            32,
            num_classes
        )

    # Forward function
    # Defines how data moves through the network

    def forward(self, x):

        # Pass image through first block
        x = self.conv1(x)

        # Pass through second block
        x = self.conv2(x)

        # Pass through third block
        x = self.conv3(x)

        # Pass through fourth block
        x = self.conv4(x)

        # Apply dropout
        x = self.dp(x)

        # Flatten multidimensional tensor
        # into a single vector

        x = x.view(-1, 128 * 8 * 8)

        # Pass through fully connected layer
        x = self.fc1(x)

        # Pass through output layer
        x = self.fc2(x)

        # Apply log_softmax
        # Converts outputs into probabilities

        out = F.log_softmax(x, dim=1)

        return out


# Select computing device
# If GPU is available → use CUDA
# Otherwise use CPU

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f" Using device: {device}\n")

# Create model instance
# Move model to selected device

net = ConvNet(
    num_classes=len(dataset.classes)
).to(device)

# Define optimizer
# Adam optimizer adjusts weights during training

optimizer = optim.Adam(
    net.parameters(),
    lr=0.001
)

# Define loss function
# CrossEntropyLoss is commonly used
# for multi-class classification problems

loss_func = torch.nn.CrossEntropyLoss()

# Number of training epochs
# One epoch means the model sees
# the entire training dataset once

epochs = 20

print("▶️ Starting Training...")

# Training loop

for epoch in range(epochs):

    # Set model to training mode

    net.train()

    # Store total training loss

    running_loss_train = 0.0

    # Loop through training batches

    for i, data in enumerate(trainloader, 0):

        # Extract images and labels

        inputs, targets = data

        # Move data to device

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Clear previous gradients

        optimizer.zero_grad()

        # Forward pass
        # Generate predictions

        outputs = net(inputs)

        # Calculate loss value

        loss = loss_func(outputs, targets)

        # Backpropagation
        # Calculate gradients

        loss.backward()

        # Update model weights

        optimizer.step()

        # Add batch loss to total loss

        running_loss_train += loss.item()

    # Calculate average training loss

    avg_train_loss = (
        running_loss_train / len(trainloader)
    )

    # Set model to evaluation mode

    net.eval()

    # Variables for accuracy calculation

    correct = 0
    total = 0

    # Store testing loss

    running_loss_test = 0.0

    # Disable gradient calculation during testing
    # Saves memory and improves speed

    with torch.no_grad():

        # Loop through testing batches

        for data in testloader:

            images, targets = data

            # Move data to device

            images = images.to(device)
            targets = targets.to(device)

            # Generate predictions

            outputs = net(images)

            # Calculate testing loss

            loss = loss_func(outputs, targets)

            running_loss_test += loss.item()

            # Get predicted class index

            _, predicted = torch.max(outputs, 1)

            # Increase total images count

            total += targets.size(0)

            # Count correct predictions

            correct += (
                predicted == targets
            ).sum().item()

    # Average testing loss

    avg_test_loss = (
        running_loss_test / len(testloader)
    )

    # Calculate accuracy percentage

    accuracy = 100 * correct / total

    # Print epoch results

    print(
        f"Epoch [{epoch + 1}/{epochs}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Test Loss: {avg_test_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )

# Training completed

print(' Training and evaluation complete!')

# Create path for saving the trained model

model_path = os.path.join(
    BASE_DIR,
    "pytorch_alzheimer_model.pth"
)

# Save trained model weights

torch.save(
    net.state_dict(),
    model_path
)

# Print confirmation message

print(
    f" Model saved successfully at: "
    f"{model_path}"
)

print(
    "You can now run the server using "
    "python3 app.py and test the website!"
)