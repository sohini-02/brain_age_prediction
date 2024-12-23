# imports
import dicom2nifti
import nibabel as nib
import nilearn as nil
from nilearn import plotting
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.transforms import functional as tf
import random
from torchvision import transforms
import time as time
from scipy.ndimage import zoom
from tensorflow.keras.utils import plot_model


start_time = time.time()
batch_size = 16
target_size = (128, 128)

# path defined
images_path = 'data_gz'
target_path = 'participants.tsv'


# preprocess labels
labels_all = pd.read_csv(target_path, sep='\t')
labels = (labels_all)['Age']


def resize_slice(slice_data, target_size):
    # Ensure slice_data is a NumPy array
    slice_data = np.array(slice_data)
    #print("Data is for sure a numpy array")
    
    # Ensure the slice is 2D
    if len(slice_data.shape) != 2:
        raise ValueError(f"Expected a 2D slice, got shape {slice_data.shape}")

    zoom_factors = [128 / slice_data.shape[0], 128 / slice_data.shape[1]]
    resized_slice = zoom(slice_data, zoom_factors, order=1)
    return resized_slice


# Define the normalization function
def normalize_labels(labels):
    min_age = labels.min()
    max_age = labels.max()
    return (labels - min_age) / (max_age - min_age), min_age, max_age


# Modify your BrainAgeDataset class to normalize labels
class BrainAgeDataset(Dataset):
    def __init__(self, image_dir, labels, target_size=(128, 128)):
        self.file_paths = sorted(glob.glob(os.path.join(image_dir, 'sub-pixar*_T1w.nii.gz')))
        self.labels = labels
        self.target_size = target_size
        self.augmented_images = []
        self.augmented_labels = []

        # Normalize labels
        self.normalized_labels, self.min_age, self.max_age = normalize_labels(self.labels)

        # Preprocess all data and store augmented versions
        self._augment_and_store()

    def _augment_and_store(self):
        """Perform augmentation and store images and labels."""
        for idx, file_path in enumerate(self.file_paths):
            # Load the 3D image
            single_image = nib.load(file_path)
            single_image_data = single_image.get_fdata()

            # Extract the middle slice for each view
            sagittal_index = single_image_data.shape[0] // 2
            coronal_index = single_image_data.shape[1] // 2
            axial_index = single_image_data.shape[2] // 2

            middle_sagittal_slice = single_image_data[sagittal_index, :, :]
            middle_coronal_slice = single_image_data[:, coronal_index, :]
            middle_axial_slice = single_image_data[:, :, axial_index]

            # Resize slices to target size
            middle_sagittal_slice = resize_slice(middle_sagittal_slice, self.target_size)
            middle_coronal_slice = resize_slice(middle_coronal_slice, self.target_size)
            middle_axial_slice = resize_slice(middle_axial_slice, self.target_size)

            # Convert slices to tensors
            original_set = torch.stack([
                torch.tensor(middle_sagittal_slice, dtype=torch.float32),
                torch.tensor(middle_coronal_slice, dtype=torch.float32),
                torch.tensor(middle_axial_slice, dtype=torch.float32)
            ], dim=0)

            # Append original images and normalized labels
            self.augmented_images.append(original_set)
            self.augmented_labels.append(self.normalized_labels[idx])

            # Create augmented (flipped) slices
            flipped_sagittal = torch.flip(torch.tensor(middle_sagittal_slice, dtype=torch.float32), dims=[1])
            flipped_coronal = torch.flip(torch.tensor(middle_coronal_slice, dtype=torch.float32), dims=[1])
            flipped_axial = torch.flip(torch.tensor(middle_axial_slice, dtype=torch.float32), dims=[1])

            flipped_set = torch.stack([flipped_sagittal, flipped_coronal, flipped_axial], dim=0)

            # Append flipped images and normalized labels
            self.augmented_images.append(flipped_set)
            self.augmented_labels.append(self.normalized_labels[idx])

    def __len__(self):
        """Return the total number of augmented images."""
        return len(self.augmented_images)

    def __getitem__(self, idx):
        """Retrieve augmented image and normalized label."""
        images = self.augmented_images[idx]
        label = self.augmented_labels[idx]  

        labels = torch.tensor([label], dtype=torch.float32)  # Wrap the single label in a list if needed
        return images, labels


class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(conv_bn_relu, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        #print("conv_bn_relu initialized: ", self.conv_bn_relu is not None)
        return self.conv_block(x)


# residual module
class residual_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residual_module, self).__init__()
        self.conv1 = conv_bn_relu(in_channels, out_channels)
        #print('conv block 1 in res')
        self.conv2 = conv_bn_relu(in_channels, out_channels)
        #print('conv block 2 in res')
        self.conv3 = conv_bn_relu(out_channels, out_channels)
        #print('conv block 3 in res')
    def forward(self, x):
        residual_module = x
        out1 = self.conv1(residual_module)
        #print('After first layer of res block')
        out2 = self.conv2(residual_module)
        #print('After 2nd layer of res block')
        out3 = self.conv3(out1)
        #print('After 3rd layer of res block')
        out = out2 + out3 
        #print('After res add')
        return out


class DeepSets(nn.Module):
    def __init__(self, in_features = 8388608, set_features=50):
        super(DeepSets, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        
        # Feature extractor layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, set_features)   #(o/p): (batch, 50)
        )
        
        # Log feature extractor layers
        self.log_feature_extractor = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, set_features),
            nn.ReLU(inplace=True)   #(o/p): (batch, 50)
        )
                                    
        # Linear transformations after concatenation
        self.l1 = nn.Linear(100, 30)   # (batch, 100) -> (batch, 30)
        self.lp = nn.ReLU()

        # Regressor for final output
        self.regressor = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Linear(130, 30),  #(batch, 130) -> (batch, 30)
            nn.ELU(inplace=True),
            nn.Linear(30, 10),  #(batch, 30) -> (batch, 10)
            nn.ELU(inplace=True),
            nn.Linear(10, 1), #(batch, 10) -> (batch, 1)
        )
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
    
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, input):
        # Extract features using both extractors
        #print('DS input')
        x1 = self.feature_extractor(input)      # (batch, 50)
        #print('DS fe')
        x2 = self.log_feature_extractor(input) + 0.001  # (batch, 50)
        #print('DS log_fe')

        # Apply log operation to x2
        x2 = x2.log()                          # (batch, 50)

        # Ensure both feature tensors have the same shape before concatenation
        #print(f"x1 shape before concatenation: {x1.shape}")  # (batch, 50)
        #print(f"x2 shape before concatenation: {x2.shape}")  # (batch, 50)

        # Concatenate the feature tensors along the feature dimension (dim=1)
        if x1.shape[0] == x2.shape[0] and x1.shape[1] == x2.shape[1]:
            x = torch.cat((x1, x2), dim=1)  # Concatenate along feature dimension   # (batch, 100)
        else:
            raise ValueError(f"Shape mismatch: x1 {x1.shape}, x2 {x2.shape}")

        #print(f"Concatenated tensor shape: {x.shape}")    # (batch, 100)

        # Apply linear transformations to the concatenated features
        #print('DS cat')
        x1 = self.l1(x)   # (batch, 30)
        #print('DS self 1: ', x1.shape)
        x2 = self.lp(x) + 0.001   # (batch, 100)
        #print('DS self 2: ', x2.shape)

        # Concatenate the results of the linear transformations
        x = torch.cat((x1, x2), dim=1)   # (batch, 130)
        #print('DS cat 2: ', x.shape)     # (batch, 130)

        # Pass through the regressor to get the final output
        x = self.regressor(x)      # (batch, 1)
        #print('DS regress')
        return x                # (batch, 1)


class final_model(nn.Module):
    def __init__(self, in_channels=3, set_features=50):
        super(final_model, self).__init__()
        
        # Conv-BN-ReLU Block
        #self.conv_bn_relu = conv_bn_relu(in_channels = 3, out_channels = 64)  
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #print('After 1st conv_bn_relu block in model')
        
        # Residual Modules (4 in total)
        self.residual1 = residual_module(64, 128)
        #print('Res1')
        self.residual2 = residual_module(128, 256)
        #print('Res2')
        self.residual3 = residual_module(256, 512)
        #print('Res3')
        #self.residual4 = residual_module(512, 512)
        #print('Res4')
        self.dropout = nn.Dropout(0.5)
        
        self.flatten = nn.Flatten()
        #print('Flatten')
        
        # DeepSets module
        self.deepsets = DeepSets(in_features = 8388608, set_features = set_features)  # (batch, 8388608) -> (batch, 1)
        #print('Deepsets')
    
    def forward(self, x):
        # Pass through Conv-BN-ReLU
        #print('Before 1st conv_bn_relu block in model')
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        #print('After 1st conv_bn_relu block: ', x.shape)

        # Pass through residual modules
        x = self.residual1(x)
        x = self.dropout(x)
        #print('Res1 shape: ', x.shape)
        x = self.residual2(x)
        x = self.dropout(x)
        #print('Res2 shape: ', x.shape)
        x = self.residual3(x)
        x = self.dropout(x)
        #print('Res3 shape: ', x.shape)
        #x = self.residual4(x)
        #print('Res4 shape: ', x.shape)
        x = self.dropout(x)
        
        # Flatten the output for DeepSets
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_features)
        x = self.dropout(x)
        #print('Flattened shape: ', x.shape)
        
        # Pass through DeepSets module
        x = self.deepsets(x)
        x = self.dropout(x)
        #print('DeepSets output shape: ', x.shape)
        
        return x
    
dataset = BrainAgeDataset(images_path, labels, target_size=(128, 128))
dataloader = DataLoader(dataset, batch_size = 24, shuffle=True)

# train-test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Train-validation split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


# Parameters:
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
model = final_model(in_channels = 3)
criterion = nn.MSELoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
batch_size = 16
#num_epochs = 200
num_epochs = 5
num_channels = 3

# Visualize model
#plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)


# training loop
max_train_accuracy = 0
min_train_loss = float('inf')


# training loop
max_train_accuracy = 0
min_train_loss = float('inf')


def train(model, train_loader, criterion, optimizer, epoch, tolerance=1.0):
    global max_train_accuracy, min_train_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_loss, correct, total = 0.0, 0, 0
    correct = 0
    total = 0
    model.train()
    
    for images, labels in tqdm(train_loader):
        # Send both images and labels to the device
        images = images.to(device)  # Ensure images are on the right device
        labels = labels.to(device)  # Ensure labels are on the right device
        #print("Input to model:", images.shape)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        correct += (torch.abs(outputs - labels) < tolerance).sum().item()
        
        # Update total with the batch size (number of samples in the batch)
        total += labels.size(0)
        
    train_accuracy = 100 * correct / total  # Ensure total is updated correctly
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_accuracy)
    print(f'Epoch [{epoch}], Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')


def validate(model, val_loader, criterion, tolerance = 1.0):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate how many predictions are within the tolerance
            correct += (torch.abs(outputs - labels) < tolerance).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total if total > 0 else 0
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    return val_loss, val_accuracy

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training if patience is reached
        return False

# Create an early stopping object
early_stopping = EarlyStopping(patience=5)

# Additional evaluation metrics: PAD, MAE-PAD, Mean PAD, SD-PAD
def evaluate_metrics(outputs, labels):
    """Calculate PAD, MAE-PAD, Mean PAD, and SD-PAD."""
    with torch.no_grad():
        # Compute PAD
        PAD = outputs.squeeze() - labels.squeeze()
        
        # Compute Mean Absolute PAD (MAE-PAD)
        MAE_PAD = torch.mean(torch.abs(PAD)).item()
        
        # Compute Mean PAD
        Mean_PAD = torch.mean(PAD).item()
        
        # Compute Standard Deviation of PAD (SD-PAD)
        SD_PAD = torch.std(PAD).item()
    
    return PAD, MAE_PAD, Mean_PAD, SD_PAD


def denormalize(predictions, min_age, max_age):
    return predictions * (max_age - min_age) + min_age

# In the evaluation section, apply denormalization to the outputs
def test_with_metrics(model, test_loader, criterion, tolerance=1.0):
    """Evaluate the model and compute evaluation metrics."""
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for faster evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the right device
            outputs = model(images)

            # Denormalize the outputs
            min_age = 3.0
            max_age = 40.0
            outputs = denormalize(outputs, min_age, max_age)
            
            # Compute loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)  # Accumulate total loss
            
            # Store outputs and labels for metric computation
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            
            # Update correct and total for accuracy computation
            total += labels.size(0)
            correct += (torch.abs(outputs - labels) < tolerance).sum().item()
    
    # Concatenate all outputs and labels into tensors
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    # Compute PAD, MAE-PAD, Mean PAD, and SD-PAD
    PAD, MAE_PAD, Mean_PAD, SD_PAD = evaluate_metrics(all_outputs, all_labels)
    
    # Compute accuracy
    accuracy = 100 * correct / total if total > 0 else 0  # Prevent division by zero
    
    # Print results
    avg_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"MAE-PAD: {MAE_PAD:.4f}, Mean PAD: {Mean_PAD:.4f}, SD-PAD: {SD_PAD:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return avg_loss, MAE_PAD, Mean_PAD, SD_PAD


# function to plot and save training/validation accuracy and loss
def plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig("tv_loss.png")
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.savefig("tv_accuracy.png")
    plt.show()
    

# Device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.backends.cudnn.enabled = False

# Training and validate loop with early stopping
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch, tolerance=1.0)
    validate(model, val_loader, criterion, tolerance = 1.0)
    val_loss, val_accuracy = validate(model, val_loader, criterion, tolerance=1.0)
    
    if early_stopping(val_loss):
        print("Early stopping triggered!")
        break

# Save plots
plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies)


# Testing
criterion = nn.MSELoss()
test_loss, MAE_PAD, Mean_PAD, SD_PAD = test_with_metrics(model, test_loader, criterion, tolerance = 1.0)


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")
