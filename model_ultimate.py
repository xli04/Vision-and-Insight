import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import random
from torchsummary import summary

# Set random seed to ensure reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TransformSubset(torch.utils.data.Dataset):
    """Custom dataset subset class that can apply different transformations"""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __getitem__(self, idx):
        image, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(image).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def __len__(self):
        return len(self.indices)


# Custom dataset class for handling AI art dataset
class ArtDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Define art style categories
        self.art_styles = ['art_nouveau', 'baroque', 'expressionism', 'impressionism', 
                          'post_impressionism', 'realism', 'renaissance', 'romanticism', 
                          'surrealism', 'ukiyo-e', 'ukiyo_e']
        
        # Normalize art style names (handle ukiyo-e and ukiyo_e cases)
        self.normalize_style = {'ukiyo-e': 'ukiyo_e'}
        
        # Create combined label list and mapping
        self.combined_labels = []
        for style in self.art_styles:
            if style != 'ukiyo-e':
                style = self.normalize_style.get(style, style)
                self.combined_labels.extend([f'AI_SD_{style}', f'AI_LD_{style}', f'human_{style}'])
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.combined_labels)}
        
        # Load samples
        self.samples = self._load_samples()
        print(f"loaded {len(self.samples)} samples")
        
    def _load_samples(self):
        samples = []
        
        # Iterate through all folders
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            
            # Parse folder name to get label information
            style = folder
            if folder.startswith('AI_LD_') or folder.startswith('AI_SD_'):
                style = '_'.join(folder.split('_')[2:])
            
            # Normalize style name
            if style in self.normalize_style:
                style = self.normalize_style[style]
            
            # Skip folders not in predefined style list
            if style not in self.art_styles or style == 'ukiyo-e':
                continue
            
            # Build combined label
            if folder.startswith('AI_LD_'):
                combined_label = f'AI_LD_{style}'
            elif folder.startswith('AI_SD_'):
                combined_label = f'AI_SD_{style}'
            else:
                combined_label = f'human_{style}'
            
            # Get label index
            label_idx = self.label_to_idx[combined_label]
            
            # Get all images in the folder
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Store image path and combined label
                    samples.append((img_path, label_idx))
                    
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Return image and combined label
        return image, label

# Define Channel Attention Module (from CBAM)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

# Define Spatial Attention Module (from CBAM)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # Calculate average and maximum values along channel dimension
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Concatenate average and maximum values
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid activation
        x = self.conv(x)
        return self.sigmoid(x)

# Define CBAM Attention Module (Combining Channel and Spatial Attention)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        return x

# Enhanced Spatial Pyramid Pooling Module
class EnhancedSPP(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedSPP, self).__init__()
        
        # More scales of pooling
        self.pool1 = nn.AdaptiveAvgPool2d(1)  # Global pooling
        self.pool2 = nn.AdaptiveAvgPool2d(2)  # 2x2 grid pooling
        self.pool3 = nn.AdaptiveAvgPool2d(4)  # 4x4 grid pooling
        self.pool4 = nn.AdaptiveAvgPool2d(8)  # 8x8 grid pooling - add finer scale
        
        # Dimension reduction convolution after each pooling branch
        self.conv1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1, bias=False)
        
        # Batch normalization and activation function
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Adjustment convolution after fusion
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        size = x.size()
        
        # Pooling and dimension reduction
        x1 = F.interpolate(self.conv1(self.pool1(x)), size[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.conv2(self.pool2(x)), size[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.conv3(self.pool3(x)), size[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.conv4(self.pool4(x)), size[2:], mode='bilinear', align_corners=True)
        
        # Connect all scales
        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        
        # Final adjustment
        x = self.relu(self.bn(self.fusion_conv(x)))
        return x

# Enhanced Channel Importance Module
class EnhancedCIM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(EnhancedCIM, self).__init__()
        
        # Global average pooling + Global max pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Channel modulation network
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.LayerNorm(channels // reduction_ratio),  # Replace with LayerNorm for better stability
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Use both average pooling and max pooling to extract channel information
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        
        # Merge two pooling information
        y = y_avg + y_max
        
        # Generate channel weights through FC layers
        y = self.fc(y).view(b, c, 1, 1)
        
        # Recalibrate feature maps
        return x * y

# Complementary Cascade Module - Specially designed for effective integration of CBAM, SPP and CIM
class ComplementaryCascade(nn.Module):
    def __init__(self, in_channels):
        super(ComplementaryCascade, self).__init__()
        
        # Stage 1: Spatial Pyramid Pooling extracts multi-scale features
        self.spp = EnhancedSPP(in_channels)
        
        # Stage 2: CBAM attention refines feature focus
        self.cbam = CBAM(in_channels)
        
        # Stage 3: Channel Importance modulates for final output
        self.cim = EnhancedCIM(in_channels)
        
        # Residual connection adjustment layer - for dimension matching
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # Final feature adjustment
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Save input for residual connection
        identity = self.residual_conv(x)
        
        # Stage 1: Spatial Pyramid Pooling - solve receptive field and multi-scale issues
        # SPP is responsible for expanding receptive field and capturing multi-scale context
        x = self.spp(x)
        
        # Stage 2: CBAM attention - learn importance at channel and spatial levels
        # CBAM further refines which channels and regions are more important based on SPP
        x = self.cbam(x)
        
        # Stage 3: Channel Importance Module - final feature channel calibration
        # CIM performs final feature channel importance calibration after previous two stages
        x = self.cim(x)
        
        # Add residual connection - maintain information flow and gradient propagation
        x = x + identity
        
        # Final adjustment
        x = self.final_conv(x)
        return x

# FinalCNN V3 - Complementary Cascade Architecture
class FinalCNN(nn.Module):
    def __init__(self, num_classes=30):
        super(FinalCNN, self).__init__()
        
        # Stage 1: Shallow feature extraction
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: Shallow feature enhancement
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Complementary Cascade Block 1 - mid-level feature processing
        self.cc_block1 = ComplementaryCascade(128)
        
        # Stage 3: Mid to deep transition
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Complementary Cascade Block 2 - deep feature processing
        self.cc_block2 = ComplementaryCascade(256)
        
        # Global context processing
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),  # Add BatchNorm for better stability
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Stage 1-2: Basic feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        # Complementary Cascade Module 1: process mid-level features
        x = self.cc_block1(x)
        
        # Stage 3: Extract deeper features
        x = self.conv_block3(x)
        
        # Complementary Cascade Module 2: process deep features
        x = self.cc_block2(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        return x

# Train model function (for single label classification)
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward propagation and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward propagation
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}')
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)  # Adjust learning rate based on validation accuracy
            else:
                scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_final_cnn_v3_model.pth')
            print(f'Model saved with validation accuracy: {val_acc:.4f}')
    
    return model, history

# Model evaluation function (for single label classification)
def evaluate_model(model, test_loader, criterion, device='cuda'):
    model = model.to(device)
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    # For calculating confusion matrix and classification report
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward propagation
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Collect predictions and labels for metric calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total
    
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Calculate confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds)
    
    return test_loss, test_acc, cm, cr

# Visualize training history
def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_final_cnn_v3.png')
    plt.show()

# Visualize confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', filename='confusion_matrix_final_cnn_v3.png'):
    # Calculate row sums for normalization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Set diagonal elements to 0 to highlight misclassifications
    np.fill_diagonal(cm_normalized, 0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, cmap='YlOrRd', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Adjust label display
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

# Main function
def main():
    print("Starting FinalCNN V3 Training")
    # Set random seed
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Keep the same data preprocessing and augmentation as original CNN
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and data loaders
    train_data_dir = 'Real_AI_SD_LD_Dataset/train'
    test_data_dir = 'Real_AI_SD_LD_Dataset/test'
    
    # Create training and test datasets
    train_dataset = ArtDataset(root_dir=train_data_dir, transform=train_transform)
    test_dataset = ArtDataset(root_dir=test_data_dir, transform=test_transform)
    
    # Split training and validation sets (85%/15%)
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Generate indices using random seed
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}, Test set size: {len(test_dataset)}")
    
    # Create Complementary Cascade CNN model
    num_classes = len(train_dataset.dataset.combined_labels)  # Number of combined labels (10 styles x 3 sources = 30)
    model = FinalCNNv3(num_classes=num_classes)
    print(model)
    
    # Print model summary
    summary(model, (3, 224, 224), device='cpu')
    
    # Use the same loss function and optimizer as original CNN
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Use the same learning rate scheduler as original CNN
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,  # Keep the same number of training epochs as original CNN
        device=device
    )
    
    # Visualize training history
    plot_history(history)
    
    # Evaluate model
    test_loss, test_acc, cm, cr = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # Print classification report
    print("\nClassification Report:")
    print(cr)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm=cm,
        classes=train_dataset.dataset.combined_labels,
        title='Confusion Matrix (Misclassifications)',
        filename='final_cnn_v3_confusion_matrix.png'
    )

if __name__ == '__main__':
    main()