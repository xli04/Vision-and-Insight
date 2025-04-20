import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
from model_basic import BasicCNN, ArtDataset
from model_filter import EnhancedCNN  # Import for the enhanced CNN with SPP and CIM

# Set random seed to ensure reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Image preprocessing function
def preprocess_image(img_path, transform=None):
    """Preprocess image"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor, img

# Directly visualize convolutional filter weights - adapted for both model types
def visualize_filters(model, layer_name, num_filters=16, save_path=None, model_type="basic", true_label=None, pred_label=None):
    """Visualize convolutional filter weights of specified layer"""
    # Get the target layer based on model type
    if model_type == "basic":
        if layer_name == 'conv_block1':
            layer = model.conv_block1[0]
        elif layer_name == 'conv_block2':
            layer = model.conv_block2[0]
        elif layer_name == 'conv_block3':
            layer = model.conv_block3[0]
        elif layer_name == 'conv_block4':
            layer = model.conv_block4[0]
        else:
            raise ValueError(f"Unknown layer name: {layer_name} for basic model")
    elif model_type == "enhanced":
        if layer_name == 'conv_block1':
            layer = model.conv_block1[0]
        elif layer_name == 'conv_block2':
            layer = model.conv_block2[0]
        elif layer_name == 'conv_block3':
            layer = model.conv_block3[0]
        elif layer_name == 'conv_block4':
            layer = model.conv_block4[0]
        elif layer_name == 'spp':
            layer = model.spp.conv
        elif layer_name == 'cim1':
            return  # CIM modules don't have direct filter weights to visualize
        elif layer_name == 'cim2':
            return
        elif layer_name == 'cim3':
            return
        elif layer_name == 'cim4':
            return
        else:
            raise ValueError(f"Unknown layer name: {layer_name} for enhanced model")
    
    # Get weights
    weights = layer.weight.data.cpu().numpy()
    
    # Determine number of filters to display
    num_filters = min(num_filters, weights.shape[0])
    
    # Calculate subplot layout
    n_cols = 4
    n_rows = (num_filters + n_cols - 1) // n_cols
    
    # Create figure
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    
    # Iterate and display each filter
    for i in range(num_filters):
        # For first layer, display RGB channels directly
        if weights.shape[1] == 3:
            # Transpose to (H, W, C) format for display
            filter_img = weights[i].transpose(1, 2, 0)
            
            # Normalize to [0, 1] range
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
            
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(filter_img)
            plt.axis('off')
            plt.title(f'Filter {i+1}')
        else:
            # For deeper layers, take first input channel's weights
            filter_img = weights[i, 0]
            
            # Normalize to [0, 1] range
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
            
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(filter_img, cmap='viridis')
            plt.axis('off')
            plt.title(f'Filter {i+1}')
    
    plt.tight_layout()
    
    # Create title with additional label information
    title = f'{model_type.capitalize()} Model - Filters from {layer_name}'
    if true_label and pred_label:
        title += f'\nTrue: {true_label} | Predicted: {pred_label}'
        
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Filter visualization saved to: {save_path}")
    
    return plt.gcf()

# Visualize filter activations using gradient ascent - adapted for both model types
class FilterVisualizer:
    def __init__(self, model, layer_name, model_type="basic"):
        self.model = model
        self.model.eval()
        self.model_type = model_type
        
        # Get the target layer based on model type
        if model_type == "basic":
            if layer_name == 'conv_block1':
                self.target_layer = model.conv_block1[0]
            elif layer_name == 'conv_block2':
                self.target_layer = model.conv_block2[0]
            elif layer_name == 'conv_block3':
                self.target_layer = model.conv_block3[0]
            elif layer_name == 'conv_block4':
                self.target_layer = model.conv_block4[0]
            else:
                raise ValueError(f"Unknown layer name: {layer_name} for basic model")
        elif model_type == "enhanced":
            if layer_name == 'conv_block1':
                self.target_layer = model.conv_block1[0]
            elif layer_name == 'conv_block2':
                self.target_layer = model.conv_block2[0]
            elif layer_name == 'conv_block3':
                self.target_layer = model.conv_block3[0]
            elif layer_name == 'conv_block4':
                self.target_layer = model.conv_block4[0]
            elif layer_name == 'spp':
                self.target_layer = model.spp.conv
            elif layer_name == 'cim1':
                self.target_layer = model.cim1.fc[0]  # First layer of the FC in CIM
            elif layer_name == 'cim2':
                self.target_layer = model.cim2.fc[0]
            elif layer_name == 'cim3':
                self.target_layer = model.cim3.fc[0]
            elif layer_name == 'cim4':
                self.target_layer = model.cim4.fc[0]
            else:
                raise ValueError(f"Unknown layer name: {layer_name} for enhanced model")
        
        self.activations = None
        
        # Register hook
        self.hook = self.target_layer.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        self.activations = output
    
    def _normalize(self, img):
        """Normalize image to [0, 1] range"""
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        return img
    
    def visualize(self, filter_idx, img_size=224, num_iterations=30, learning_rate=0.1, save_path=None):
        """Visualize specified filter activation using gradient ascent"""
        # Enable gradient anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        # Create random noise image as starting point
        device = next(self.model.parameters()).device
        img = torch.randn(1, 3, img_size, img_size, requires_grad=True, device=device)
        
        # Optimize image to maximize specified filter activation
        optimizer = torch.optim.Adam([img], lr=learning_rate)
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward propagation
            self.model(img.clone())  # Use clone() to avoid in-place operations
            
            # Get specified filter activation
            if self.activations is None:
                print("Warning: No activation detected")
                break
                
            # Select specified filter
            if len(self.activations.shape) >= 4:  # For convolutional layers
                activation = self.activations[0, filter_idx]
                loss = -activation.mean()
            else:  # For FC layers (like in CIM modules)
                if filter_idx < self.activations.shape[1]:
                    activation = self.activations[0, filter_idx]
                    loss = -activation
                else:
                    print(f"Filter index {filter_idx} out of range for layer with {self.activations.shape[1]} filters")
                    break
            
            # Backward propagation
            loss.backward(retain_graph=True)
            
            # Update image
            optimizer.step()
            
            # Apply regularization to keep image visible
            with torch.no_grad():
                img.data = torch.clamp(img.data, -1, 1)
        
        # Remove hook
        self.hook.remove()
        
        # Convert to visualization format
        img_np = img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        img_np = self._normalize(img_np)
        
        # Display result
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f'{self.model_type.capitalize()} Model - {filter_idx} Activation')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Filter activation visualization saved to: {save_path}")
        
        return plt.gcf(), img_np

# Visualize feature maps - adapted for both model types
class FeatureMapVisualizer:
    def __init__(self, model, model_type="basic"):
        self.model = model
        self.model.eval()
        self.model_type = model_type
        self.hooks = []
        self.feature_maps = {}
    
    def _hook_fn(self, name):
        def hook(module, input, output):
            self.feature_maps[name] = output.detach()
        return hook
    
    def register_hooks(self, layer_names):
        """Register hooks for specified layers"""
        # Clear existing hooks
        self.remove_hooks()
        
        for name in layer_names:
            # For basic CNN model
            if self.model_type == "basic":
                if name == 'conv_block1':
                    hook = self.model.conv_block1.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
                elif name == 'conv_block2':
                    hook = self.model.conv_block2.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
                elif name == 'conv_block3':
                    hook = self.model.conv_block3.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
                elif name == 'conv_block4':
                    hook = self.model.conv_block4.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
            # For enhanced CNN model
            elif self.model_type == "enhanced":
                if name == 'conv_block1':
                    hook = self.model.conv_block1.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
                elif name == 'conv_block2':
                    hook = self.model.conv_block2.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
                elif name == 'conv_block3':
                    hook = self.model.conv_block3.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
                elif name == 'conv_block4':
                    hook = self.model.conv_block4.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
                elif name == 'spp':
                    hook = self.model.spp.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
                elif name == 'cim1':
                    hook = self.model.cim1.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
                elif name == 'cim2':
                    hook = self.model.cim2.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
                elif name == 'cim3':
                    hook = self.model.cim3.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
                elif name == 'cim4':
                    hook = self.model.cim4.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_feature_maps(self, img_tensor, layer_name, num_features=16, save_path=None, true_label=None, pred_label=None):
        """Visualize feature maps of specified layer"""
        # Ensure hooks are registered
        self.register_hooks([layer_name])
        
        # Forward propagation
        device = next(self.model.parameters()).device
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            self.model(img_tensor)
        
        # Get feature maps
        if layer_name not in self.feature_maps:
            print(f"Warning: Feature maps for layer {layer_name} not found")
            self.remove_hooks()
            return None
        
        feature_maps = self.feature_maps[layer_name][0].cpu().numpy()
        
        # Handle CIM modules which may have different output dimensions
        if len(feature_maps.shape) == 1 and layer_name.startswith('cim'):
            # CIM modules might have channel attention weights as 1D
            feature_maps = feature_maps.reshape(1, -1)
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(feature_maps[0])), feature_maps[0])
            
            # Create title with additional label information
            title = f'{self.model_type.capitalize()} Model - Channel Attention Weights from {layer_name}'
            if true_label and pred_label:
                title += f'\nTrue: {true_label} | Predicted: {pred_label}'
                
            plt.title(title)
            plt.xlabel('Channel Index')
            plt.ylabel('Weight')
            
            if save_path:
                plt.savefig(save_path)
                print(f"CIM attention weights visualization saved to: {save_path}")
            
            self.remove_hooks()
            return plt.gcf()
            
        # Determine number of feature maps to display
        num_features = min(num_features, feature_maps.shape[0])
        
        # Calculate subplot layout
        n_cols = 4
        n_rows = (num_features + n_cols - 1) // n_cols
        
        # Create figure
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        
        # Iterate and display each feature map
        for i in range(num_features):
            feature = feature_maps[i]
            
            # Normalize to [0, 1] range
            feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
            
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(feature, cmap='viridis')
            plt.axis('off')
            plt.title(f'Feature {i+1}')
        
        plt.tight_layout()
        
        # Create title with true and predicted label information
        title = f'{self.model_type.capitalize()} Model - Feature Maps from {layer_name}'
        if true_label and pred_label:
            title += f'\nTrue: {true_label} | Predicted: {pred_label}'
            
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Feature map visualization saved to: {save_path}")
        
        # Remove hooks
        self.remove_hooks()
        
        return plt.gcf()

# Generate Grad-CAM heatmaps for both model types
# Note: This class has been removed as per requirements

def main():
    # Set random seed
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset to get class information
    test_data_dir = 'Real_AI_SD_LD_Dataset/test'
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset to get class information
    test_dataset = ArtDataset(test_data_dir, transform=test_transform)
    
    # Load both models
    basic_model = BasicCNN(num_classes=len(test_dataset.combined_labels))
    basic_model.load_state_dict(torch.load('best_model.pth', map_location=device))
    basic_model.to(device)
    basic_model.eval()
    
    enhanced_model = EnhancedCNN(num_classes=len(test_dataset.combined_labels))
    enhanced_model.load_state_dict(torch.load('best_enhanced_cnn.pth', map_location=device))
    enhanced_model.to(device)
    enhanced_model.eval()
    
    # Create output directory
    base_output_dir = 'model_comparison_visualization'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Dictionary to track found samples per label
    found_samples_per_label = {}
    
    # Find samples misclassified by basic but correctly classified by enhanced
    style_dirs = os.listdir(test_data_dir)
    
    # Debug: print all available directories and labels
    print("\nAll directories in test_data_dir:")
    for dir_name in style_dirs:
        print(f"  - {dir_name}")
    
    print("\nAll labels in test_dataset.label_to_idx:")
    for label in sorted(test_dataset.label_to_idx.keys()):
        print(f"  - {label}")
    
    # Track statistics for all source types
    source_stats = {
        "AI_SD": {"total": 0, "found": 0, "correctly_classified_by_basic": 0, "misclassified_by_enhanced": 0},
        "AI_LD": {"total": 0, "found": 0, "correctly_classified_by_basic": 0, "misclassified_by_enhanced": 0},
        "human": {"total": 0, "found": 0, "correctly_classified_by_basic": 0, "misclassified_by_enhanced": 0}
    }
    
    # Set the number of samples to find for each label
    samples_per_label = 3
    
    for style_dir in style_dirs:
        style_path = os.path.join(test_data_dir, style_dir)
        if not os.path.isdir(style_path):
            continue
            
        # Use the directory name directly as the label
        true_label = style_dir
        
        # Determine source type for statistics
        if true_label.startswith("AI_SD_"):
            source_type = "AI_SD"
        elif true_label.startswith("AI_LD_"):
            source_type = "AI_LD"
        else:
            source_type = "human"
        
        # Debug: print current directory as labels
        print(f"\nProcessing directory: {style_dir} (Source: {source_type})")

        if source_type == "human":
            true_label = "human_" + true_label
        
        true_idx = test_dataset.label_to_idx.get(true_label)
        
        if true_idx is None:
            print(f"WARNING: Could not find index for label '{true_label}'. Skipping.")
            continue
        
        if true_label not in found_samples_per_label:
            found_samples_per_label[true_label] = []
        
        # Only collect up to samples_per_label samples for each label
        if len(found_samples_per_label[true_label]) >= samples_per_label:
            print(f"Already have {samples_per_label} samples for {true_label}. Skipping.")
            continue
        
        images = [f for f in os.listdir(style_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Update total count for this source type
        source_stats[source_type]["total"] += len(images)
        
        for img_name in images:
            if len(found_samples_per_label[true_label]) >= samples_per_label:
                break
                
            img_path = os.path.join(style_path, img_name)
            
            img_tensor, _ = preprocess_image(img_path, test_transform)
            img_tensor = img_tensor.to(device)
            
            with torch.no_grad():
                basic_outputs = basic_model(img_tensor)
                _, basic_pred_idx = torch.max(basic_outputs, 1)
                basic_pred_idx = basic_pred_idx.item()
                
                enhanced_outputs = enhanced_model(img_tensor)
                _, enhanced_pred_idx = torch.max(enhanced_outputs, 1)
                enhanced_pred_idx = enhanced_pred_idx.item()
            
            # Get basic prediction label for logging
            basic_pred_label = None
            for label, idx in test_dataset.label_to_idx.items():
                if idx == basic_pred_idx:
                    basic_pred_label = label
                    break
                    
            # Get enhanced prediction label for logging
            enhanced_pred_label = None
            for label, idx in test_dataset.label_to_idx.items():
                if idx == enhanced_pred_idx:
                    enhanced_pred_label = label
                    break
            
            # Track classification results
            if basic_pred_idx == true_idx:
                source_stats[source_type]["correctly_classified_by_basic"] += 1
                # print(f"  {img_name} - Basic model correctly classified as {basic_pred_label}")
                continue
                
            if enhanced_pred_idx != true_idx:
                source_stats[source_type]["misclassified_by_enhanced"] += 1
                # print(f"  {img_name} - Basic: {basic_pred_label} (wrong), Enhanced: {enhanced_pred_label} (also wrong, true: {true_label})")
                continue
            
            # If basic model misclassifies but enhanced gets it right
            if basic_pred_idx != true_idx and enhanced_pred_idx == true_idx:
                print(f"  {img_name} - Basic: {basic_pred_label} (wrong), Enhanced: {enhanced_pred_label} (correct)")
                
                found_samples_per_label[true_label].append({
                    'img_path': img_path,
                    'true_label': true_label,
                    'basic_pred_label': basic_pred_label,
                    'true_idx': true_idx,
                    'basic_pred_idx': basic_pred_idx
                })
                # Update found count for this source type
                source_stats[source_type]["found"] += 1
                print(f"Found sample for {true_label}: {img_path} ({len(found_samples_per_label[true_label])}/{samples_per_label})")
    
    # Process each sample for visualization
    total_samples = sum(len(samples) for samples in found_samples_per_label.values())
    print(f"\nTotal samples found: {total_samples}")
    print("Samples per label:")
    for label, samples in found_samples_per_label.items():
        print(f"  {label}: {len(samples)}")
    
    # Print statistics by source type
    print("\nStatistics by source type:")
    for source, stats in source_stats.items():
        if stats["total"] > 0:
            percentage = (stats["found"] / stats["total"]) * 100
            print(f"  {source}: Found {stats['found']} out of {stats['total']} ({percentage:.2f}%)")
        else:
            print(f"  {source}: No samples processed")
    
    # Process each sample for visualization
    for label, samples in found_samples_per_label.items():
        if "AI_LD" in label or "AI_SD" in label:
            continue
        for i, sample in enumerate(samples):
            sample_name = os.path.basename(sample['img_path']).split('.')[0]
            sample_label = sample['true_label']
            sample_pred = sample['basic_pred_label']
            
            # Create output directories
            output_dir = os.path.join(base_output_dir, f"{sample_label}_vs_{sample_pred}", sample_name)
            basic_dir = os.path.join(output_dir, 'basic')
            enhanced_dir = os.path.join(output_dir, 'enhanced')
            
            os.makedirs(basic_dir, exist_ok=True)
            os.makedirs(enhanced_dir, exist_ok=True)
            
            print(f"\nProcessing sample {i+1}/{len(samples)} for {label}: {sample['img_path']}")
            print(f"True label: {sample['true_label']}")
            print(f"Basic model predicted: {sample['basic_pred_label']}")
            
            img_tensor, orig_img = preprocess_image(sample['img_path'], test_transform)
            img_tensor = img_tensor.to(device)
            
            # Get enhanced model prediction
            with torch.no_grad():
                enhanced_outputs = enhanced_model(img_tensor)
                _, enhanced_pred_idx = torch.max(enhanced_outputs, 1)
                enhanced_pred_idx = enhanced_pred_idx.item()
                
                # Get enhanced prediction label
                enhanced_pred_label = None
                for label, idx in test_dataset.label_to_idx.items():
                    if idx == enhanced_pred_idx:
                        enhanced_pred_label = label
                        break
            
            # 1. Visualize filter weights for early layers
            # Basic model
            visualize_filters(
                model=basic_model,
                layer_name='conv_block1',
                num_filters=16,
                save_path=os.path.join(basic_dir, 'conv_block1_filters.png'),
                model_type="basic",
                true_label=sample['true_label'],
                pred_label=sample['basic_pred_label']
            )
            
            # Enhanced model
            visualize_filters(
                model=enhanced_model,
                layer_name='conv_block1',
                num_filters=16,
                save_path=os.path.join(enhanced_dir, 'conv_block1_filters.png'),
                model_type="enhanced",
                true_label=sample['true_label'],
                pred_label=enhanced_pred_label
            )
            
            # 2. Visualize feature maps for different layers
            # Basic model
            basic_feature_viz = FeatureMapVisualizer(basic_model, model_type="basic")
            for layer_name in ['conv_block1', 'conv_block2', 'conv_block3', 'conv_block4']:
                basic_feature_viz.visualize_feature_maps(
                    img_tensor=img_tensor,
                    layer_name=layer_name,
                    num_features=16,
                    save_path=os.path.join(basic_dir, f'{layer_name}_features.png'),
                    true_label=sample['true_label'],
                    pred_label=sample['basic_pred_label']
                )
            
            # Enhanced model
            enhanced_feature_viz = FeatureMapVisualizer(enhanced_model, model_type="enhanced")
            for layer_name in ['conv_block1', 'conv_block2', 'conv_block3', 'conv_block4']:
                enhanced_feature_viz.visualize_feature_maps(
                    img_tensor=img_tensor,
                    layer_name=layer_name,
                    num_features=16,
                    save_path=os.path.join(enhanced_dir, f'{layer_name}_features.png'),
                    true_label=sample['true_label'],
                    pred_label=enhanced_pred_label
                )
            
            # 3. Visualize SPP and CIM features for enhanced model
            # SPP features
            try:
                enhanced_feature_viz.visualize_feature_maps(
                    img_tensor=img_tensor,
                    layer_name='spp',
                    num_features=16,
                    save_path=os.path.join(enhanced_dir, 'spp_features.png'),
                    true_label=sample['true_label'],
                    pred_label=enhanced_pred_label
                )
            except Exception as e:
                print(f"Error visualizing SPP features: {e}")
            
            # CIM features
            for cim_idx in range(1, 5):
                try:
                    enhanced_feature_viz.visualize_feature_maps(
                        img_tensor=img_tensor,
                        layer_name=f'cim{cim_idx}',
                        num_features=16,
                        save_path=os.path.join(enhanced_dir, f'cim{cim_idx}_features.png'),
                        true_label=sample['true_label'],
                        pred_label=enhanced_pred_label
                    )
                except Exception as e:
                    print(f"Error visualizing CIM{cim_idx} features: {e}")
    
    print("\nAll visualizations completed!")
    print(f"Found and processed {total_samples} samples that were misclassified by the basic model but correctly classified by the enhanced model.")
    print(f"Results saved in: {base_output_dir}")

if __name__ == "__main__":
    main()