import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
from collections import defaultdict
from model_basic import BasicCNN, ArtDataset
from model_attention import BasicCNNWithAttention  # 导入带注意力的模型

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Register hooks
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        # Register forward and backward hooks
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        
    def __call__(self, x, class_idx=None):
        # Forward propagation
        b, c, h, w = x.size()
        logits = self.model(x)
        
        if class_idx is None:
            # If no class is specified, use the predicted class
            class_idx = logits.argmax(dim=1).item()
        
        # Backward propagation
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate weights - use global average pooling to get importance of each channel
        gradients = self.gradients
        # Ensure gradients are not None
        if gradients is None:
            print("Warning: Gradients are None, possibly due to model structure or hook registration issues")
            return np.zeros((int(h), int(w))), class_idx
            
        # Get the number of channels in the feature map
        feature_channels = self.activations.size(1)
        
        # Improved weight calculation method: global average pooling of gradients for each channel
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Ensure the shape of weights matches the number of channels in the feature map
        if weights.size(1) != feature_channels:
            print(f"Warning: Weight channel count ({weights.size(1)}) does not match feature map channel count ({feature_channels}), adjusting")
            # If the number of channels in weights doesn't match the feature map, create new weights using the feature map's channel count
            weights = weights.repeat(1, feature_channels // weights.size(1), 1, 1)
        
        # Improved CAM generation method
        # Multiply weights directly with feature maps, maintaining dimensional consistency
        weighted_activations = weights * self.activations
        
        # Sum along the channel dimension
        cam = torch.sum(weighted_activations, dim=1).squeeze()
        
        # Apply ReLU to ensure only positive contributions are considered
        cam = F.relu(cam)
        
        # Normalize
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # Resize to input image dimensions
        cam = cam.cpu().numpy()
        
        # Ensure dimensions are valid positive integers
        target_width = max(1, int(w))
        target_height = max(1, int(h))
        
        # Ensure cam is a valid non-empty array
        if cam.size == 0 or np.isnan(cam).any():
            print("Warning: Generated CAM is invalid, returning empty heatmap")
            return np.zeros((target_height, target_width)), class_idx
            
        cam = cv2.resize(cam, (target_width, target_height))
        
        return cam, class_idx

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

def show_basic_model_cam(img, cam_basic, save_path=None, title_text=""):
    """Display CAM heatmap for only the basic model"""
    # Convert PIL image to numpy array
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Ensure image is in RGB format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Resize heatmap to match image
    cam_basic = cv2.resize(cam_basic, (img.shape[1], img.shape[0]))
    
    # Apply heatmap for basic model
    heatmap_basic = cv2.applyColorMap(np.uint8(255 * cam_basic), cv2.COLORMAP_JET)
    heatmap_basic = cv2.cvtColor(heatmap_basic, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap and original image
    superimposed_basic = heatmap_basic * 0.4 + img * 0.6
    superimposed_basic = np.uint8(superimposed_basic)
    
    # Create figure with only basic model visualization
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Basic model heatmap
    axs[1].imshow(heatmap_basic)
    axs[1].set_title('Basic Model Heatmap')
    axs[1].axis('off')
    
    # Basic model superimposed
    axs[2].imshow(superimposed_basic)
    axs[2].set_title('Basic Model Superimposed')
    axs[2].axis('off')
    
    plt.suptitle(title_text, fontsize=14, wrap=True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Basic model heatmap saved to: {save_path}")
    
    plt.close()
    
    return superimposed_basic

def show_comparison_cam(img, cam_basic, cam_attention, save_path=None, title_text=""):
    """Display CAM heatmap comparing basic and attention models"""
    # Convert PIL image to numpy array
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Ensure image is in RGB format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Resize heatmaps to match image
    cam_basic = cv2.resize(cam_basic, (img.shape[1], img.shape[0]))
    cam_attention = cv2.resize(cam_attention, (img.shape[1], img.shape[0]))
    
    # Apply heatmap for basic model
    heatmap_basic = cv2.applyColorMap(np.uint8(255 * cam_basic), cv2.COLORMAP_JET)
    heatmap_basic = cv2.cvtColor(heatmap_basic, cv2.COLOR_BGR2RGB)
    
    # Apply heatmap for attention model
    heatmap_attention = cv2.applyColorMap(np.uint8(255 * cam_attention), cv2.COLORMAP_JET)
    heatmap_attention = cv2.cvtColor(heatmap_attention, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmaps on original image
    superimposed_basic = heatmap_basic * 0.4 + img * 0.6
    superimposed_basic = np.uint8(superimposed_basic)
    
    superimposed_attention = heatmap_attention * 0.4 + img * 0.6
    superimposed_attention = np.uint8(superimposed_attention)
    
    # Create figure with comparison visualization
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Basic model heatmap
    axs[1].imshow(heatmap_basic)
    axs[1].set_title('Basic Model Heatmap')
    axs[1].axis('off')
    
    # Attention model heatmap
    axs[2].imshow(heatmap_attention)
    axs[2].set_title('Attention Model Heatmap')
    axs[2].axis('off')
    
    # Attention model superimposed
    axs[3].imshow(superimposed_attention)
    axs[3].set_title('Attention Model\nSuperimposed')
    axs[3].axis('off')
    
    plt.suptitle(title_text, fontsize=14, wrap=True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Comparison heatmap saved to: {save_path}")
    
    plt.close()
    
    return superimposed_attention

def get_class_name(idx, dataset):
    """Get class name"""
    for label, i in dataset.label_to_idx.items():
        if i == idx:
            return label
    return f"Unknown (idx: {idx})"

def get_source_type(label):
    """Extract source type (AI_SD, AI_LD, or human) from label"""
    if label.startswith('AI_SD_'):
        return 'AI_SD'
    elif label.startswith('AI_LD_'):
        return 'AI_LD'
    elif label.startswith('human_'):
        return 'human'
    else:
        return 'unknown'

def extract_style_name(label):
    """Extract just the style part from a label, e.g., 'AI_SD_impressionism' -> 'impressionism'"""
    if '_' in label:
        parts = label.split('_')
        if label.startswith('AI_'):
            return '_'.join(parts[2:])
        elif label.startswith('human_'):
            return '_'.join(parts[1:])
    return label

def main():
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
    
    # Load the dataset to get class information
    test_dataset = ArtDataset(test_data_dir, transform=test_transform)
    
    # Load both models
    print("Loading basic model...")
    basic_model = BasicCNN(num_classes=len(test_dataset.combined_labels))
    basic_model.load_state_dict(torch.load('best_model.pth', map_location=device))
    basic_model.to(device)
    basic_model.eval()
    
    print("Loading attention model...")
    attention_model = BasicCNNWithAttention(num_classes=len(test_dataset.combined_labels))
    attention_model.load_state_dict(torch.load('best_model_with_attention.pth', map_location=device))
    attention_model.to(device)
    attention_model.eval()
    
    # Initialize Grad-CAM for both models
    grad_cam_basic = GradCAM(basic_model, basic_model.conv_block4[0])
    grad_cam_attention = GradCAM(attention_model, attention_model.conv_block4[0])
    
    # Create directory to save heatmaps
    output_dir = 'diverse_styles_gradcam'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for basic and comparison outputs
    basic_dir = os.path.join(output_dir, 'basic_only')
    comparison_dir = os.path.join(output_dir, 'comparison')
    os.makedirs(basic_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Select images for visualization
    test_images_dir = test_data_dir
    style_dirs = os.listdir(test_images_dir)
    
    # Target samples per source type
    samples_per_source = 100
    
    # Maximum samples per art style within each source type
    max_samples_per_style = 10
    
    # Group directories by source type
    source_dirs = {
        'AI_LD': [],
        'AI_SD': [],
        'human': []
    }
    
    # First, categorize all directories by source type
    for style_dir in style_dirs:
        if style_dir.startswith('AI_LD_'):
            source_dirs['AI_LD'].append(style_dir)
        elif style_dir.startswith('AI_SD_'):
            source_dirs['AI_SD'].append(style_dir)
        else:
            # Assume it's human if not AI_LD or AI_SD
            source_dirs['human'].append(style_dir)
    
    # Process each source type separately
    found_samples = {
        'AI_LD': 0,
        'AI_SD': 0,
        'human': 0
    }
    
    # Track art styles per source to ensure diversity
    style_counts = {
        'AI_LD': defaultdict(int),
        'AI_SD': defaultdict(int),
        'human': defaultdict(int)
    }
    
    # Counter for output filenames
    sample_counter = 0
    
    # Process each source type
    for source_type, dirs in source_dirs.items():
        print(f"\nProcessing {source_type} source type...")
        
        # Process each directory for this source type
        for style_dir in dirs:
            if found_samples[source_type] >= samples_per_source:
                print(f"Found {samples_per_source} samples for {source_type}, moving to next source type")
                break
                
            style_path = os.path.join(test_images_dir, style_dir)
            if not os.path.isdir(style_path):
                continue
                
            # Get all images for this style
            images = [f for f in os.listdir(style_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                continue
            
            # Extract true label from folder name
            style = style_dir
            true_label = style_dir
            true_idx = None
            
            # Handle different label formats
            if style_dir.startswith('AI_LD_') or style_dir.startswith('AI_SD_'):
                # For AI generated art, use the folder name directly
                for label, idx in test_dataset.label_to_idx.items():
                    if label == true_label:
                        true_idx = idx
                        break
            else:
                # For human art, need to add 'human_' prefix
                # First normalize style name if needed
                if style in test_dataset.normalize_style:
                    style = test_dataset.normalize_style[style]
                
                # Try with human_ prefix
                human_label = f'human_{style}'
                for label, idx in test_dataset.label_to_idx.items():
                    if label == human_label:
                        true_label = human_label  # Update true_label to match the format in label_to_idx
                        true_idx = idx
                        break
            
            if true_idx is None:
                print(f"Could not find label index for {style_dir}, skipping this folder")
                continue
            
            # Extract the pure art style from the label
            art_style = extract_style_name(true_label)
            
            # Skip if we already have enough samples of this art style
            if style_counts[source_type][art_style] >= max_samples_per_style:
                print(f"Already have {max_samples_per_style} samples of {art_style} for {source_type}, skipping")
                continue
            
            # Look for samples where basic model is wrong but attention model is correct
            for img_file in images:
                if found_samples[source_type] >= samples_per_source:
                    break
                    
                # Skip if we already have enough samples of this art style
                if style_counts[source_type][art_style] >= max_samples_per_style:
                    break
                    
                img_path = os.path.join(style_path, img_file)
                
                # Preprocess image
                img_tensor, orig_img = preprocess_image(img_path, test_transform)
                img_tensor = img_tensor.to(device)
                
                # Get model predictions
                with torch.no_grad():
                    # Basic model prediction
                    basic_outputs = basic_model(img_tensor)
                    _, basic_pred_idx = torch.max(basic_outputs, 1)
                    basic_pred_idx = basic_pred_idx.item()
                    
                    # Attention model prediction
                    attention_outputs = attention_model(img_tensor)
                    _, attention_pred_idx = torch.max(attention_outputs, 1)
                    attention_pred_idx = attention_pred_idx.item()
                
                # Check if basic model is wrong but attention model is correct
                if basic_pred_idx != true_idx and attention_pred_idx == true_idx:
                    # Increment counts
                    found_samples[source_type] += 1
                    style_counts[source_type][art_style] += 1
                    sample_counter += 1
                    
                    print(f"Found sample {found_samples[source_type]}/{samples_per_source} for {source_type}, art style: {art_style}, count: {style_counts[source_type][art_style]}/{max_samples_per_style}")
                    print(f"Image path: {img_path}")
                    sample_name = os.path.splitext(os.path.basename(img_path))[0]
                    
                    # Get class names
                    basic_pred_class = get_class_name(basic_pred_idx, test_dataset)
                    attention_pred_class = get_class_name(attention_pred_idx, test_dataset)
                    
                    print(f"True class: {true_label} (index: {true_idx})")
                    print(f"Basic model prediction: {basic_pred_class} (Wrong)")
                    print(f"Attention model prediction: {attention_pred_class} (Correct)")
                    
                    # Generate Grad-CAM for both models
                    cam_basic, _ = grad_cam_basic(img_tensor, class_idx=basic_pred_idx)
                    cam_attention, _ = grad_cam_attention(img_tensor, class_idx=attention_pred_idx)
                    
                    # Create title in the required format: "True Label: ... Basic model: ... Model with Attention: ..."
                    title_text = f"True Label: {true_label} Basic model: {basic_pred_class} Model with Attention: {attention_pred_class}"
                    
                    title_basic = f"True Label: {true_label} Basic model: {basic_pred_class}"
                    
                    # 1. Generate and save basic model only visualization
                    basic_save_path = os.path.join(basic_dir, f"{sample_counter:02d}_{source_type}_{art_style}_{sample_name}_basic_only.png")
                    show_basic_model_cam(orig_img, cam_basic, basic_save_path, title_basic)
                    
                    # 2. Generate and save comparison visualization
                    comparison_save_path = os.path.join(comparison_dir, f"{sample_counter:02d}_{source_type}_{art_style}_{sample_name}_comparison.png")
                    show_comparison_cam(orig_img, cam_basic, cam_attention, comparison_save_path, title_text)
                    
                    # Save raw CAM data for further analysis
                    basic_cam_path = os.path.join(output_dir, f"{sample_counter:02d}_{source_type}_{art_style}_{sample_name}_basic_cam.npy")
                    attention_cam_path = os.path.join(output_dir, f"{sample_counter:02d}_{source_type}_{art_style}_{sample_name}_attention_cam.npy")
                    np.save(basic_cam_path, cam_basic)
                    np.save(attention_cam_path, cam_attention)
    
    # Print summary of found samples
    print("\nGrad-CAM visualization complete!")
    for source_type, count in found_samples.items():
        print(f"{source_type}: {count}/{samples_per_source} samples found")
        
        # Print art style distribution
        print(f"  Art style distribution for {source_type}:")
        for style, count in style_counts[source_type].items():
            print(f"    {style}: {count} samples")
    
    print(f"Total samples processed: {sample_counter}")
    print(f"Basic model outputs saved to: {basic_dir}")
    print(f"Comparison outputs saved to: {comparison_dir}")

if __name__ == "__main__":
    main()