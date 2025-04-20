# Vision with Insight: Improving CNN Architectures Through Interpretability for Artistic Style Detection

Deep learning models often function as black boxes, limiting understanding of their decision-making processes and opportunities for targeted improvements. This project specifically implement a systematic approach CNN refinement guided by interpretability techniques for classifying artistic styles.

## Features

- Art style classification across multiple artistic movements
- Supports detection of AI-generated vs human-created art
- Multiple CNN model architectures:
  - Basic CNN (`model_basic.py`)
  -CNN with CBAM (`model_attention.py`)
  - Enhanced CNN with SPP/CIM (`model_filter.py`)
  - Ultimate model with both CBAM and SPP/CIM (`model_ultimate.py`)

## Visualization Techniques

- Grad-CAM for visualizing model focus areas
- Convolutional filter visualization
- Feature map visualization
- Filter activation through gradient ascent

## Data Structure

The system is designed to work with art images organized by:
- Art style categories (impressionism, renaissance, surrealism, etc.)
- Source type (AI_SD_, AI_LD_, human_)
You can download the dataset from the kaggle website: https://www.kaggle.com/datasets/ravidussilva/real-ai-art

## Requirements

```
pandas==1.5.3
matplotlib==3.7.1
scikit-learn==1.2.2
opencv-python==4.7.0.72
tqdm==4.65.0
Pillow==9.5.0
seaborn==0.12.2
torchsummary==1.5.1
torch==2.0.1+cu118
torchvision==0.15.2+cu118
torchaudio==2.0.2+cu118
numpy>=1.22.0
```

## Usage

The project contains multiple scripts that can be run independently:
- Train and evaluate models with the main functions in model scripts
- Generate visualizations using `Grad_CAM.py` and `filter_visualization.py` 