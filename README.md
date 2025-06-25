# TML25 Assignment 2: Model Stealing Against B4B Defense

## Overview

This repository contains our implementation of a model stealing attack against an encoder protected by the B4B (Bucks for Buckets) defense mechanism. The goal is to create a stolen copy of a victim encoder that minimizes the L2 distance between output representations on private test images.

## Team Information
- **Assignment**: TML25 Assignment 2 - Model Stealing
- **Defense Target**: B4B (Bucks for Buckets) protected encoder
- **Objective**: Minimize L2 distance between stolen and victim model representations


## Key Components

### 1. Enhanced Model Architecture (`ImprovedStolenEncoder`)

Our stolen model features:
- **Residual CNN Architecture**: Custom ResNet-inspired design with 4 residual blocks
- **Progressive Channel Expansion**: 64 → 128 → 256 → 512 → 512 channels
- **Advanced Projection Head**: Multi-layer projection with batch normalization and dropout
- **ONNX Compatibility**: Designed specifically for ONNX export requirements
- **B4B Defense Awareness**: Architecture designed to handle noisy representations

**Key Features:**
- Input: 3×32×32 RGB images
- Output: 1024-dimensional representations
- Residual connections with proper skip connections
- Batch normalization for stable training
- Dropout for regularization

### 2. B4B-Robust Training Pipeline (`B4BRobustTrainer`)

Our training approach specifically targets B4B defense weaknesses:

#### Advanced Loss Function
- **Multi-component Loss**: Combines MSE, Cosine Similarity, Huber, L1, and Correlation losses
- **Adaptive Weighting**: Epoch-dependent loss component weighting
- **Robustness to Noise**: Huber loss component handles B4B-induced noise
- **Relationship Preservation**: Correlation loss maintains representation structure

#### Enhanced Training Strategy
- **Curriculum Learning**: Gradual increase in correlation loss importance
- **Advanced Optimization**: AdamW optimizer with cosine annealing warm restarts
- **Gradient Clipping**: Adaptive gradient norm clipping
- **Early Stopping**: Multi-criteria early stopping with patience
- **Comprehensive Validation**: Multiple similarity metrics for model selection

#### Data Preprocessing
- **Robust Augmentation**: RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop
- **Normalization**: ImageNet statistics normalization
- **Channel Handling**: Automatic grayscale to RGB conversion
- **Train/Validation Split**: 80/20 split with stratification

### 3. API Integration

#### Model Stealing Workflow
1. **API Request**: Request new encoder API with authentication token
2. **Query Execution**: Send batches of 1000 images to victim encoder
3. **Response Processing**: Extract and store noisy representations
4. **Data Management**: Save stolen data for training

#### Key Constraints
- **Query Limit**: 100k queries maximum per API session
- **Batch Size**: 1000 images per query
- **Input Format**: Base64-encoded PNG images
- **Output Format**: 1024-dimensional float vectors

### 4. ONNX Export and Submission

#### Export Pipeline
- **Model Conversion**: PyTorch → ONNX with opset version 11
- **Optimization**: Constant folding and training mode optimization
- **Validation**: Comprehensive output shape and range validation
- **Compatibility**: Dynamic batch size support for evaluation

#### Submission Process
- **Authentication**: Token and seed-based authentication
- **Model Upload**: ONNX file submission to evaluation endpoint
- **Result Tracking**: Automatic result saving and logging

## Implementation Details

### B4B Defense Counter-Strategies

Our approach specifically addresses B4B defense mechanisms:

1. **Noise Robustness**: Huber loss handles representation noise
2. **Relationship Preservation**: Correlation loss maintains embedding structure
3. **Multi-metric Optimization**: Combined loss prevents overfitting to single metric
4. **Advanced Architecture**: Residual connections improve gradient flow through noise

### Training Configuration

```python
# Model Architecture
- Input Channels: 3 (RGB)
- Output Dimensions: 1024
- Residual Blocks: 4 layers
- Channel Progression: 64→128→256→512→512

# Training Parameters
- Epochs: 150 (with early stopping)
- Learning Rate: 0.001 (with cosine annealing)
- Batch Size: 64
- Optimizer: AdamW (weight_decay=1e-4)
- Loss Components: MSE(0.4) + Cosine(0.2) + Huber(0.2) + L1(0.1) + Correlation(0.1)
```

### Loss Function Components

1. **MSE Loss (40%)**: Primary reconstruction loss
2. **Cosine Similarity Loss (20%)**: Direction preservation
3. **Huber Loss (20%)**: Robust to B4B noise
4. **L1 Loss (10%)**: Sparsity encouragement
5. **Correlation Loss (10%)**: Relationship structure preservation

## Usage Instructions

### 1. Setup Environment

```bash
pip install torch torchvision onnxruntime requests numpy scikit-learn
```

### 2. Configure Authentication

```python
TOKEN = "your_token_here"  # Replace with your actual token
```

### 3. Execute Full Pipeline

```python
# Train the enhanced model
train_enhanced_model()

# Export and submit
main()
```

### 4. Manual Steps (if needed)

```python
# Load and train model
images, representations = load_stolen_data()
model = ImprovedStolenEncoder(input_dim=3, output_dim=1024)
trainer = B4BRobustTrainer(model)
train_loader, val_loader = trainer.create_dataloader(images, representations)
history = trainer.train(train_loader, val_loader, epochs=150)

# Export to ONNX
export_model_to_onnx()

# Submit for evaluation
submit_for_evaluation()
```

## Performance Monitoring

The code provides comprehensive training monitoring:

- **Real-time Loss Tracking**: All loss components tracked separately
- **Validation Metrics**: Cosine similarity, L2 distance, multi-component loss
- **Model Checkpointing**: Best model saved based on validation performance
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: Cosine annealing with warm restarts

## Key Innovations

### 1. B4B-Specific Architecture
- Residual connections handle gradient flow through noisy representations
- Advanced projection head with multiple bottlenecks
- Batch normalization stabilizes training with noisy targets

### 2. Robust Loss Design
- Multi-component loss specifically designed for B4B defense
- Adaptive weighting based on training progress
- Correlation preservation maintains embedding space structure

### 3. Advanced Training Strategy
- Curriculum learning with epoch-dependent loss weighting
- Sophisticated optimization with cosine annealing warm restarts
- Multi-criteria early stopping prevents overfitting

### 4. ONNX Compatibility
- Architecture designed for seamless ONNX export
- Validation pipeline ensures compatibility
- Dynamic batch size support for evaluation

## Expected Results

Based on our enhanced approach:
- **Improved Robustness**: Better handling of B4B-induced noise
- **Preserved Relationships**: Maintained semantic relationships in stolen representations
- **Stable Training**: Robust convergence even with noisy targets
- **High Fidelity**: Minimized L2 distance on private test set

## Troubleshooting

### Common Issues

1. **ONNX Export Failures**: Ensure model is on CPU before export
2. **Dimension Mismatches**: Verify input (3×32×32) and output (1024) dimensions
3. **Authentication Errors**: Check token and seed values
4. **Memory Issues**: Reduce batch size if encountering OOM errors

### Debugging Tips

- Check model output shapes with dummy inputs
- Validate ONNX model before submission
- Monitor loss components during training
- Verify API session data is properly saved

## References

- B4B Defense Paper: [Bucks for Buckets defense mechanism]
- Model Stealing Literature: [Relevant papers on model extraction]
- PyTorch ONNX Documentation: [Official export guidelines]

## Contact

For questions or issues with this implementation, please refer to the course materials or contact the teaching assistants.
