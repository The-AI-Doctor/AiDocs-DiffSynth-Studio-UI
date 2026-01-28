# DiffSynth-Studio UI

A comprehensive Gradio-based web interface for managing DiffSynth-Studio models and training custom LoRA adapters for the Z-Image model.

## Features

### 🔧 Installation
- **One-Click Setup**: Install DiffSynth-Studio package automatically
- **Git LFS Support**: Ensure Git LFS is properly configured for large model downloads
- **Dependency Management**: Automatic pip installation of all required packages

### ⬇️ Model Download
- **Z-Image-i2 Repository**: Clone the official model repository from ModelScope
- **Z-Image-i2L Weights**: Download pre-trained model weights
- **Automatic Git LFS**: Handles large file transfers seamlessly
- **Progress Tracking**: Real-time download status updates

### 🎯 LoRA Training
- **Configuration Generator**: Create training configs with customizable hyperparameters
- **Model Reasoning**: Integrated understanding of Z-Image architecture and LoRA methodology
- **Flexible Parameters**:
  - Rank: Control adapter complexity (4-128)
  - Alpha: LoRA scaling factor (8-256)
  - Learning Rate: Customizable optimizer settings
  - Batch Size: Memory-efficient training options
  - Epochs & Steps: Fine-grained training control

### 📊 System Status
- **Monitor Setup**: Check installation status and downloaded models
- **LoRA Management**: View all created LoRA configurations
- **Dependency Verification**: Confirm Git, Git LFS, and Python setup

## Installation

### Prerequisites
- Python 3.8+
- Git (with Git LFS support)
- 8GB+ RAM (16GB+ recommended)
- NVIDIA GPU with CUDA support (for faster training)

### Quick Start

#### Windows
```bash
# Double-click run.bat or run from PowerShell/Command Prompt
run.bat
```

#### Linux/macOS
```bash
# Make script executable
chmod +x run.sh

# Run the launcher
./run.sh
```

#### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the UI
python app.py
```

## Usage

### 1. Installation Phase
1. Open http://localhost:7860 in your browser
2. Go to the **Installation** tab
3. Click "Setup Git LFS" to initialize Git LFS
4. Click "Install DiffSynth-Studio" to install the package

### 2. Download Models
1. Navigate to the **Model Download** tab
2. Click "Download Z-Image-i2" to clone the model repository
3. Click "Download Z-Image-i2L Weights" to get pre-trained weights
4. Monitor the status for completion

### 3. Configure LoRA Training
1. Go to the **LoRA Training** tab
2. Enter your LoRA name (e.g., "my_anime_style")
3. Adjust hyperparameters:
   - **Rank**: 32 (good default), increase for more capacity
   - **Alpha**: 64 (typically 2x rank)
   - **Learning Rate**: 0.0001 (standard for LoRA)
   - **Batch Size**: 1 (adjust based on VRAM)
   - **Epochs**: 100 (increase for more training)
4. Click "Create Configuration"
5. Use the generated config file for your training script

### 4. Monitor Status
1. Open the **Status** tab to view:
   - Directory structure
   - Downloaded models
   - Created LoRA configurations
   - Git/Git LFS status

## Model Reasoning

### Z-Image-i2 Architecture
- **Type**: Image-to-Image Diffusion Model
- **Backbone**: UNet-based architecture
- **Input**: Image + prompt/conditioning
- **Output**: Transformed/generated image

### LoRA Adaptation Strategy
- **Target Modules**: Attention layers (to_q, to_k, to_v, to_out)
- **Rank**: Controls expressiveness of adaptation (32-64 typical)
- **Alpha**: Balances between base model and LoRA influence
- **Advantage**: 1-5% additional parameters vs full fine-tuning

### Training Process
1. **Freeze** base model weights
2. **Inject** LoRA adapters in attention layers
3. **Train** only adapter parameters
4. **Save** lightweight LoRA weights (~50-100MB)
5. **Inference** merges LoRA with base model

## Configuration Details

Each LoRA configuration includes:

```json
{
  "lora_config": {
    "rank": 32,
    "alpha": 64,
    "target_modules": ["to_k", "to_v", "to_q", "to_out"],
    "bias": "none"
  },
  "training": {
    "learning_rate": 0.0001,
    "num_epochs": 100,
    "batch_size": 1,
    "optimizer": "adamw_8bit",
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0
  },
  "inference": {
    "guidance_scale": 7.5,
    "num_inference_steps": 50
  }
}
```

## Directory Structure

```
~/DiffSynth-Studio/
├── models/
│   ├── Z-Image-i2/          # Model repository
│   └── Z-Image-i2L/         # Pre-trained weights
└── loras/
    └── <lora_name>/
        ├── config.json       # Training configuration
        ├── dataset/          # Training data directory
        └── checkpoints/      # Model checkpoints
```

## Recommended Hyperparameters

### For General Fine-tuning
```
Rank: 32
Alpha: 64
Learning Rate: 0.0001
Batch Size: 2-4
Epochs: 50-100
```

### For Specific Styles
```
Rank: 64
Alpha: 128
Learning Rate: 0.00005
Batch Size: 1-2
Epochs: 100-200
```

### For Limited VRAM (<8GB)
```
Rank: 16
Alpha: 32
Learning Rate: 0.0001
Batch Size: 1
Epochs: 50
```

## Troubleshooting

### Git LFS Issues
```powershell
# Reinstall Git LFS
git lfs install
git lfs pull
```

### Download Failures
- Ensure stable internet connection
- Try manual git clone:
```powershell
git clone https://www.modelscope.cn/DiffSynth-Studio/Z-Image-i2.git
```

### Memory Issues
- Reduce batch size to 1
- Lower rank to 16-32
- Enable gradient checkpointing in training script

### CUDA Errors
```powershell
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA 11.8 compatible PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Performance Tips

1. **GPU Acceleration**: Uses CUDA if available
2. **8-bit Optimization**: Reduces memory usage by ~50%
3. **Gradient Checkpointing**: Saves memory at slight speed cost
4. **Mixed Precision**: FP16 training reduces memory and speeds up training

## File Structure

```
AiDocs-DiffSynth-Studio-UI/
├── app.py                # Main Gradio application
├── requirements.txt      # Python dependencies
├── run.bat              # Windows launcher
├── run.sh               # Linux/macOS launcher
└── README.md            # This file
```

## API Reference

### DiffSynthStudioUI Class

#### Installation
- `install_diffsynth()` - Install DiffSynth-Studio package
- `install_git_lfs()` - Initialize Git LFS

#### Model Download
- `download_z_image_model(model_version)` - Clone model repository
- `download_model_weights(model_path)` - Download pre-trained weights

#### LoRA Training
- `create_lora_config(...)` - Generate training configuration
- `list_loras()` - List all created LoRA configs
- `get_status()` - Get system status

## Resources

- **DiffSynth-Studio**: https://www.modelscope.cn/DiffSynth-Studio
- **Z-Image-i2**: https://www.modelscope.cn/models/DiffSynth-Studio/Z-Image-i2L
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **ModelScope**: https://www.modelscope.cn

## Notes

- All model downloads are stored in `~/DiffSynth-Studio/models/`
- LoRA configurations are stored in `~/DiffSynth-Studio/loras/`
- Large model files (2-10GB) may take 10-30 minutes to download
- First-time setup may take 20-30 minutes including dependency installation

## License

This UI is provided as-is. Please refer to DiffSynth-Studio and Z-Image model licenses for usage terms.

## Support

For issues with:
- **DiffSynth-Studio**: See official repo
- **LoRA Training**: Consult PEFT documentation
- **Model Downloads**: Check ModelScope documentation

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-27  
**Status**: Production Ready
