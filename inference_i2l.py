"""
Z-Image-i2L Inference Script
Converts reference style images into a LoRA model using the Z-Image-i2L pipeline
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional
import torch
from PIL import Image
from safetensors.torch import save_file

try:
    from diffsynth.pipelines.z_image import (
        ZImagePipeline, ModelConfig,
        ZImageUnit_Image2LoRAEncode, ZImageUnit_Image2LoRADecode
    )
except ImportError:
    print("Error: diffsynth package not found.")
    print("Please install it with: pip install -e DiffSynth-Studio/DiffSynth-Studio-repo")
    exit(1)


# Helper functions for safe cross-platform output
def safe_print(*args, **kwargs):
    """Print with safe encoding handling for Windows"""
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        # Fallback for Windows cmd/PowerShell encoding issues
        text = " ".join(str(arg) for arg in args)
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        print(text, flush=True)


class ZImageI2LGenerator:
    """Generator for creating LoRA models from reference images using Z-Image-i2L"""
    
    def __init__(self, device: str = None, torch_dtype=None):
        # Detect CUDA availability
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Select dtype based on device
        if torch_dtype is None:
            if device == 'cuda':
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype
        self.pipe = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the Z-Image-i2L pipeline with all required models"""
        safe_print("Loading Z-Image-i2L pipeline...")
        safe_print("This may take a few minutes on first run (downloading models)...")
        
        try:
            # VRAM configuration for optimal memory usage
            vram_config = {
                "offload_dtype": self.torch_dtype,
                "offload_device": self.device,
                "onload_dtype": self.torch_dtype,
                "onload_device": self.device,
                "preparing_dtype": self.torch_dtype,
                "preparing_device": self.device,
                "computation_dtype": self.torch_dtype,
                "computation_device": self.device,
            }
            
            safe_print(f"\nLoading Z-Image base model...")
            safe_print(f"Device: {self.device} | Data type: {self.torch_dtype}")
            
            # Load pipeline with all required model components
            self.pipe = ZImagePipeline.from_pretrained(
                torch_dtype=self.torch_dtype,
                device=self.device,
                model_configs=[
                    ModelConfig(
                        model_id="Tongyi-MAI/Z-Image",
                        origin_file_pattern="transformer/*.safetensors",
                        **vram_config
                    ),
                    ModelConfig(
                        model_id="Tongyi-MAI/Z-Image-Turbo",
                        origin_file_pattern="text_encoder/*.safetensors"
                    ),
                    ModelConfig(
                        model_id="Tongyi-MAI/Z-Image-Turbo",
                        origin_file_pattern="vae/diffusion_pytorch_model.safetensors"
                    ),
                    ModelConfig(
                        model_id="DiffSynth-Studio/General-Image-Encoders",
                        origin_file_pattern="SigLIP2-G384/model.safetensors"
                    ),
                    ModelConfig(
                        model_id="DiffSynth-Studio/General-Image-Encoders",
                        origin_file_pattern="DINOv3-7B/model.safetensors"
                    ),
                    ModelConfig(
                        model_id="DiffSynth-Studio/Z-Image-i2L",
                        origin_file_pattern="model.safetensors"
                    ),
                ],
                tokenizer_config=ModelConfig(
                    model_id="Tongyi-MAI/Z-Image-Turbo",
                    origin_file_pattern="tokenizer/"
                ),
            )
            safe_print("[OK] Pipeline loaded successfully")
            safe_print(f"[OK] All models initialized and ready\n")
            
        except Exception as e:
            safe_print(f"[ERROR] Error loading pipeline: {e}")
            raise
    
    def generate_lora_from_images(self, image_paths: List[str]) -> dict:
        """
        Generate LoRA from reference images
        
        Args:
            image_paths: List of paths to reference images (3-4 recommended)
            
        Returns:
            Dictionary containing the generated LoRA weights
        """
        if not self.pipe:
            raise RuntimeError("Pipeline not loaded. Call _load_pipeline first.")
        
        if len(image_paths) == 0:
            raise ValueError("No images provided")
        
        if len(image_paths) > 20:
            safe_print(f"[WARNING] {len(image_paths)} images provided. This may use significant VRAM.")
            safe_print(f"          Consider using 2-10 images for optimal performance.")
        
        safe_print(f"\n{'='*60}")
        safe_print(f"Starting LoRA Generation from {len(image_paths)} image(s)")
        safe_print(f"{'='*60}")
        safe_print(f"Loading {len(image_paths)} reference images...")
        
        # Load and convert images
        images = []
        for i, img_path in enumerate(image_paths, 1):
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
                safe_print(f"  [{i}/{len(image_paths)}] [OK] Loaded: {Path(img_path).name} | Size: {img.size}")
            except Exception as e:
                safe_print(f"  [{i}/{len(image_paths)}] [ERROR] Failed to load {img_path}: {e}")
                raise
        
        safe_print(f"\n{'='*60}")
        safe_print(f"Generating LoRA from {len(images)} image(s)...")
        safe_print(f"{'='*60}\n")
        
        try:
            with torch.no_grad():
                # Encode images to LoRA representation
                safe_print("  Step 1/2: Encoding reference images...")
                safe_print("           Analyzing visual patterns and style characteristics...")
                embs = ZImageUnit_Image2LoRAEncode().process(
                    self.pipe,
                    image2lora_images=images
                )
                safe_print("           [OK] Encoding complete\n")
                
                # Decode to actual LoRA weights
                safe_print("  Step 2/2: Decoding to LoRA weights...")
                safe_print("           Converting embeddings to LoRA adapter weights...")
                result = ZImageUnit_Image2LoRADecode().process(
                    self.pipe,
                    **embs
                )
                safe_print("           [OK] Decoding complete\n")
                
                lora_weights = result.get("lora")
                if lora_weights is None:
                    raise RuntimeError("Failed to extract LoRA weights from pipeline")
                
                safe_print(f"{'='*60}")
                safe_print("[OK] LoRA Generation Completed Successfully!")
                safe_print(f"{'='*60}\n")
                
                # Print LoRA weight statistics
                total_params = 0
                for key, value in lora_weights.items():
                    if isinstance(value, torch.Tensor):
                        params = value.numel()
                        total_params += params
                        safe_print(f"  {key}: {value.shape} ({params:,} parameters)")
                
                safe_print(f"\n  Total LoRA parameters: {total_params:,}")
                safe_print(f"  Ready to save and use for image generation!\n")
                
                return {"lora": lora_weights, "embs": embs}
                
        except Exception as e:
            safe_print(f"\n[ERROR] Error during LoRA generation: {e}")
            raise
    
    def save_lora(self, lora_weights: dict, output_path: str):
        """
        Save LoRA weights to a .safetensors file
        
        Args:
            lora_weights: LoRA weights dictionary from generate_lora_from_images
            output_path: Path where to save the .safetensors file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        safe_print(f"\n{'='*60}")
        safe_print("Saving LoRA to .safetensors format")
        safe_print(f"{'='*60}\n")
        safe_print(f"  Preparing weights for storage...")
        
        # Ensure weights are on CPU and in float format for saving
        cpu_weights = {}
        for key, value in lora_weights.items():
            if isinstance(value, torch.Tensor):
                cpu_weights[key] = value.cpu().to(torch.float32)
            else:
                cpu_weights[key] = value
        
        safe_print(f"  Converting {len(cpu_weights)} weight tensors to CPU...")
        save_file(cpu_weights, str(output_path))
        
        file_size = output_path.stat().st_size / 1024 / 1024
        safe_print(f"\n[OK] LoRA saved successfully!")
        safe_print(f"  Output: {output_path}")
        safe_print(f"  Size: {file_size:.2f} MB")
        safe_print(f"  Format: .safetensors")
        safe_print(f"\n{'='*60}\n")
    
    def cleanup(self):
        """Free GPU memory"""
        if self.pipe:
            del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Generate LoRA from reference images using Z-Image-i2L"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Paths to reference images (3-4 recommended for best results)"
    )
    parser.add_argument(
        "--output",
        default="generated_lora.safetensors",
        help="Output path for the generated LoRA (.safetensors format)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use: 'cuda' or 'cpu' (auto-detects if not specified)"
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["bfloat16", "float32"],
        help="Data type for computation (auto-selects based on device if not specified)"
    )
    
    args = parser.parse_args()
    
    # Validate input images
    for img_path in args.images:
        if not Path(img_path).exists():
            safe_print(f"[ERROR] Image not found: {img_path}")
            exit(1)
    
    # Auto-detect device if not specified
    if args.device:
        device = args.device
    else:
        # Check CUDA with diagnostics
        try:
            is_cuda_available = torch.cuda.is_available()
            cuda_device_count = torch.cuda.device_count() if is_cuda_available else 0
            safe_print(f"[DEBUG] CUDA available: {is_cuda_available}, Device count: {cuda_device_count}")
            device = 'cuda' if is_cuda_available and cuda_device_count > 0 else 'cpu'
        except Exception as e:
            safe_print(f"[DEBUG] CUDA check failed: {e}")
            device = 'cpu'
    
    safe_print(f"[DEBUG] Selected device: {device}")
    
    # Auto-select dtype if not specified (bfloat16 for CUDA, float32 for CPU)
    if args.dtype:
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    else:
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    
    # Print header
    safe_print(f"\n{'='*60}")
    safe_print("Z-Image-i2L: Image to LoRA Generation")
    safe_print(f"{'='*60}\n")
    safe_print(f"Configuration:")
    safe_print(f"  Device: {device}")
    safe_print(f"  Data Type: {dtype}")
    safe_print(f"  Input Images: {len(args.images)}")
    safe_print(f"  Output: {args.output}")
    safe_print(f"\n")
    
    # Generate LoRA
    try:
        safe_print("Initializing Z-Image-i2L pipeline...")
        generator = ZImageI2LGenerator(device=device, torch_dtype=dtype)
        
        result = generator.generate_lora_from_images(args.images)
        
        generator.save_lora(result["lora"], args.output)
        
        safe_print(f"{'='*60}")
        safe_print("[SUCCESS] LoRA generation COMPLETED SUCCESSFULLY!")
        safe_print(f"{'='*60}\n")
        safe_print(f"Recommended inference parameters:")
        safe_print(f"  - cfg_scale: 4")
        safe_print(f"  - sigma_shift: 8")
        safe_print(f"  - positive_only_lora: true")
        safe_print(f"  - num_inference_steps: 50\n")
        safe_print(f"Use the generated LoRA with Z-Image for text-to-image generation!\n")
        
        generator.cleanup()
        
    except Exception as e:
        safe_print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
