"""
Z-Image Text-to-Image Generation Script using Z-Image-i2L Generated LoRA
Consistent with official Z-Image example: https://modelscope.cn/models/DiffSynth-Studio/Z-Image-i2L
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import torch
from PIL import Image

try:
    from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
    from safetensors.torch import load_file
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install -e DiffSynth-Studio/DiffSynth-Studio-repo")
    exit(1)


# Safe print helper for Windows compatibility
def safe_print(*args, **kwargs):
    """Print with safe encoding handling for Windows"""
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        text = " ".join(str(arg) for arg in args)
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        print(text, flush=True)


def main():
    """
    Main execution function - follows official example pattern
    Loads pipeline once, then generates image with optional LoRA
    """
    parser = argparse.ArgumentParser(
        description="Generate images using Z-Image with optional i2L LoRA"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="Path to LoRA .safetensors file (optional)"
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt (uses recommended i2L prompt if not specified)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)"
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale (default: 4.0 for i2L)"
    )
    parser.add_argument(
        "--sigma-shift",
        type=float,
        default=8.0,
        help="Sigma shift parameter (default: 8.0 for i2L)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (-1 for random)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Image height (default: 768)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Image width (default: 768)"
    )
    parser.add_argument(
        "--output",
        default="generated_image.png",
        help="Output path for generated image"
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
    
    # Validate LoRA path if provided
    if args.lora and not Path(args.lora).exists():
        safe_print(f"[ERROR] LoRA file not found: {args.lora}")
        exit(1)
    
    # Auto-detect device if not specified
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Auto-select dtype if not specified (bfloat16 for CUDA, float32 for CPU)
    if args.dtype:
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    else:
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    
    try:
        safe_print(f"\n{'='*60}")
        safe_print("Z-Image Text-to-Image Generation (Consistent with Official Example)")
        safe_print(f"{'='*60}\n")
        
        # Load pipeline (same pattern as official example)
        safe_print(f"Loading Z-Image pipeline on {device} with {dtype}...")
        
        # VRAM configuration (consistent with official)
        vram_config = {
            "offload_dtype": dtype,
            "offload_device": device,
            "onload_dtype": dtype,
            "onload_device": device,
            "preparing_dtype": dtype,
            "preparing_device": device,
            "computation_dtype": dtype,
            "computation_device": device,
        }
        
        # Load pipeline (same as official example)
        pipe = ZImagePipeline.from_pretrained(
            torch_dtype=dtype,
            device=device,
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
            ],
            tokenizer_config=ModelConfig(
                model_id="Tongyi-MAI/Z-Image-Turbo",
                origin_file_pattern="tokenizer/"
            ),
        )
        safe_print("[OK] Pipeline loaded successfully\n")
        
        # Load LoRA if provided
        lora = None
        if args.lora:
            lora_path = Path(args.lora)
            safe_print(f"Loading LoRA from: {lora_path}")
            try:
                lora = load_file(str(lora_path))
                file_size_mb = lora_path.stat().st_size / 1024 / 1024
                safe_print(f"[OK] LoRA loaded ({file_size_mb:.2f} MB)\n")
            except Exception as e:
                safe_print(f"[ERROR] Error loading LoRA: {e}")
                raise
        
        # Use recommended negative prompt if not provided
        negative_prompt = args.negative_prompt
        if negative_prompt is None:
            negative_prompt = (
                "泛黄，发绿，模糊，低分辨率，低质量图像，扭曲的肢体，诡异的外观，丑陋，"
                "AI感，噪点，网格感，JPEG压缩条纹，异常的肢体，水印，乱码，意义不明的字符"
            )
        
        # Generate image
        safe_print(f"Generating image...")
        safe_print(f"  Prompt: {args.prompt}")
        if lora:
            safe_print(f"  Using LoRA style")
        safe_print(f"  Steps: {args.steps}, CFG: {args.cfg}, Sigma shift: {args.sigma_shift}")
        safe_print(f"  Size: {args.height}x{args.width}\n")
        
        try:
            with torch.no_grad():
                # Build generation kwargs (consistent with official example)
                kwargs = {
                    "prompt": args.prompt,
                    "negative_prompt": negative_prompt,
                    "seed": None if args.seed == -1 else args.seed,
                    "cfg_scale": args.cfg,
                    "num_inference_steps": args.steps,
                    "sigma_shift": args.sigma_shift,
                    "height": args.height,
                    "width": args.width,
                }
                
                # Apply LoRA if available (consistent with official example)
                # Ensure LoRA is on correct device
                if lora:
                    safe_print(f"[DEBUG] Moving LoRA to device: {device}")
                    lora_on_device = {}
                    for key, value in lora.items():
                        if isinstance(value, torch.Tensor):
                            lora_on_device[key] = value.to(device=device, dtype=dtype)
                        else:
                            lora_on_device[key] = value
                    kwargs["positive_only_lora"] = lora_on_device
                
                safe_print(f"[DEBUG] Generating with device: {device}, dtype: {dtype}")
                image = pipe(**kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e) or "device" in str(e):
                safe_print(f"[ERROR] Device/GPU Error: {e}")
                safe_print(f"[DEBUG] Device: {device}, Available CUDA: {torch.cuda.is_available()}")
            raise
        
        # Save image
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(output_path))
        
        safe_print(f"[OK] Image generated and saved successfully!")
        safe_print(f"  Output: {output_path}")
        safe_print(f"  Size: {image.size}\n")
        
    except Exception as e:
        safe_print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
