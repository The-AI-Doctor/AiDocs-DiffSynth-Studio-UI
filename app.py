"""
Gradio UI for DiffSynth-Studio with Model Download and LoRA Training
"""

import gradio as gr
import os
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple
import shutil
import sys
from PIL import Image

class DiffSynthStudioUI:
    def __init__(self):
        # Set paths relative to the UI folder
        self.ui_folder = Path(__file__).parent
        self.base_dir = self.ui_folder / "DiffSynth-Studio"
        self.models_dir = self.base_dir / "models"
        self.loras_dir = self.base_dir / "loras"
        self.z_image_i2l_dir = self.models_dir / "Z-Image-i2L"
        self.status_log = []
        
        # Track installation and model status for UI visibility
        self.installation_complete = False
        self.models_downloaded = False
        
        # Auto-initialize on startup
        self._initialize_on_startup()
        
    def _initialize_on_startup(self):
        """Auto-initialize DiffSynth-Studio and Git LFS on startup"""
        try:
            # Create directories
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.loras_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*60}")
            print("DiffSynth-Studio UI - Startup Initialization")
            print(f"{'='*60}")
            print(f"Base directory: {self.base_dir}")
            print()
            
            # Auto-install DiffSynth-Studio from GitHub if not present
            self._auto_install_diffsynth_repo()
            
            # Auto-install ModelScope if not present
            self._auto_install_modelscope()
            
            # Auto-install Git LFS if not present
            self._auto_install_git_lfs()
            
            # Check installation and model status
            self._check_startup_status()
            
            print(f"\n{'='*60}")
            print("Startup initialization complete!")
            if self.installation_complete and self.models_downloaded:
                print("✓ All components ready - UI tabs optimized")
            else:
                if not self.installation_complete:
                    print("⚠ Installation tab available - verify setup if needed")
                if not self.models_downloaded:
                    print("⚠ Model Download tab available - download models to get started")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"Startup initialization warning: {str(e)}\n")
    
    def _auto_install_diffsynth_repo(self):
        """Auto-install DiffSynth-Studio from GitHub repository if not present"""
        repo_dir = self.base_dir / "DiffSynth-Studio-repo"
        
        if repo_dir.exists():
            print(f"✓ DiffSynth-Studio repository already cloned")
            print(f"  Location: {repo_dir}")
            return
        
        print("Cloning DiffSynth-Studio from GitHub...")
        print("(This may take a few minutes on first run)")
        
        try:
            result = subprocess.run(
                ["git", "clone", "https://github.com/modelscope/DiffSynth-Studio.git", str(repo_dir)],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                print(f"✓ DiffSynth-Studio cloned successfully")
                print(f"  Location: {repo_dir}")
                
                # Try to install from the cloned repo
                print("Installing DiffSynth-Studio from source...")
                install_result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_dir)],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if install_result.returncode == 0:
                    print("✓ DiffSynth-Studio source installed successfully")
                else:
                    print("⚠ DiffSynth-Studio installation had issues, but repository is available")
            else:
                print(f"⚠ Clone failed: {result.stderr[:200]}")
                print(f"  You can manually clone: git clone https://github.com/modelscope/DiffSynth-Studio.git")
                
        except FileNotFoundError:
            print("⚠ Git not found - cannot clone DiffSynth-Studio repository")
            print("  Install Git from: https://git-scm.com/download/win")
        except subprocess.TimeoutExpired:
            print("⚠ Clone timeout - network may be slow")
        except Exception as e:
            print(f"⚠ Clone error: {str(e)}")
    
    def _auto_install_diffsynth(self):
        """Auto-install DiffSynth-Studio package if not already installed"""
        try:
            import diffsynth
            print("✓ DiffSynth-Studio package already installed")
        except ImportError:
            print("Installing DiffSynth-Studio package...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "diffsynth"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print("✓ DiffSynth-Studio package installed successfully")
            else:
                print(f"⚠ DiffSynth-Studio installation had issues")
                print(f"  Error: {result.stderr[:200]}")
    
    def _auto_install_modelscope(self):
        """Auto-install ModelScope package if not already installed"""
        try:
            import modelscope
            print("✓ ModelScope package already installed")
        except ImportError:
            print("Installing ModelScope package...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "modelscope>=1.13.2"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print("✓ ModelScope package installed successfully")
            else:
                print(f"⚠ ModelScope installation had issues")
                print(f"  Error: {result.stderr[:200]}")
    
    def _auto_install_git_lfs(self):
        """Auto-initialize Git LFS if not already done"""
        try:
            result = subprocess.run(
                ["git", "lfs", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print("✓ Git LFS already installed and initialized")
            else:
                print("Initializing Git LFS...")
                init_result = subprocess.run(
                    ["git", "lfs", "install"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if init_result.returncode == 0:
                    print("✓ Git LFS initialized successfully")
                else:
                    print("⚠ Git LFS initialization may need manual setup")
        except FileNotFoundError:
            print("⚠ Git not found - you may need to install Git and Git LFS manually")
            print("  Download from: https://git-scm.com/download/win")
        except Exception as e:
            print(f"⚠ Git LFS check: {str(e)}")
    
    def _check_startup_status(self):
        """Check installation and model download status on startup"""
        # Check installation status
        installation_ok = True
        
        # Check DiffSynth-Studio repo
        repo_dir = self.base_dir / "DiffSynth-Studio-repo"
        if not repo_dir.exists():
            installation_ok = False
        
        # Check packages
        try:
            import diffsynth
        except ImportError:
            installation_ok = False
        
        try:
            import modelscope
        except ImportError:
            installation_ok = False
        
        # Check Git and Git LFS
        try:
            subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
            subprocess.run(["git", "lfs", "version"], capture_output=True, text=True, timeout=5)
        except:
            installation_ok = False
        
        self.installation_complete = installation_ok
        
        # Check model download status
        models_ok = self.z_image_i2l_dir.exists() and (self.z_image_i2l_dir / "README.md").exists()
        self.models_downloaded = models_ok
    
    def log_status(self, message: str):
        """Log status messages"""
        self.status_log.append(message)
        print(message)
        return "\n".join(self.status_log[-50:])  # Keep last 50 messages
    
    # ============= Installation Functions (Handled at Startup) =============
    
    def check_installation(self) -> str:
        """Check if all components are properly installed"""
        status = "=== Installation Check ===\n\n"
        
        # Check DiffSynth-Studio repo
        repo_dir = self.base_dir / "DiffSynth-Studio-repo"
        if repo_dir.exists():
            status += "✓ DiffSynth-Studio repository cloned\n"
        else:
            status += "✗ DiffSynth-Studio repository not found\n"
        
        # Check package installation
        try:
            import diffsynth
            status += "✓ DiffSynth-Studio package installed\n"
        except ImportError:
            status += "✗ DiffSynth-Studio package not installed\n"
        
        # Check ModelScope installation
        try:
            import modelscope
            status += "✓ ModelScope package installed\n"
        except ImportError:
            status += "✗ ModelScope package not installed\n"
        
        # Check Git
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                status += f"✓ Git installed: {result.stdout.strip()}\n"
            else:
                status += "✗ Git not found\n"
        except:
            status += "✗ Git not found\n"
        
        # Check Git LFS
        try:
            result = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True)
            if result.returncode == 0:
                status += "✓ Git LFS installed and initialized\n"
            else:
                status += "✗ Git LFS not initialized\n"
        except:
            status += "✗ Git LFS not found\n"
        
        return status
    
    def create_dataset(self, dataset_name: str) -> str:
        """Create a new dataset folder"""
        try:
            if not dataset_name.strip():
                return "✗ Dataset name cannot be empty"
            
            dataset_dir = self.loras_dir / dataset_name / "images"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            self.log_status(f"✓ Dataset created: {dataset_name}")
            return f"✓ Dataset '{dataset_name}' created successfully!\n\nPath: {dataset_dir}"
            
        except Exception as e:
            msg = f"✗ Error creating dataset: {str(e)}"
            self.log_status(msg)
            return msg
    
    def list_datasets(self) -> list:
        """List all available datasets"""
        try:
            if not self.loras_dir.exists():
                return []
            
            datasets = []
            for item in self.loras_dir.iterdir():
                if item.is_dir():
                    images_dir = item / "images"
                    if images_dir.exists():
                        image_count = len(list(images_dir.glob("*.*")))
                        datasets.append(f"{item.name} ({image_count} images)")
            
            return sorted(datasets) if datasets else ["No datasets yet"]
            
        except Exception as e:
            self.log_status(f"Error listing datasets: {str(e)}")
            return ["Error loading datasets"]
    
    def get_dataset_images(self, dataset_name: str) -> list:
        """Get list of images in a dataset"""
        try:
            if not dataset_name or dataset_name == "No datasets yet":
                return []
            
            # Extract actual name from "name (count images)" format
            actual_name = dataset_name.split(" (")[0]
            images_dir = self.loras_dir / actual_name / "images"
            
            if not images_dir.exists():
                return []
            
            image_files = []
            # Use case-insensitive glob to avoid duplicates
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                image_files.extend(images_dir.glob(ext))
                # Also get uppercase versions, but avoid duplicates on case-insensitive filesystems
                image_files.extend(images_dir.glob(ext.upper()))
            
            # Remove duplicates (important for Windows case-insensitive filesystem)
            unique_images = {}
            for img in image_files:
                # Use lower() to detect duplicates on case-insensitive systems
                key = img.name.lower()
                if key not in unique_images:
                    unique_images[key] = img.name
            
            return sorted(unique_images.values())
            
        except Exception as e:
            self.log_status(f"Error loading images: {str(e)}")
            return []
    
    def delete_image(self, dataset_name: str, image_name: str) -> Tuple[str, list]:
        """Delete an image from dataset"""
        try:
            if not dataset_name or dataset_name == "No datasets yet":
                return "✗ No dataset selected", []
            
            actual_name = dataset_name.split(" (")[0]
            image_path = self.loras_dir / actual_name / "images" / image_name
            
            if image_path.exists():
                image_path.unlink()
                self.log_status(f"✓ Deleted: {image_name}")
                return f"✓ Deleted: {image_name}", self.get_dataset_images(dataset_name)
            else:
                return f"✗ Image not found: {image_name}", self.get_dataset_images(dataset_name)
                
        except Exception as e:
            msg = f"✗ Error deleting image: {str(e)}"
            self.log_status(msg)
            return msg, self.get_dataset_images(dataset_name)
    
    def delete_dataset(self, dataset_name: str) -> Tuple[str, list]:
        """Delete entire dataset"""
        try:
            if not dataset_name or dataset_name == "No datasets yet":
                return "✗ No dataset selected", []
            
            actual_name = dataset_name.split(" (")[0]
            dataset_dir = self.loras_dir / actual_name
            
            if dataset_dir.exists():
                import shutil
                shutil.rmtree(dataset_dir)
                self.log_status(f"✓ Dataset deleted: {actual_name}")
                return f"✓ Dataset deleted: {actual_name}", self.list_datasets()
            else:
                return f"✗ Dataset not found: {actual_name}", self.list_datasets()
                
        except Exception as e:
            msg = f"✗ Error deleting dataset: {str(e)}"
            self.log_status(msg)
            return msg, self.list_datasets()
    
    # ============= Model Download Functions =============
    
    def download_z_image_i2l_model(self) -> str:
        """Download Z-Image-i2L model - tries ModelScope API first, then Git LFS fallback"""
        try:
            self.log_status("Starting Z-Image-i2L model download...")
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            target_dir = self.z_image_i2l_dir
            
            if target_dir.exists():
                self.log_status(f"✓ Z-Image-i2L already exists at {target_dir}")
                return f"✓ Z-Image-i2L already downloaded"
            
            # First, try using modelscope CLI for better performance
            self.log_status("Attempting download via ModelScope API...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "modelscope", "download", 
                     "--model", "DiffSynth-Studio/Z-Image-i2L",
                     "--local_dir", str(target_dir)],
                    capture_output=True,
                    text=True,
                    timeout=3600
                )
                
                if result.returncode == 0:
                    self.log_status("✓ Z-Image-i2L downloaded successfully via ModelScope")
                    return "✓ Z-Image-i2L downloaded successfully via ModelScope"
                else:
                    self.log_status("ModelScope API download failed, falling back to Git LFS...")
                    return self._download_z_image_i2l_with_git_lfs()
                    
            except Exception as e:
                self.log_status(f"ModelScope download error: {str(e)}, falling back to Git LFS...")
                return self._download_z_image_i2l_with_git_lfs()
                
        except Exception as e:
            msg = f"✗ Download error: {str(e)}"
            self.log_status(msg)
            return msg
    
    def _download_z_image_i2l_with_git_lfs(self) -> str:
        """Fallback: Download Z-Image-i2L using Git LFS"""
        try:
            target_dir = self.z_image_i2l_dir
            
            # Ensure Git LFS is installed
            self.log_status("Ensuring Git LFS is installed...")
            subprocess.run(["git", "lfs", "install"], capture_output=True)
            
            model_url = "https://www.modelscope.cn/models/DiffSynth-Studio/Z-Image-i2L.git"
            
            self.log_status(f"Cloning Z-Image-i2L from {model_url}...")
            result = subprocess.run(
                ["git", "clone", model_url, str(target_dir)],
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                self.log_status("✓ Z-Image-i2L downloaded successfully via Git LFS")
                return "✓ Z-Image-i2L downloaded successfully via Git LFS"
            else:
                error_msg = result.stderr or result.stdout
                self.log_status(f"✗ Git LFS clone failed: {error_msg[:200]}")
                return f"✗ Git LFS clone failed: {error_msg[:200]}"
                
        except subprocess.TimeoutExpired:
            msg = "✗ Download timeout (took too long)"
            self.log_status(msg)
            return msg
        except Exception as e:
            msg = f"✗ Git LFS download error: {str(e)}"
            self.log_status(msg)
            return msg
    
    def download_all_models(self) -> str:
        """Download Z-Image-i2L model with ModelScope API first, then Git LFS fallback"""
        try:
            self.log_status("\n" + "="*60)
            self.log_status("Starting Z-Image-i2L Model Download")
            self.log_status("="*60)
            
            # Download Z-Image-i2L (ModelScope first, then Git LFS fallback)
            result = self.download_z_image_i2l_model()
            self.log_status(result)
            
            self.log_status("\n" + "="*60)
            self.log_status("Download Complete!")
            self.log_status("="*60 + "\n")
            
            return result
                
        except Exception as e:
            msg = f"✗ Download error: {str(e)}"
            self.log_status(msg)
            return msg
    
    # ============= LoRA Training Configuration =============
    
    def create_lora_config(
        self,
        lora_name: str,
        rank: int = 32,
        alpha: int = 64,
        learning_rate: float = 0.0001,
        num_epochs: int = 100,
        batch_size: int = 1,
        steps_per_epoch: int = 1000,
    ) -> str:
        """Create LoRA training configuration"""
        try:
            if not lora_name.strip():
                return "✗ LoRA name cannot be empty"
            
            self.loras_dir.mkdir(parents=True, exist_ok=True)
            lora_dir = self.loras_dir / lora_name
            lora_dir.mkdir(exist_ok=True)
            
            # Create config file
            config = {
                "name": lora_name,
                "model_reasoning": {
                    "base_model": "Z-Image-i2L",
                    "model_type": "image-to-image",
                    "architecture": "UNet",
                },
                "lora_config": {
                    "rank": rank,
                    "alpha": alpha,
                    "target_modules": ["to_k", "to_v", "to_q", "to_out"],
                    "r_dropout": 0.0,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                },
                "training": {
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "steps_per_epoch": steps_per_epoch,
                    "optimizer": "adamw_8bit",
                    "warmup_steps": 500,
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "logging_steps": 100,
                    "save_steps": 500,
                },
                "data": {
                    "dataset_path": str(lora_dir / "dataset"),
                    "image_size": 512,
                    "center_crop": True,
                    "random_flip": True,
                },
                "inference": {
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50,
                    "negative_prompt": "low quality, blurry, distorted",
                }
            }
            
            config_path = lora_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            self.log_status(f"✓ Created LoRA config: {lora_name}")
            self.log_status(f"  - Rank: {rank}, Alpha: {alpha}")
            self.log_status(f"  - Learning Rate: {learning_rate}, Epochs: {num_epochs}")
            
            return f"""✓ LoRA Configuration Created!

Name: {lora_name}
Location: {lora_dir}

LoRA Settings:
  - Rank: {rank}
  - Alpha: {alpha}
  - Target Modules: to_k, to_v, to_q, to_out

Training Settings:
  - Learning Rate: {learning_rate}
  - Epochs: {num_epochs}
  - Batch Size: {batch_size}
  - Steps per Epoch: {steps_per_epoch}

Model Reasoning:
  - Base Model: Z-Image-i2 (Image-to-Image)
  - Architecture: UNet with LoRA adapters
  - Training Objective: Fine-tune Z-Image for custom styles

Config file saved to: {config_path}
"""
            
        except Exception as e:
            msg = f"✗ Configuration error: {str(e)}"
            self.log_status(msg)
            return msg
    
    def list_loras(self) -> str:
        """List available LoRA configurations"""
        try:
            if not self.loras_dir.exists():
                return "No LoRAs created yet"
            
            loras = [d.name for d in self.loras_dir.iterdir() if d.is_dir()]
            
            if not loras:
                return "No LoRAs created yet"
            
            output = "Available LoRAs:\n"
            for lora in loras:
                config_path = self.loras_dir / lora / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                        output += f"\n📦 {lora}\n"
                        output += f"  Rank: {config['lora_config']['rank']}\n"
                        output += f"  LR: {config['training']['learning_rate']}\n"
                else:
                    output += f"\n📦 {lora}\n"
            
            return output
            
        except Exception as e:
            return f"✗ Error: {str(e)}"
    
    def train_lora(self, lora_name: str) -> Tuple[str, dict]:
        """Start training a LoRA configuration
        
        Returns:
            Tuple of (status_message, process_info_dict)
        """
        try:
            lora_dir = self.loras_dir / lora_name
            config_path = lora_dir / "config.json"
            
            if not config_path.exists():
                return f"✗ LoRA config not found: {config_path}", {"status": "error"}
            
            # Check if dataset has images
            dataset_dir = lora_dir / "dataset"
            if not dataset_dir.exists() or not list(dataset_dir.glob("*")):
                return f"✗ No training images found in {dataset_dir}\nPlease upload images first.", {"status": "error"}
            
            # Create training process
            try:
                # Use the train_lora.py script
                train_script = Path(__file__).parent / "train_lora.py"
                
                process = subprocess.Popen(
                    [sys.executable, str(train_script), "--config", str(config_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Store process info
                process_info = {
                    "status": "running",
                    "pid": process.pid,
                    "lora_name": lora_name,
                    "config_path": str(config_path),
                    "process": process
                }
                
                # Store in instance for later reference
                if not hasattr(self, '_training_processes'):
                    self._training_processes = {}
                self._training_processes[lora_name] = process_info
                
                msg = f"✓ Training started for {lora_name}\nProcess ID: {process.pid}\n\nTraining will run in the background.\nCheck status updates below."
                self.log_status(f"Started training: {lora_name}")
                
                return msg, process_info
                
            except FileNotFoundError:
                return "✗ train_lora.py script not found", {"status": "error"}
                
        except Exception as e:
            msg = f"✗ Training error: {str(e)}"
            self.log_status(msg)
            return msg, {"status": "error"}
    
    def stop_training(self, lora_name: str) -> str:
        """Stop an ongoing training process"""
        try:
            if not hasattr(self, '_training_processes'):
                return f"✗ No active training for {lora_name}"
            
            process_info = self._training_processes.get(lora_name)
            if not process_info:
                return f"✗ No active training for {lora_name}"
            
            process = process_info.get("process")
            if process and process.poll() is None:
                process.terminate()
                self.log_status(f"Stopped training: {lora_name}")
                return f"✓ Training stopped for {lora_name}"
            else:
                return f"✗ Training is not running for {lora_name}"
                
        except Exception as e:
            return f"✗ Error stopping training: {str(e)}"
    
    def get_training_status(self, lora_name: str) -> str:
        """Get training status and output"""
        try:
            if not hasattr(self, '_training_processes'):
                return "No training in progress"
            
            process_info = self._training_processes.get(lora_name)
            if not process_info:
                return "No training in progress"
            
            process = process_info.get("process")
            if not process:
                return "No training process found"
            
            # Read available output without blocking
            output = []
            if process.stdout:
                import select
                # Use select for non-blocking read on available systems
                try:
                    # Try to read any available output
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        output.append(line.rstrip())
                except:
                    pass
            
            # Check if process is still running
            poll_status = process.poll()
            if poll_status is None:
                status_text = "🔄 Training in progress..."
            elif poll_status == 0:
                status_text = "✓ Training completed successfully"
            else:
                status_text = f"✗ Training failed (exit code: {poll_status})"
            
            # Combine with recent status logs
            all_output = [status_text] + output + self.status_log[-20:]
            return "\n".join(all_output[-30:])  # Show last 30 lines
                
        except Exception as e:
            return f"Error getting training status: {str(e)}"
    
    def test_lora(self, lora_name: str, prompt: str, 
                  num_steps: int = 50, guidance_scale: float = 7.5) -> Tuple[str, Optional[Image.Image]]:
        """Test a trained LoRA with image generation
        
        Args:
            lora_name: Name of the LoRA to test
            prompt: Text prompt for generation
            num_steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            
        Returns:
            Tuple of (status_message, generated_image)
        """
        try:
            lora_dir = self.loras_dir / lora_name
            config_path = lora_dir / "config.json"
            
            if not config_path.exists():
                return f"✗ LoRA config not found: {config_path}", None
            
            # Check if checkpoints exist
            checkpoint_dir = lora_dir / "checkpoints"
            if not checkpoint_dir.exists() or not list(checkpoint_dir.rglob("*.bin")) and not list(checkpoint_dir.rglob("*.safetensors")):
                return f"✗ No trained checkpoints found. Please train the LoRA first.", None
            
            try:
                # Import the inference module
                inference_script = Path(__file__).parent / "inference_lora.py"
                if not inference_script.exists():
                    return f"✗ Inference script not found", None
                
                # Import LoRAInference class
                import importlib.util
                spec = importlib.util.spec_from_file_location("inference_lora", inference_script)
                inference_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(inference_module)
                
                # Create inference engine
                inference = inference_module.LoRAInference(str(lora_dir))
                
                # Generate image
                image = inference.generate(
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale
                )
                
                # Save output
                output_dir = lora_dir / "test_outputs"
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"test_{len(list(output_dir.glob('*'))):03d}.png"
                image.save(str(output_path))
                
                inference.cleanup()
                
                msg = f"✓ Generated image: {output_path}\nPrompt: {prompt}"
                self.log_status(msg)
                
                return msg, image
                
            except ImportError as e:
                return f"✗ Could not import inference module: {str(e)}", None
            except Exception as e:
                return f"✗ Generation error: {str(e)}", None
                
        except Exception as e:
            msg = f"✗ Test error: {str(e)}"
            self.log_status(msg)
            return msg, None
    
    # ============= Z-Image-i2L: Image to LoRA Conversion =============
    
    def generate_lora_from_images(self, image_paths: list, lora_name: str) -> Tuple[str, Optional[str]]:
        """Generate a LoRA model from reference images using Z-Image-i2L
        
        Args:
            image_paths: List of paths to reference images (3-4 recommended)
            lora_name: Name for the generated LoRA
            
        Returns:
            Tuple of (status_message, lora_file_path)
        """
        try:
            if not image_paths or len(image_paths) == 0:
                return "✗ No images selected", None
            
            if not lora_name.strip():
                return "✗ LoRA name cannot be empty", None
            
            # Create LoRA directory
            self.loras_dir.mkdir(parents=True, exist_ok=True)
            lora_dir = self.loras_dir / lora_name
            lora_dir.mkdir(exist_ok=True)
            
            # Save LoRA info
            lora_info = {
                "name": lora_name,
                "type": "image-to-lora",
                "created_from": len(image_paths) if isinstance(image_paths, list) else 1,
                "model": "Z-Image-i2L",
                "inference_params": {
                    "cfg_scale": 4,
                    "sigma_shift": 8,
                    "positive_only_lora": True,
                    "num_inference_steps": 50
                }
            }
            
            info_path = lora_dir / "info.json"
            with open(info_path, "w") as f:
                json.dump(lora_info, f, indent=2)
            
            # Call inference_i2l.py script
            try:
                inference_script = Path(__file__).parent / "inference_i2l.py"
                if not inference_script.exists():
                    return f"✗ inference_i2l.py not found at {inference_script}", None
                
                # Convert image paths (handle both Path objects and strings)
                image_args = [str(p) for p in image_paths]
                
                # Generate LoRA
                output_path = lora_dir / "lora.safetensors"
                
                # Use Popen for real-time output streaming
                # Force CUDA device
                process = subprocess.Popen(
                    [sys.executable, "-u", str(inference_script)] + 
                    ["--images"] + image_args + 
                    ["--output", str(output_path)] +
                    ["--device", "cuda"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line buffered
                    universal_newlines=True
                )
                
                # Collect output in real-time
                output_lines = []
                try:
                    for line in process.stdout:
                        line = line.rstrip()
                        output_lines.append(line)
                        print(line)  # Print to console for debugging
                        self.log_status(line)  # Also log to status
                except Exception as e:
                    print(f"Error reading output: {e}")
                
                # Wait for process to complete
                returncode = process.wait(timeout=1800)  # 30 minute timeout
                
                if returncode == 0:
                    if output_path.exists():
                        file_size = output_path.stat().st_size / 1024 / 1024
                        final_msg = f"✓ LoRA generated successfully!\n\nName: {lora_name}\nSize: {file_size:.2f} MB\nPath: {output_path}"
                        self.log_status(final_msg)
                        return final_msg, str(output_path)
                    else:
                        return f"✗ LoRA file was not created", None
                else:
                    error_output = "\n".join(output_lines[-20:]) if output_lines else "Unknown error"
                    return f"✗ LoRA generation failed:\n{error_output}", None
                    
            except subprocess.TimeoutExpired:
                return "✗ LoRA generation timeout (took too long)", None
            except FileNotFoundError:
                return f"✗ inference_i2l.py script not found", None
            except Exception as e:
                return f"✗ Error running LoRA generation: {str(e)}", None
                
        except Exception as e:
            msg = f"✗ Error generating LoRA: {str(e)}"
            self.log_status(msg)
            return msg, None
    
    def list_generated_loras(self) -> list:
        """List LoRAs generated from images (vs. trained LoRAs)"""
        try:
            if not self.loras_dir.exists():
                return []
            
            loras = []
            for item in self.loras_dir.iterdir():
                if item.is_dir():
                    info_path = item / "info.json"
                    if info_path.exists():
                        try:
                            with open(info_path) as f:
                                info = json.load(f)
                                if info.get("type") == "image-to-lora":
                                    lora_path = item / "lora.safetensors"
                                    if lora_path.exists():
                                        size_mb = lora_path.stat().st_size / 1024 / 1024
                                        loras.append(f"{item.name} ({size_mb:.1f} MB)")
                        except:
                            pass
            
            return loras if loras else ["No generated LoRAs yet"]
            
        except Exception as e:
            self.log_status(f"Error listing LoRAs: {str(e)}")
            return ["Error loading LoRAs"]
    
    def generate_image_with_i2l_lora(self, lora_name: str, prompt: str, 
                                      seed: int = 0, steps: int = 50, cfg_scale: float = 4.0, 
                                      sigma_shift: float = 8.0, negative_prompt: str = None) -> Tuple[str, Optional[Image.Image]]:
        """Generate an image using a Z-Image-i2L generated LoRA
        
        Args:
            lora_name: Name of the generated LoRA
            prompt: Text prompt for generation
            seed: Random seed
            steps: Number of inference steps
            cfg_scale: Classifier-free guidance scale
            sigma_shift: Sigma shift parameter
            negative_prompt: Negative prompt (optional)
            
        Returns:
            Tuple of (status_message, generated_image)
        """
        try:
            # Find LoRA file
            lora_dir = None
            if lora_name:
                clean_name = lora_name.split(" (")[0] if "(" in lora_name else lora_name
                lora_dir = self.loras_dir / clean_name
            
            if not lora_dir or not lora_dir.exists():
                return f"✗ LoRA not found: {lora_name}", None
            
            lora_path = lora_dir / "lora.safetensors"
            if not lora_path.exists():
                return f"✗ LoRA file not found: {lora_path}", None
            
            # Generate image with provided parameters
            try:
                generate_script = Path(__file__).parent / "generate_image.py"
                if not generate_script.exists():
                    return f"✗ generate_image.py not found", None
                
                output_dir = lora_dir / "generated_images"
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"generated_{len(list(output_dir.glob('*'))):03d}.png"
                
                # Build subprocess arguments with user-provided parameters
                process = subprocess.Popen(
                    [sys.executable, "-u", str(generate_script),
                     "--prompt", prompt,
                     "--lora", str(lora_path),
                     "--seed", str(seed),
                     "--cfg", str(cfg_scale),
                     "--sigma-shift", str(sigma_shift),
                     "--steps", str(steps)] +
                    (["--negative-prompt", negative_prompt] if negative_prompt else []) +
                    ["--device", "cuda",
                     "--output", str(output_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Read output in real-time
                output_lines = []
                try:
                    for line in process.stdout:
                        line_clean = line.rstrip()
                        output_lines.append(line_clean)
                        print(line_clean)  # Print to console for debugging
                    process.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return "✗ Image generation timeout", None
                
                if process.returncode == 0 and output_path.exists():
                    image = Image.open(output_path)
                    msg = f"✓ Generated image with i2L LoRA\nPrompt: {prompt}\nSaved: {output_path}"
                    self.log_status(msg)
                    return msg, image
                else:
                    error_output = "\n".join(output_lines[-30:]) if output_lines else "Unknown error"
                    return f"✗ Image generation failed:\n{error_output[:1000]}", None
                    
            except subprocess.TimeoutExpired:
                return "✗ Image generation timeout", None
            except FileNotFoundError:
                return f"✗ generate_image.py script not found", None
            except Exception as e:
                return f"✗ Error generating image: {str(e)}", None
                
        except Exception as e:
            msg = f"✗ Error: {str(e)}"
            self.log_status(msg)
            return msg, None
    
    def unload_models(self) -> str:
        """Unload generated LoRA models from memory to free VRAM
        
        Returns:
            Status message
        """
        try:
            import torch
            import gc
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                msg = "✓ Models unloaded and GPU memory cleared"
            else:
                gc.collect()
                msg = "✓ Memory cleared (no GPU available)"
            
            self.log_status(msg)
            return msg
            
        except Exception as e:
            msg = f"✗ Error unloading models: {str(e)}"
            self.log_status(msg)
            return msg
    
    def get_status(self) -> str:
        """Get current system status"""
        status = "=== DiffSynth-Studio Status ===\n\n"
        
        # Check base directory
        status += f"Base Directory: {self.base_dir}\n"
        status += f"Exists: {'✓' if self.base_dir.exists() else '✗'}\n\n"
        
        # Check models
        status += "Models:\n"
        if self.models_dir.exists():
            models = [d.name for d in self.models_dir.iterdir() if d.is_dir()]
            if models:
                for model in models:
                    status += f"  ✓ {model}\n"
            else:
                status += "  (none downloaded)\n"
        else:
            status += "  (directory not created)\n"
        
        # Check LoRAs
        status += "\nLoRAs:\n"
        if self.loras_dir.exists():
            loras = [d.name for d in self.loras_dir.iterdir() if d.is_dir()]
            if loras:
                for lora in loras:
                    status += f"  ✓ {lora}\n"
            else:
                status += "  (none created)\n"
        else:
            status += "  (directory not created)\n"
        
        # Check Git
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                status += f"\nGit: ✓ {result.stdout.strip()}\n"
            else:
                status += "\nGit: ✗ Not found\n"
        except:
            status += "\nGit: ✗ Not found\n"
        
        # Check Git LFS
        try:
            result = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True)
            if result.returncode == 0:
                status += f"Git LFS: ✓ Installed\n"
            else:
                status += "Git LFS: ✗ Not initialized\n"
        except:
            status += "Git LFS: ✗ Not found\n"
        
        return status


def create_ui():
    """Create the Gradio UI"""
    ui = DiffSynthStudioUI()
    
    with gr.Blocks(title="DiffSynth-Studio UI") as demo:
        gr.Markdown("""
        # 🎨 DiffSynth-Studio UI
        
        Complete interface for DiffSynth-Studio model management and LoRA training
        """)
        
        with gr.Tabs():
            # ============= Installation Tab =============
            with gr.Tab("🔧 Installation", visible=not ui.installation_complete):
                gr.Markdown("### Setup Status & Verification")
                
                with gr.Column():
                    gr.Markdown("""
                    **Installation Status**
                    
                    The following components are automatically installed on startup:
                    - ✓ DiffSynth-Studio repository
                    - ✓ Python dependencies
                    - ✓ Git LFS
                    
                    Click the button below to verify everything is properly installed.
                    """)
                    
                    check_btn = gr.Button("Check Installation Status", scale=1, size="lg")
                    
                    status_output = gr.Textbox(
                        label="Installation Status",
                        lines=15,
                        max_lines=20,
                        interactive=False
                    )
                
                check_btn.click(
                    fn=ui.get_status,
                    outputs=status_output
                )
            
            # ============= Model Download Tab =============
            with gr.Tab("⬇️ Model Download", visible=not ui.models_downloaded):
                gr.Markdown("### Download Z-Image-i2L Model")
                
                gr.Markdown("""
                This will download the Z-Image-i2L model (pre-trained image-to-image diffusion model).
                
                **Model**: Z-Image-i2L - Advanced image-to-image generation
                **Size**: ~7GB | **Time**: 10-30 minutes depending on connection
                
                The download will use ModelScope API first for best performance, with Git LFS as fallback.
                """)
                
                with gr.Column():
                    download_btn = gr.Button("Download All Models", scale=1, size="lg")
                    
                    download_output = gr.Textbox(
                        label="Download Status",
                        lines=15,
                        max_lines=20,
                        interactive=False
                    )
                
                download_btn.click(
                    fn=ui.download_all_models,
                    outputs=download_output
                )
            
            # ============= Dataset Management Tab =============
            with gr.Tab("📁 Datasets"):
                gr.Markdown("### Manage Training Datasets")
                
                # Hidden state to track current dataset
                current_dataset_state = gr.State("")
                
                with gr.Column():
                    gr.Markdown("#### Create New Dataset")
                    
                    with gr.Row():
                        dataset_input = gr.Textbox(
                            label="Dataset Name",
                            placeholder="e.g., anime_style, portrait_photos",
                            value=""
                        )
                        create_dataset_btn = gr.Button("Create Dataset", size="lg")
                
                gr.Markdown("---")
                
                with gr.Column():
                    gr.Markdown("#### Manage Datasets")
                    
                    # Refresh datasets list
                    dataset_choices = ui.list_datasets() or ["No datasets yet"]
                    datasets_list = gr.Dropdown(
                        choices=dataset_choices,
                        value=dataset_choices[0] if dataset_choices else "No datasets yet",
                        label="Select Dataset",
                        interactive=True,
                        allow_custom_value=False
                    )
                    refresh_btn = gr.Button("Refresh Datasets", size="sm")
                    
                    with gr.Row():
                        upload_btn = gr.UploadButton(
                            label="Upload Images",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        delete_dataset_btn = gr.Button("🗑️ Delete Dataset", size="sm", variant="stop")
                
                gr.Markdown("---")
                
                with gr.Column():
                    gr.Markdown("#### Dataset Images")
                    
                    # State to track selected image index
                    selected_image_index = gr.State(value=-1)
                    
                    # Standard Gradio gallery with proper configuration
                    images_gallery = gr.Gallery(
                        label="Images",
                        show_label=False,
                        columns=4,
                        rows=3,
                        object_fit="scale-down",
                        height="auto",
                        allow_preview=True,
                        interactive=False
                    )
                    
                    delete_btn = gr.Button("🗑️ Delete Selected Image", size="lg", variant="stop")
                
                # Event handlers
                def refresh_datasets():
                    """Refresh the datasets list"""
                    new_choices = ui.list_datasets() or ["No datasets yet"]
                    return gr.update(choices=new_choices, value=new_choices[0])
                

                def on_gallery_select(evt: gr.SelectData):
                    """Track which image is selected"""
                    return evt.index
                
                def delete_selected_image(gallery_data, dataset_name, selected_idx):
                    """Delete selected image from gallery"""
                    if not gallery_data or not dataset_name or dataset_name == "No datasets yet":
                        return gallery_data, -1
                    
                    if selected_idx < 0 or selected_idx >= len(gallery_data):
                        return gallery_data, selected_idx
                    
                    try:
                        actual_name = dataset_name.split(" (")[0]
                        
                        # gallery_data is a list of items (could be tuples or strings)
                        if isinstance(gallery_data, list) and len(gallery_data) > 0:
                            selected_item = gallery_data[selected_idx]
                            
                            # Handle both tuple (path, caption) and string (path) formats
                            if isinstance(selected_item, tuple):
                                image_path = selected_item[0]
                            else:
                                image_path = selected_item
                            
                            image_name = Path(image_path).name
                            
                            # Delete the image
                            delete_msg, updated_images = ui.delete_image(dataset_name, image_name)
                            
                            # Return updated gallery and reset selection
                            updated_paths = [str(ui.loras_dir / actual_name / "images" / img) for img in updated_images]
                            return updated_paths, -1
                    except Exception as e:
                        return gallery_data, selected_idx
                    
                    return gallery_data, selected_idx
                
                def on_dataset_select(dataset_name):
                    """Return list of image paths for gallery when dataset is selected"""
                    if not dataset_name or dataset_name == "No datasets yet":
                        return []
                    
                    images = ui.get_dataset_images(dataset_name)
                    actual_name = dataset_name.split(" (")[0]
                    
                    if not images:
                        return []
                    
                    # Return full paths for gallery component
                    image_paths = [str(ui.loras_dir / actual_name / "images" / img) for img in images]
                    return image_paths
                
                def upload_images(files, dataset_name):
                    """Handle image upload"""
                    if not dataset_name or dataset_name == "No datasets yet":
                        return "✗ Please select a dataset first"
                    
                    if not files:
                        return "✗ No files selected"
                    
                    try:
                        import time
                        from pathlib import Path
                        
                        actual_name = dataset_name.split(" (")[0]
                        images_dir = ui.loras_dir / actual_name / "images"
                        images_dir.mkdir(parents=True, exist_ok=True)
                        
                        count = 0
                        failed = []
                        
                        for file in files:
                            try:
                                file_path = Path(file) if not isinstance(file, Path) else file
                                dest_path = images_dir / file_path.name
                                
                                # Ensure source file exists
                                if not file_path.exists():
                                    failed.append(f"{file_path.name} (source not found)")
                                    continue
                                
                                # Aggressive retry with read/write instead of copy
                                max_retries = 5
                                success = False
                                
                                for attempt in range(max_retries):
                                    try:
                                        # Increase delay significantly on each retry
                                        delay = 0.2 * (attempt + 1)
                                        time.sleep(delay)
                                        
                                        # Read file content first to release the lock
                                        try:
                                            with open(file_path, 'rb') as src:
                                                file_content = src.read()
                                            
                                            # Write to destination
                                            with open(dest_path, 'wb') as dst:
                                                dst.write(file_content)
                                            
                                            # Verify file was written
                                            if dest_path.exists() and dest_path.stat().st_size > 0:
                                                count += 1
                                                success = True
                                                break
                                        except (OSError, IOError) as e:
                                            if attempt == max_retries - 1:
                                                failed.append(f"{file_path.name} (read/write failed: {str(e)[:40]})")
                                            if attempt < max_retries - 1:
                                                continue
                                            
                                    except Exception as e:
                                        if attempt == max_retries - 1:
                                            failed.append(f"{file_path.name} ({str(e)[:40]})")
                                        if attempt < max_retries - 1:
                                            continue
                                
                            except Exception as e:
                                failed.append(f"{str(file)[:30]} ({str(e)[:30]})")
                                ui.log_status(f"Error with {file}: {str(e)}")
                        
                        status_msg = f"✓ Uploaded {count} images to {actual_name}"
                        if failed:
                            status_msg += f"\n⚠ Failed ({len(failed)}): {', '.join(failed[:3])}"
                        
                        ui.log_status(status_msg)
                        return status_msg
                        
                    except Exception as e:
                        msg = f"✗ Upload error: {str(e)}"
                        ui.log_status(msg)
                        return msg
                
                # Connect events
                create_dataset_btn.click(
                    fn=ui.create_dataset,
                    inputs=dataset_input,
                    outputs=None
                ).then(
                    fn=refresh_datasets,
                    outputs=datasets_list
                )
                
                refresh_btn.click(
                    fn=refresh_datasets,
                    outputs=datasets_list
                )
                
                datasets_list.change(
                    fn=on_dataset_select,
                    inputs=datasets_list,
                    outputs=images_gallery
                )
                
                # Track gallery selection
                images_gallery.select(
                    fn=on_gallery_select,
                    outputs=selected_image_index
                )
                
                # Handle image deletion from custom delete button
                delete_btn.click(
                    fn=delete_selected_image,
                    inputs=[images_gallery, datasets_list, selected_image_index],
                    outputs=[images_gallery, selected_image_index]
                )
                
                upload_btn.upload(
                    fn=upload_images,
                    inputs=[upload_btn, datasets_list],
                    outputs=None
                ).then(
                    fn=on_dataset_select,
                    inputs=datasets_list,
                    outputs=images_gallery
                )
                
                delete_dataset_btn.click(
                    fn=ui.delete_dataset,
                    inputs=datasets_list,
                    outputs=datasets_list
                ).then(
                    fn=lambda: [],
                    outputs=images_gallery
                )
            
            # ============= LoRA Training Tab =============
            with gr.Tab("🎯 LoRA Training"):
                # ============= Configuration Section =============
                with gr.Group():
                    gr.Markdown("## 1️⃣ Configure LoRA Training")
                    
                    with gr.Column():
                        lora_name = gr.Textbox(
                            label="LoRA Name",
                            placeholder="e.g., my_style_lora",
                            value="custom_lora"
                        )
                        
                        with gr.Row():
                            rank = gr.Slider(
                                label="Rank",
                                minimum=4,
                                maximum=128,
                                value=32,
                                step=4
                            )
                            alpha = gr.Slider(
                                label="Alpha",
                                minimum=8,
                                maximum=256,
                                value=64,
                                step=8
                            )
                        
                        with gr.Row():
                            learning_rate = gr.Number(
                                label="Learning Rate",
                                value=0.0001
                            )
                            batch_size = gr.Slider(
                                label="Batch Size",
                                minimum=1,
                                maximum=16,
                                value=1,
                                step=1
                            )
                        
                        with gr.Row():
                            num_epochs = gr.Slider(
                                label="Number of Epochs",
                                minimum=10,
                                maximum=1000,
                                value=100,
                                step=10
                            )
                            steps_per_epoch = gr.Slider(
                                label="Steps per Epoch",
                                minimum=100,
                                maximum=5000,
                                value=1000,
                                step=100
                            )
                        
                        create_config_btn = gr.Button("Create Configuration", size="lg", variant="primary")
                        config_output = gr.Textbox(
                            label="Configuration",
                            lines=12,
                            max_lines=15,
                            interactive=False
                        )
                    
                    create_config_btn.click(
                        fn=ui.create_lora_config,
                        inputs=[lora_name, rank, alpha, learning_rate, num_epochs, batch_size, steps_per_epoch],
                        outputs=config_output
                    )
                
                gr.Markdown("---")
                
                # ============= Training Execution Section =============
                with gr.Group():
                    gr.Markdown("## 2️⃣ Train LoRA")
                    
                    with gr.Column():
                        # Refresh LoRA list
                        with gr.Row():
                            lora_select = gr.Dropdown(
                                label="Select LoRA to Train",
                                choices=["No LoRAs available"],
                                value="No LoRAs available"
                            )
                            refresh_loras_btn = gr.Button("🔄 Refresh", size="sm")
                        
                        with gr.Row():
                            train_btn = gr.Button("▶️ Start Training", size="lg", variant="primary")
                            stop_btn = gr.Button("⏹️ Stop Training", size="lg", variant="stop")
                        
                        training_output = gr.Textbox(
                            label="Training Status",
                            lines=12,
                            max_lines=15,
                            interactive=False
                        )
                        
                        # Status update interval
                        training_status = gr.State({})
                        
                        def refresh_lora_list():
                            """Get list of available LoRAs"""
                            if not ui.loras_dir.exists():
                                return gr.update(choices=["No LoRAs available"], value="No LoRAs available")
                            
                            loras = [d.name for d in ui.loras_dir.iterdir() if d.is_dir()]
                            if not loras:
                                return gr.update(choices=["No LoRAs available"], value="No LoRAs available")
                            
                            return gr.update(choices=loras, value=loras[0])
                        
                        refresh_loras_btn.click(
                            fn=refresh_lora_list,
                            outputs=lora_select
                        )
                        
                        def start_training(lora_name):
                            """Start training"""
                            if lora_name == "No LoRAs available":
                                return "✗ No LoRA selected. Please create a configuration first."
                            
                            msg, process_info = ui.train_lora(lora_name)
                            return msg
                        
                        train_btn.click(
                            fn=start_training,
                            inputs=lora_select,
                            outputs=training_output
                        )
                        
                        stop_btn.click(
                            fn=ui.stop_training,
                            inputs=lora_select,
                            outputs=training_output
                        )
                        
                        # Auto-refresh training status every 5 seconds
                        timer = gr.Timer(value=5)
                        timer.tick(
                            fn=ui.get_training_status,
                            inputs=lora_select,
                            outputs=training_output
                        )
                
                gr.Markdown("---")
                
                # ============= Testing/Inference Section =============
                with gr.Group():
                    gr.Markdown("## 3️⃣ Test LoRA (Generate Images)")
                    
                    with gr.Column():
                        # Refresh LoRA list for testing
                        with gr.Row():
                            test_lora_select = gr.Dropdown(
                                label="Select Trained LoRA",
                                choices=["No LoRAs available"],
                                value="No LoRAs available"
                            )
                            refresh_test_loras_btn = gr.Button("🔄 Refresh", size="sm")
                        
                        test_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="e.g., a beautiful landscape in the style of this LoRA",
                            value="a beautiful landscape"
                        )
                        
                        with gr.Row():
                            test_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=5
                            )
                            test_guidance = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.5
                            )
                        
                        test_btn = gr.Button("🎨 Generate Image", size="lg", variant="primary")
                        
                        with gr.Row():
                            test_output_image = gr.Image(
                                label="Generated Image",
                                type="pil"
                            )
                            test_output_text = gr.Textbox(
                                label="Generation Status",
                                interactive=False
                            )
                        
                        def refresh_test_lora_list():
                            """Get list of trained LoRAs"""
                            if not ui.loras_dir.exists():
                                return gr.update(choices=["No LoRAs available"], value="No LoRAs available")
                            
                            loras = [d.name for d in ui.loras_dir.iterdir() if d.is_dir()]
                            if not loras:
                                return gr.update(choices=["No LoRAs available"], value="No LoRAs available")
                            
                            return gr.update(choices=loras, value=loras[0])
                        
                        refresh_test_loras_btn.click(
                            fn=refresh_test_lora_list,
                            outputs=test_lora_select
                        )
                        
                        def generate_image(lora_name, prompt, steps, guidance):
                            """Generate image with LoRA"""
                            if lora_name == "No LoRAs available":
                                return None, "✗ No LoRA selected. Please train one first."
                            
                            msg, image = ui.test_lora(lora_name, prompt, int(steps), float(guidance))
                            return image, msg
                        
                        test_btn.click(
                            fn=generate_image,
                            inputs=[test_lora_select, test_prompt, test_steps, test_guidance],
                            outputs=[test_output_image, test_output_text]
                        )
                
                gr.Markdown("---")
                
                # ============= LoRA Management Section =============
                with gr.Group():
                    gr.Markdown("## 📋 Manage LoRAs")
                    
                    with gr.Column():
                        list_btn = gr.Button("List All LoRAs", size="lg")
                        lora_list = gr.Textbox(
                            label="LoRAs",
                            lines=8,
                            interactive=False
                        )
                    
                    list_btn.click(
                        fn=ui.list_loras,
                        outputs=lora_list
                    )
            
            # ============= Image-to-LoRA (i2L) Tab =============
            with gr.Tab("🖼️ Image-to-LoRA (i2L)"):
                gr.Markdown("""
                # Z-Image-i2L: Convert Style Images to LoRA
                
                Select a dataset and convert its images directly into a LoRA model that captures that style.
                
                **How it works:**
                1. Select a dataset with 2+ images showing a consistent style
                2. Generate a LoRA from those images (one-shot, no training needed!)
                3. Use the generated LoRA to create images with that style
                
                **Recommended negative prompt for best quality:**
                - Chinese: 泛黄，发绿，模糊，低分辨率，低质量图像，扭曲的肢体，诡异的外观，丑陋，AI感，噪点，网格感，JPEG压缩条纹，异常的肢体，水印，乱码，意义不明的字符
                - English: Yellowed, green-tinted, blurry, low-resolution, low-quality image, distorted limbs, eerie appearance, ugly, AI-looking, noise, grid-like artifacts, JPEG compression artifacts, abnormal limbs, watermark, garbled text, meaningless characters
                """)
                
                # ============= Step 1: Select Dataset =============
                with gr.Group():
                    gr.Markdown("## Step 1️⃣ Select Dataset")
                    
                    with gr.Column():
                        gr.Markdown("Choose a dataset with style reference images")
                        
                        # Refresh datasets list
                        with gr.Row():
                            i2l_dataset_select = gr.Dropdown(
                                label="Select Dataset",
                                choices=ui.list_datasets() or ["No datasets yet"],
                                value=(ui.list_datasets() or ["No datasets yet"])[0]
                            )
                            i2l_refresh_datasets_btn = gr.Button("🔄 Refresh", size="sm")
                        
                        i2l_images_gallery = gr.Gallery(
                            label="Dataset Images",
                            columns=3,
                            rows=2,
                            object_fit="scale-down",
                            height="auto",
                            show_label=False
                        )
                        
                        i2l_selected_dataset_state = gr.State((ui.list_datasets() or ["No datasets yet"])[0])
                        
                        def refresh_i2l_datasets():
                            """Refresh available datasets"""
                            choices = ui.list_datasets() or ["No datasets yet"]
                            return gr.update(choices=choices, value=choices[0])
                        
                        i2l_refresh_datasets_btn.click(
                            fn=refresh_i2l_datasets,
                            outputs=i2l_dataset_select
                        )
                        
                        def on_i2l_dataset_select(dataset_name):
                            """Load images from selected dataset and update state"""
                            if not dataset_name or dataset_name == "No datasets yet":
                                return [], dataset_name
                            
                            images = ui.get_dataset_images(dataset_name)
                            actual_name = dataset_name.split(" (")[0]
                            
                            if not images:
                                return [], dataset_name
                            
                            # Return full paths for gallery component
                            image_paths = [str(ui.loras_dir / actual_name / "images" / img) for img in images]
                            return image_paths, dataset_name
                        
                        i2l_dataset_select.change(
                            fn=on_i2l_dataset_select,
                            inputs=i2l_dataset_select,
                            outputs=[i2l_images_gallery, i2l_selected_dataset_state]
                        )
                        
                        # Load initial dataset on page load
                        demo.load(
                            fn=on_i2l_dataset_select,
                            inputs=i2l_dataset_select,
                            outputs=[i2l_images_gallery, i2l_selected_dataset_state]
                        )
                
                # ============= Step 2: Generate LoRA =============
                with gr.Group():
                    gr.Markdown("## Step 2️⃣ Generate LoRA from Dataset")
                    
                    with gr.Column():
                        i2l_lora_name = gr.Textbox(
                            label="LoRA Name",
                            placeholder="e.g., watercolor_style, anime_aesthetic",
                            value="my_style_lora"
                        )
                        
                        i2l_generate_btn = gr.Button("🚀 Generate LoRA", size="lg", variant="primary")
                        
                        i2l_generate_output = gr.Textbox(
                            label="Generation Status",
                            lines=8,
                            interactive=False
                        )
                        
                        def generate_lora_from_dataset(dataset_name, lora_name):
                            """Generate LoRA from selected dataset"""
                            if not dataset_name or dataset_name == "No datasets yet":
                                return "✗ No dataset selected"
                            
                            # Get images from dataset
                            actual_name = dataset_name.split(" (")[0]
                            images_dir = ui.loras_dir / actual_name / "images"
                            
                            if not images_dir.exists():
                                return "✗ Dataset images not found"
                            
                            image_files = []
                            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                                image_files.extend(images_dir.glob(ext))
                                image_files.extend(images_dir.glob(ext.upper()))
                            
                            # Remove duplicates
                            unique_images = {}
                            for img in image_files:
                                key = img.name.lower()
                                if key not in unique_images:
                                    unique_images[key] = str(img)
                            
                            image_paths = list(unique_images.values())
                            
                            if len(image_paths) == 0:
                                return "✗ No images found in dataset"
                            
                            if len(image_paths) == 1:
                                return f"⚠ Warning: Only 1 image in dataset. 2+ images recommended for better quality."
                            
                            msg, lora_path = ui.generate_lora_from_images(image_paths, lora_name)
                            return msg
                        
                        i2l_generate_btn.click(
                            fn=generate_lora_from_dataset,
                            inputs=[i2l_selected_dataset_state, i2l_lora_name],
                            outputs=i2l_generate_output
                        )
                
                # ============= Step 3: Test Generated LoRA =============
                with gr.Group():
                    gr.Markdown("## Step 3️⃣ Generate Images with LoRA")
                    
                    with gr.Column():
                        gr.Markdown("Select a generated LoRA and create images with that style")
                        
                        # Refresh LoRAs list
                        with gr.Row():
                            i2l_lora_select = gr.Dropdown(
                                label="Generated LoRAs",
                                choices=ui.list_generated_loras() or ["No generated LoRAs yet"],
                                value=(ui.list_generated_loras() or ["No generated LoRAs yet"])[0]
                            )
                            i2l_refresh_loras_btn = gr.Button("🔄 Refresh", size="sm")
                        
                        i2l_test_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="e.g., a cat, a landscape, a person",
                            value="a beautiful landscape"
                        )
                        
                        # Advanced settings for image generation
                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Row():
                                i2l_test_seed = gr.Number(
                                    label="Seed",
                                    value=0,
                                    precision=0
                                )
                                i2l_test_steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=10,
                                    maximum=100,
                                    step=1,
                                    value=50
                                )
                            
                            with gr.Row():
                                i2l_test_cfg = gr.Slider(
                                    label="CFG Scale",
                                    minimum=1.0,
                                    maximum=20.0,
                                    step=0.5,
                                    value=4.0
                                )
                                i2l_test_sigma = gr.Slider(
                                    label="Sigma Shift",
                                    minimum=1.0,
                                    maximum=20.0,
                                    step=0.5,
                                    value=8.0
                                )
                            
                            i2l_test_negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="Enter negative prompt...",
                                value="泛黄，发绿，模糊，低分辨率，低质量图像，扭曲的肢体，诡异的外观，丑陋，AI感，噪点，网格感，JPEG压缩条纹，异常的肢体，水印，乱码，意义不明的字符",
                                lines=3
                            )
                        
                        i2l_test_btn = gr.Button("🎨 Generate Image", size="lg", variant="primary")
                        
                        i2l_unload_btn = gr.Button("🗑️ Unload Models", size="lg", variant="secondary")
                        
                        with gr.Row():
                            i2l_test_output_image = gr.Image(
                                label="Generated Image",
                                type="pil"
                            )
                            i2l_test_output_text = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                        
                        def refresh_i2l_lora_list():
                            """Refresh list of generated LoRAs"""
                            choices = ui.list_generated_loras() or ["No generated LoRAs yet"]
                            return gr.update(choices=choices, value=choices[0])
                        
                        i2l_refresh_loras_btn.click(
                            fn=refresh_i2l_lora_list,
                            outputs=i2l_lora_select
                        )
                        
                        def generate_image_i2l(lora_name, prompt, seed, steps, cfg, sigma, negative_prompt):
                            """Generate image with i2L LoRA"""
                            if not lora_name or lora_name == "No generated LoRAs yet":
                                return None, "✗ Please select a LoRA first"
                            
                            seed = int(seed) if seed else 0
                            steps = int(steps) if steps else 50
                            cfg = float(cfg) if cfg else 4.0
                            sigma = float(sigma) if sigma else 8.0
                            
                            msg, image = ui.generate_image_with_i2l_lora(
                                lora_name, prompt, seed, 
                                steps=steps, cfg_scale=cfg, sigma_shift=sigma,
                                negative_prompt=negative_prompt
                            )
                            return image, msg
                        
                        i2l_test_btn.click(
                            fn=generate_image_i2l,
                            inputs=[i2l_lora_select, i2l_test_prompt, i2l_test_seed, 
                                   i2l_test_steps, i2l_test_cfg, i2l_test_sigma, 
                                   i2l_test_negative_prompt],
                            outputs=[i2l_test_output_image, i2l_test_output_text]
                        )
                        
                        i2l_unload_btn.click(
                            fn=ui.unload_models,
                            outputs=i2l_test_output_text
                        )
            
            # ============= Status Tab =============
            with gr.Tab("📊 Status"):
                gr.Markdown("### System Status")
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=15,
                    max_lines=20,
                    interactive=False
                )
                
                refresh_btn = gr.Button("Refresh Status", size="lg")
                refresh_btn.click(
                    fn=ui.get_status,
                    outputs=status_output
                )
                
                # Auto-load on tab change
                demo.load(
                    fn=ui.get_status,
                    outputs=status_output
                )
            
            # ============= Help Tab =============
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                ## DiffSynth-Studio UI Guide
                
                ### 📌 Smart Tab Visibility
                The UI automatically hides tabs when their work is complete:
                - **Installation tab**: Hidden when all components are properly installed
                - **Model Download tab**: Hidden when Z-Image-i2L is downloaded
                - **Other tabs**: Always visible for your workflow
                
                ### 🔧 Installation (Automatic)
                Installation happens automatically on startup:
                - DiffSynth-Studio repository is cloned from GitHub
                - Python dependencies are installed (including ModelScope)
                - Git LFS is initialized
                
                If the Installation tab is visible, click it to verify setup status.
                
                ### ⬇️ Model Download
                Download the Z-Image-i2L model needed for training:
                - **Z-Image-i2L**: Advanced image-to-image generation (~7GB)
                
                If the Model Download tab is visible, click "Download All Models". Uses ModelScope API first, Git LFS as fallback.
                
                ### 📁 Datasets
                Manage your training datasets:
                1. **Create Dataset**: Give it a name (e.g., "anime_style")
                2. **Upload Images**: Select the dataset and upload training images
                3. **Preview**: View uploaded images in the gallery
                4. **Delete**: Remove individual images or entire datasets
                
                Supported formats: JPG, JPEG, PNG, WebP
                
                ### 🎯 LoRA Training
                1. Set your desired LoRA configuration parameters
                2. Adjust hyperparameters as needed:
                   - **Rank**: Controls adapter complexity (32 is default)
                   - **Alpha**: LoRA influence scaling (typically 2x rank)
                   - **Learning Rate**: Gradient descent step size
                   - **Batch Size**: Samples per training step
                   - **Epochs**: Training passes through dataset
                3. Click "Create Configuration" to generate the config file
                4. Use the generated configuration for training
                
                ### 📊 Status
                Monitor your setup and view downloaded models/LoRAs
                
                ---
                
                **Quick Workflow:**
                1. Check Installation tab → verify setup
                2. Model Download tab → download models (takes 10-30 min)
                3. Datasets tab → create dataset & upload images
                4. LoRA Training tab → configure and create training config
                5. Run training script with the generated config
                
                **Resources:**
                - [DiffSynth-Studio Repo](https://github.com/modelscope/DiffSynth-Studio)
                - [Z-Image-i2L Model](https://www.modelscope.cn/models/DiffSynth-Studio/Z-Image-i2L)
                - [LoRA Research](https://arxiv.org/abs/2106.09685)
                """)
        
        # Load initial gallery display when interface loads
        def load_initial_gallery():
            current_dataset = dataset_choices[0] if dataset_choices else "No datasets yet"
            return on_dataset_select(current_dataset)
        
        demo.load(
            fn=load_initial_gallery,
            outputs=images_gallery
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        head="",
        css="""
        /* Gallery takes 100% width of parent container */
        .gallery-container {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        .grid-wrap {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        .grid-container {
            width: 100% !important;
            max-width: 100% !important;
        }

        .gallery-item {
            max-height: 250px
            }
        
        /* Constrain gallery images to max height while maintaining aspect ratio */
        .gallery-item img {
            max-height: 250px;
            width: auto;
            object-fit: contain;
            margin: 0 auto;
        }
        
        /* Individual gallery item */
        .gallery-item {
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #1e1e1e;
            border-radius: 4px;
            overflow: hidden;
            min-height: 200px;
        }
        """
    )
