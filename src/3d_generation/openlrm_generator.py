"""
OpenLRM 3D Model Generator

This module provides a wrapper around OpenLRM for generating 3D models from single images.
OpenLRM is a Large Reconstruction Model that uses transformers to predict 3D neural radiance
fields from 2D images.

Example:
    >>> from src.3d_generation import OpenLRMGenerator
    >>> generator = OpenLRMGenerator(model_size='base')
    >>> result = generator.generate_3d('input.png', 'output_dir')
    >>> print(f"Generated mesh: {result['mesh']}")

Note:
    OpenLRM model weights are licensed under CC-BY-NC 4.0 (non-commercial use only).
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch

# Add OpenLRM to Python path
OPENLRM_PATH = Path(__file__).parent.parent.parent / "third_party" / "OpenLRM"
if OPENLRM_PATH.exists():
    sys.path.insert(0, str(OPENLRM_PATH))

try:
    from openlrm.runners import REGISTRY_RUNNERS
    from omegaconf import OmegaConf
except ImportError as e:
    raise ImportError(
        f"Failed to import OpenLRM. Please ensure it's installed as a submodule. Error: {e}"
    )


class OpenLRMGenerator:
    """
    Wrapper class for OpenLRM 3D generation.

    This class provides a simplified interface to OpenLRM for generating 3D models
    from single 2D images. It handles model initialization, inference, and output
    management.

    Attributes:
        model_size (str): Size of the model ('small', 'base', or 'large')
        device (str): Compute device ('cuda' or 'cpu')
        config (OmegaConf): Model configuration
        inferrer: OpenLRM inference runner

    Args:
        model_size (str): Model size variant. Options:
            - 'small': 446M parameters, 224px input (fastest, lowest VRAM)
            - 'base': 1.04G parameters, 336px input (recommended)
            - 'large': 1.81G parameters, 448px input (highest quality, most VRAM)
        model_variant (str): Training data variant. Options:
            - 'mix': Trained on Objaverse + MVImgNet (recommended, better generalization)
            - 'obj': Trained on Objaverse only
        device (str): Compute device ('cuda' or 'cpu')
        enable_logging (bool): Enable detailed logging
        config_override (dict): Optional configuration overrides
    """

    MODEL_CONFIGS = {
        'small': {
            'config_path': 'configs/infer-s.yaml',
            'obj_model': 'zxhezexin/openlrm-obj-small-1.1',
            'mix_model': 'zxhezexin/openlrm-mix-small-1.1'
        },
        'base': {
            'config_path': 'configs/infer-b.yaml',
            'obj_model': 'zxhezexin/openlrm-obj-base-1.1',
            'mix_model': 'zxhezexin/openlrm-mix-base-1.1'
        },
        'large': {
            'config_path': 'configs/infer-l.yaml',
            'obj_model': 'zxhezexin/openlrm-obj-large-1.1',
            'mix_model': 'zxhezexin/openlrm-mix-large-1.1'
        }
    }

    def __init__(
        self,
        model_size: str = 'base',
        model_variant: str = 'mix',
        device: str = 'cuda',
        enable_logging: bool = True,
        config_override: Optional[Dict] = None
    ):
        """Initialize OpenLRM generator."""
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        self.logger = logging.getLogger(__name__)
        self.model_size = model_size
        self.model_variant = model_variant
        self.device = device

        # Validate inputs
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Choose from: {list(self.MODEL_CONFIGS.keys())}"
            )

        if model_variant not in ['obj', 'mix']:
            raise ValueError(f"Invalid model_variant '{model_variant}'. Choose 'obj' or 'mix'.")

        # Check CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'

        # Load configuration
        model_config = self.MODEL_CONFIGS[model_size]
        config_path = OPENLRM_PATH / model_config['config_path']

        if not config_path.exists():
            raise FileNotFoundError(
                f"OpenLRM config not found at {config_path}. "
                "Ensure OpenLRM submodule is properly initialized."
            )

        self.config = OmegaConf.load(str(config_path))
        self.config.model_name = (
            model_config['mix_model'] if model_variant == 'mix'
            else model_config['obj_model']
        )

        # Apply config overrides
        if config_override:
            self.config = OmegaConf.merge(self.config, OmegaConf.create(config_override))

        # Initialize inferrer
        self.logger.info(f"Initializing OpenLRM ({model_size}, {model_variant} variant)")
        inferrer_class = REGISTRY_RUNNERS.get("infer.lrm")
        self.inferrer = inferrer_class(self.config)

        self.logger.info(
            f"OpenLRM initialized successfully. "
            f"Model: {self.config.model_name}, Device: {self.device}"
        )

    def generate_3d(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        camera_distance: float = 2.0,
        export_video: bool = True,
        export_mesh: bool = True,
        mesh_filename: str = "mesh.ply",
        video_filename: str = "rotation.mp4"
    ) -> Dict[str, Optional[str]]:
        """
        Generate 3D model from a single image.

        Args:
            image_path: Path to input image (PNG, JPG, etc.)
            output_dir: Directory to save outputs
            camera_distance: Camera distance from object (1.0-3.5, default 2.0)
                - Smaller values: Closer view, may clip large objects
                - Larger values: Farther view, better for large objects
            export_video: Whether to generate rotation video
            export_mesh: Whether to generate 3D mesh
            mesh_filename: Output mesh filename (default: mesh.ply)
            video_filename: Output video filename (default: rotation.mp4)

        Returns:
            Dictionary containing:
                - success (bool): Whether generation succeeded
                - mesh (str|None): Path to generated mesh (if export_mesh=True)
                - video (str|None): Path to generated video (if export_video=True)
                - error (str|None): Error message (if failed)

        Raises:
            FileNotFoundError: If input image doesn't exist
            RuntimeError: If generation fails
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        # Validate input
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup output paths
        video_path = output_dir / video_filename if export_video else None
        mesh_path = output_dir / mesh_filename if export_mesh else None

        try:
            self.logger.info(f"Processing image: {image_path}")

            with self.inferrer:
                self.inferrer.infer_single(
                    image_path=str(image_path),
                    source_cam_dist=camera_distance,
                    export_video=export_video,
                    export_mesh=export_mesh,
                    dump_video_path=str(video_path) if video_path else None,
                    dump_mesh_path=str(mesh_path) if mesh_path else None
                )

            self.logger.info(f"Successfully generated 3D from {image_path.name}")

            return {
                'success': True,
                'mesh': str(mesh_path) if mesh_path and mesh_path.exists() else None,
                'video': str(video_path) if video_path and video_path.exists() else None,
                'error': None
            }

        except torch.cuda.OutOfMemoryError:
            error_msg = (
                "GPU out of memory. Try:\n"
                "1. Use a smaller model size ('small' instead of 'base')\n"
                "2. Reduce frame_size in config\n"
                "3. Close other GPU applications"
            )
            self.logger.error(error_msg)
            return {'success': False, 'mesh': None, 'video': None, 'error': error_msg}

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'mesh': None, 'video': None, 'error': error_msg}

    def generate_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        camera_distance: float = 2.0,
        export_video: bool = True,
        export_mesh: bool = True,
        create_subdirs: bool = True
    ) -> List[Dict]:
        """
        Generate 3D models for multiple images.

        Args:
            image_paths: List of input image paths
            output_dir: Base output directory
            camera_distance: Camera distance for all images
            export_video: Whether to generate videos
            export_mesh: Whether to generate meshes
            create_subdirs: Create separate subdirectory for each image

        Returns:
            List of result dictionaries, one per image
        """
        output_dir = Path(output_dir)
        results = []

        self.logger.info(f"Processing batch of {len(image_paths)} images")

        for i, img_path in enumerate(image_paths):
            img_path = Path(img_path)

            # Create subdirectory if requested
            if create_subdirs:
                img_output_dir = output_dir / img_path.stem
            else:
                img_output_dir = output_dir

            try:
                result = self.generate_3d(
                    image_path=img_path,
                    output_dir=img_output_dir,
                    camera_distance=camera_distance,
                    export_video=export_video,
                    export_mesh=export_mesh,
                    mesh_filename=f"{img_path.stem}_mesh.ply" if not create_subdirs else "mesh.ply",
                    video_filename=f"{img_path.stem}_video.mp4" if not create_subdirs else "rotation.mp4"
                )

                result['image_path'] = str(img_path)
                result['index'] = i
                results.append(result)

                if result['success']:
                    self.logger.info(
                        f"[{i+1}/{len(image_paths)}] ✓ {img_path.name}"
                    )
                else:
                    self.logger.warning(
                        f"[{i+1}/{len(image_paths)}] ✗ {img_path.name}: {result['error']}"
                    )

            except Exception as e:
                self.logger.error(f"[{i+1}/{len(image_paths)}] ✗ {img_path.name}: {e}")
                results.append({
                    'success': False,
                    'image_path': str(img_path),
                    'index': i,
                    'mesh': None,
                    'video': None,
                    'error': str(e)
                })

        # Summary
        successful = sum(1 for r in results if r['success'])
        self.logger.info(
            f"Batch complete: {successful}/{len(image_paths)} successful"
        )

        return results

    def get_model_info(self) -> Dict:
        """Get information about the current model configuration."""
        return {
            'model_size': self.model_size,
            'model_variant': self.model_variant,
            'model_name': self.config.model_name,
            'device': self.device,
            'input_resolution': getattr(self.config, 'source_size', 'unknown'),
            'render_resolution': getattr(self.config, 'render_size', 'unknown'),
            'cuda_available': torch.cuda.is_available(),
            'gpu_memory_allocated': (
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                if torch.cuda.is_available() else "N/A"
            )
        }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="OpenLRM 3D Generation")
    parser.add_argument("input", type=str, help="Input image path")
    parser.add_argument("output", type=str, help="Output directory")
    parser.add_argument("--model-size", choices=['small', 'base', 'large'],
                       default='base', help="Model size")
    parser.add_argument("--camera-dist", type=float, default=2.0,
                       help="Camera distance (1.0-3.5)")
    parser.add_argument("--no-video", action="store_true",
                       help="Skip video generation")
    parser.add_argument("--no-mesh", action="store_true",
                       help="Skip mesh generation")

    args = parser.parse_args()

    # Initialize generator
    generator = OpenLRMGenerator(model_size=args.model_size)

    # Print model info
    info = generator.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Generate 3D
    print(f"\nGenerating 3D from: {args.input}")
    result = generator.generate_3d(
        image_path=args.input,
        output_dir=args.output,
        camera_distance=args.camera_dist,
        export_video=not args.no_video,
        export_mesh=not args.no_mesh
    )

    print("\nResults:")
    if result['success']:
        print("  ✓ Generation successful!")
        if result['mesh']:
            print(f"  Mesh: {result['mesh']}")
        if result['video']:
            print(f"  Video: {result['video']}")
    else:
        print(f"  ✗ Generation failed: {result['error']}")
