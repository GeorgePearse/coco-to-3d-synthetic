# COCO to 3D Synthetic Pipeline

A pipeline for converting COCO-style instance annotation datasets into 3D models using **OpenLRM** (Large Reconstruction Model).

## Overview

This project implements a complete pipeline for transforming 2D COCO instance segmentations into high-quality 3D models:

1. **COCO Processing**: Extract objects from images using segmentation masks
2. **3D Generation**: Generate 3D models using OpenLRM (single-image-to-3D)
3. **Mesh Processing**: Convert and optimize meshes for various applications

### Powered by OpenLRM

We use [OpenLRM](https://github.com/3DTopia/OpenLRM), a state-of-the-art Large Reconstruction Model that generates 3D models from single images in ~5 seconds. OpenLRM uses a 500M+ parameter transformer architecture trained on 1M+ 3D objects.

## Project Structure

```
coco-to-3d-synthetic/
├── data/
│   ├── coco/              # COCO dataset files
│   ├── 3d_models/         # Generated 3D models
│   └── synthetic_images/  # Rendered synthetic images
├── src/
│   ├── coco_processing/   # COCO data loading and processing
│   ├── 3d_generation/     # 3D model generation from annotations
│   └── synthetic_rendering/ # Synthetic image rendering
├── notebooks/             # Jupyter notebooks for experimentation
├── configs/              # Configuration files
└── README.md
```

## Setup

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/GeorgePearse/coco-to-3d-synthetic.git
cd coco-to-3d-synthetic

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt

# For full OpenLRM support (optional, includes xFormers)
pip install -r requirements-openlrm.txt
```

### GPU Requirements

- **Minimum**: 12GB VRAM (RTX 3060 Ti, RTX 3080)
- **Recommended**: 16-24GB VRAM (RTX 3090, RTX 4090, A5000)
- **Best**: A100 (80GB) for large batches

CPU-only mode is supported but significantly slower.

## Quick Start

### Process a single category from COCO

```bash
python scripts/coco_to_3d_pipeline.py \
  --config configs/coco_pipeline.yaml \
  --category car \
  --max-images 10
```

This will:
1. Extract 10 car instances from COCO using segmentation masks
2. Generate 3D models using OpenLRM
3. Convert meshes to OBJ, GLB, and PLY formats
4. Save results to `output/car/`

### Python API Usage

```python
from src.coco_processing import COCOPreprocessor
from src.3d_generation import OpenLRMGenerator, MeshConverter

# 1. Preprocess COCO image
preprocessor = COCOPreprocessor('annotations.json', 'images/')
image, metadata = preprocessor.extract_and_prepare(
    image_id=12345,
    output_path='preprocessed.png',
    return_metadata=True
)

# 2. Generate 3D model
generator = OpenLRMGenerator(model_size='base')
result = generator.generate_3d(
    image_path='preprocessed.png',
    output_dir='output',
    camera_distance=2.0
)

# 3. Convert mesh formats
converter = MeshConverter()
converter.export_multiple_formats(
    input_path=result['mesh'],
    output_dir='output/converted',
    formats=['obj', 'glb', 'stl']
)
```

## Pipeline Stages

### Stage 1: COCO Processing

**Segmentation-Based Extraction**
- Load COCO annotation files with `pycocotools`
- Extract objects using **segmentation masks** (not bounding boxes)
- Remove backgrounds and center objects
- Generate RGBA images optimized for 3D reconstruction

**Features:**
- Polygon and RLE segmentation support
- Optional background removal with `rembg`
- Automatic centering and padding
- Batch processing by category

### Stage 2: 3D Model Generation with OpenLRM

**Single-Image-to-3D**
- Generate 3D models using OpenLRM transformer architecture
- Three model sizes: small (446M), base (1.04G), large (1.81G)
- ~5-10 seconds per model on A100 GPU
- Outputs: PLY mesh + 360° rotation video

**OpenLRM Models:**
- `openlrm-mix-base-1.1` (recommended): Trained on Objaverse + MVImgNet
- `openlrm-obj-base-1.1`: Trained on Objaverse only
- Better generalization with mixed training data

### Stage 3: Mesh Processing

**Format Conversion**
- Convert PLY to OBJ, GLB, STL, GLTF, DAE, OFF
- Mesh cleaning (merge vertices, fix normals, remove degenerate faces)
- Polygon reduction for web/mobile applications
- Batch conversion utilities

**Supported Formats:**
- **OBJ**: Widely supported, includes materials
- **GLB**: Modern web format (Three.js, Babylon.js)
- **PLY**: Native OpenLRM output
- **STL**: 3D printing
- **GLTF**: AR/VR applications

## Project Structure

```
coco-to-3d-synthetic/
├── third_party/
│   └── OpenLRM/              # OpenLRM submodule
├── src/
│   ├── coco_processing/
│   │   └── preprocessing.py  # COCO segmentation extraction
│   └── 3d_generation/
│       ├── openlrm_generator.py   # OpenLRM wrapper
│       └── mesh_converter.py      # Format conversion
├── scripts/
│   └── coco_to_3d_pipeline.py    # End-to-end pipeline
├── configs/
│   └── coco_pipeline.yaml         # Pipeline configuration
├── data/
│   ├── coco/                      # COCO dataset
│   ├── 3d_models/                 # Generated 3D models
│   └── synthetic_images/          # Future: Rendered images
├── docs/                          # Documentation (MkDocs)
├── requirements.txt               # Core dependencies
└── requirements-openlrm.txt       # Full OpenLRM dependencies
```

## Documentation

Full documentation is available at: **https://georgepearse.github.io/coco-to-3d-synthetic/**

Topics covered:
- [Installation Guide](https://georgepearse.github.io/coco-to-3d-synthetic/getting-started/installation/)
- [Pipeline Overview](https://georgepearse.github.io/coco-to-3d-synthetic/pipeline/overview/)
- [API Reference](https://georgepearse.github.io/coco-to-3d-synthetic/api/coco-processing/)
- [Research References](https://georgepearse.github.io/coco-to-3d-synthetic/references/) - SOTA methods for NeRF and Gaussian Splatting

## Requirements

- Python 3.8+
- PyTorch >= 2.1.2
- CUDA 11.8+ (for GPU support)
- 12-24GB VRAM (depending on model size)
- pycocotools, trimesh, transformers, rembg

See `requirements.txt` and `requirements-openlrm.txt` for complete lists.

## License

**Code**: MIT License

**OpenLRM Model Weights**: CC-BY-NC 4.0 (Non-Commercial Only)

**Important**: OpenLRM model weights are licensed for **research and non-commercial use only**. Commercial use requires alternative models or licensing agreements.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{hong2023lrm,
  title={LRM: Large Reconstruction Model for Single Image to 3D},
  author={Hong, Yicong and Zhang, Kai and Gu, Jiuxiang and Bi, Sai and Zhou, Yang and Liu, Difan and Liu, Feng and Sunkavalli, Kalyan and Bui, Trung and Tan, Hao},
  journal={arXiv preprint arXiv:2311.04400},
  year={2023}
}
```

## Acknowledgments

- [OpenLRM](https://github.com/3DTopia/OpenLRM) - Large Reconstruction Model
- [COCO Dataset](https://cocodataset.org/) - Common Objects in Context
- [pycocotools](https://github.com/cocodataset/cocoapi) - COCO API
- [trimesh](https://github.com/mikedh/trimesh) - Mesh processing
