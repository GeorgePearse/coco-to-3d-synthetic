# Installation

This guide will help you set up the COCO to 3D Synthetic Pipeline on your system.

## Prerequisites

- Python 3.8 or higher
- Git
- CUDA-capable GPU (recommended for 3D processing)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/GeorgePearse/coco-to-3d-synthetic.git
cd coco-to-3d-synthetic
```

### 2. Create a Virtual Environment

=== "Linux/macOS"

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

=== "Windows"

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dependencies

The project requires the following main dependencies:

- **PyTorch**: Deep learning framework for 3D processing
- **OpenCV**: Computer vision operations
- **pycocotools**: COCO dataset utilities
- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **Matplotlib**: Visualization

## Verification

To verify your installation, run:

```python
import torch
import cv2
from pycocotools.coco import COCO
print("All dependencies installed successfully!")
```

## GPU Support

For optimal performance with 3D generation, ensure you have CUDA installed:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is not available, the pipeline will fall back to CPU processing (slower).

## Next Steps

Once installation is complete, proceed to the [Quick Start Guide](quickstart.md) to run your first pipeline.
