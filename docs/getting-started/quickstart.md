# Quick Start

Get up and running with the COCO to 3D Synthetic Pipeline in minutes.

## Step 1: Prepare COCO Dataset

Place your COCO-format annotation file in the `data/coco/` directory:

```bash
data/coco/
├── annotations.json
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Step 2: Process COCO Annotations

```python
from src.coco_processing import COCOProcessor

# Initialize processor
processor = COCOProcessor('data/coco/annotations.json')

# Load and process annotations
annotations = processor.load_annotations()
masks = processor.extract_masks()
```

## Step 3: Generate 3D Models

```python
from src.3d_generation import ModelGenerator

# Initialize 3D generator
generator = ModelGenerator()

# Convert masks to 3D models
for instance_id, mask in masks.items():
    model_3d = generator.generate_from_mask(mask)
    generator.save_model(model_3d, f'data/3d_models/model_{instance_id}.obj')
```

## Step 4: Render Synthetic Images

```python
from src.synthetic_rendering import SyntheticRenderer

# Initialize renderer
renderer = SyntheticRenderer()

# Load 3D model and render from different viewpoints
model = renderer.load_model('data/3d_models/model_001.obj')
for angle in range(0, 360, 45):
    image = renderer.render(model, angle=angle)
    renderer.save_image(image, f'data/synthetic_images/synthetic_{angle}.png')
```

## Example Pipeline

Here's a complete example that runs the full pipeline:

```python
from src.coco_processing import COCOProcessor
from src.3d_generation import ModelGenerator
from src.synthetic_rendering import SyntheticRenderer

# Stage 1: COCO Processing
processor = COCOProcessor('data/coco/annotations.json')
masks = processor.extract_masks()

# Stage 2: 3D Generation
generator = ModelGenerator()
models_3d = [generator.generate_from_mask(mask) for mask in masks]

# Stage 3: Synthetic Rendering
renderer = SyntheticRenderer()
for i, model in enumerate(models_3d):
    for angle in [0, 90, 180, 270]:
        image = renderer.render(model, angle=angle)
        renderer.save_image(image, f'data/synthetic_images/model_{i}_angle_{angle}.png')

print(f"Generated {len(models_3d) * 4} synthetic images!")
```

## Next Steps

- Learn more about [COCO Processing](../pipeline/coco-processing.md)
- Explore [3D Generation options](../pipeline/3d-generation.md)
- Customize [Synthetic Rendering](../pipeline/synthetic-rendering.md)
