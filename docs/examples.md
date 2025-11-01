# Examples

Complete examples demonstrating various use cases of the COCO to 3D Synthetic Pipeline.

## Basic Pipeline Example

A complete end-to-end pipeline example:

```python
from src.coco_processing import COCOProcessor
from src.3d_generation import ModelGenerator
from src.synthetic_rendering import SyntheticRenderer

# Initialize components
processor = COCOProcessor('data/coco/annotations.json')
generator = ModelGenerator(method='midas')
renderer = SyntheticRenderer(mode='photorealistic')

# Stage 1: Extract masks from COCO
masks = processor.extract_masks(category='car')

# Stage 2: Generate 3D models
models = []
for i, mask in enumerate(masks[:10]):  # Process first 10
    model = generator.generate_from_mask(mask)
    model_path = f'data/3d_models/car_{i}.obj'
    generator.save_model(model, model_path)
    models.append(model_path)

# Stage 3: Render synthetic images
for i, model_path in enumerate(models):
    model = renderer.load_model(model_path)
    for angle in [0, 90, 180, 270]:
        image = renderer.render(model, camera_angle=angle)
        renderer.save_image(
            image,
            f'data/synthetic_images/car_{i}_angle_{angle}.png'
        )

print(f"Generated {len(models) * 4} synthetic images!")
```

## Category-Specific Processing

Process only specific object categories:

```python
from src.coco_processing import COCOProcessor

processor = COCOProcessor('data/coco/annotations.json')

# Process only people
person_masks = processor.extract_masks(category='person')

# Process multiple categories
categories = ['car', 'bicycle', 'motorcycle']
for cat in categories:
    masks = processor.extract_masks(category=cat)
    print(f"Found {len(masks)} {cat} instances")
```

## Multi-Viewpoint Dataset Generation

Generate training data from multiple viewpoints:

```python
from src.synthetic_rendering import SyntheticRenderer
import numpy as np

renderer = SyntheticRenderer(resolution=(640, 480))
model = renderer.load_model('data/3d_models/object.obj')

# Generate 360-degree views
num_views = 36
angles = np.linspace(0, 360, num_views, endpoint=False)

for i, angle in enumerate(angles):
    # Vary camera height and distance
    height = 2.0 + np.sin(np.radians(angle)) * 0.5
    distance = 5.0 + np.cos(np.radians(angle)) * 1.0

    image = renderer.render(
        model,
        camera_angle=angle,
        camera_height=height,
        camera_distance=distance
    )

    renderer.save_image(image, f'views/view_{i:03d}.png')
```

## Custom Lighting Scenarios

Generate data with different lighting conditions:

```python
from src.synthetic_rendering import SyntheticRenderer

renderer = SyntheticRenderer()
model = renderer.load_model('data/3d_models/object.obj')

lighting_presets = ['studio', 'outdoor', 'dramatic', 'soft']

for preset in lighting_presets:
    renderer.set_lighting(type=preset)

    for angle in range(0, 360, 60):
        image = renderer.render(model, camera_angle=angle)
        renderer.save_image(
            image,
            f'lighting/{preset}_angle_{angle}.png'
        )
```

## Background Variation

Generate images with varied backgrounds:

```python
from src.synthetic_rendering import SyntheticRenderer

renderer = SyntheticRenderer()
model = renderer.load_model('data/3d_models/object.obj')

# Solid color backgrounds
colors = [
    (255, 255, 255),  # White
    (128, 128, 128),  # Gray
    (0, 0, 0),        # Black
    (100, 150, 200)   # Light blue
]

for i, color in enumerate(colors):
    renderer.set_background(color=color)
    image = renderer.render(model)
    renderer.save_image(image, f'backgrounds/solid_{i}.png')

# Image backgrounds
import glob
background_images = glob.glob('backgrounds/*.jpg')

for i, bg_path in enumerate(background_images):
    renderer.set_background(image=bg_path)
    image = renderer.render(model)
    renderer.save_image(image, f'backgrounds/composite_{i}.png')
```

## Batch Processing

Efficiently process large datasets:

```python
from src.coco_processing import COCOProcessor
from src.3d_generation import ModelGenerator
from src.synthetic_rendering import SyntheticRenderer
from multiprocessing import Pool

processor = COCOProcessor('data/coco/annotations.json')
generator = ModelGenerator()
renderer = SyntheticRenderer(mode='fast')

# Extract all masks
all_masks = processor.extract_masks()

# Batch generate 3D models
models = generator.batch_generate(
    list(all_masks.values()),
    num_workers=8
)

# Save models
model_paths = []
for i, model in enumerate(models):
    path = f'data/3d_models/batch_{i}.obj'
    generator.save_model(model, path)
    model_paths.append(path)

# Batch render
images = renderer.batch_render(
    model_paths,
    num_viewpoints=8,
    num_workers=4
)

print(f"Processed {len(models)} models, generated {len(images)} images")
```

## Depth Map Generation

Generate depth maps for training:

```python
from src.synthetic_rendering import SyntheticRenderer

renderer = SyntheticRenderer()
model = renderer.load_model('data/3d_models/object.obj')

# Render RGB and depth pairs
for i in range(10):
    angle = i * 36

    # RGB image
    rgb = renderer.render(model, camera_angle=angle)
    renderer.save_image(rgb, f'pairs/rgb_{i}.png')

    # Depth map
    depth = renderer.render_depth(model)
    renderer.save_image(depth, f'pairs/depth_{i}.exr', format='exr')
```

## Quality vs Speed Comparison

Compare different rendering modes:

```python
from src.synthetic_rendering import SyntheticRenderer
import time

model_path = 'data/3d_models/object.obj'

modes = ['fast', 'photorealistic']

for mode in modes:
    renderer = SyntheticRenderer(mode=mode)
    model = renderer.load_model(model_path)

    start = time.time()
    image = renderer.render(model)
    elapsed = time.time() - start

    renderer.save_image(image, f'comparison/{mode}.png')
    print(f"{mode}: {elapsed:.2f}s")
```

## Configuration-Based Pipeline

Use configuration files for reproducible pipelines:

```python
import yaml
from src.coco_processing import COCOProcessor
from src.3d_generation import ModelGenerator
from src.synthetic_rendering import SyntheticRenderer

# Load configuration
with open('configs/pipeline.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize with config
processor = COCOProcessor(**config['coco_processing'])
generator = ModelGenerator(**config['3d_generation'])
renderer = SyntheticRenderer(**config['synthetic_rendering'])

# Run pipeline
masks = processor.extract_masks()
models = generator.batch_generate(list(masks.values()))
images = renderer.batch_render(models)
```

## Error Handling

Robust pipeline with error handling:

```python
from src.coco_processing import COCOProcessor
from src.3d_generation import ModelGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

processor = COCOProcessor('data/coco/annotations.json')
generator = ModelGenerator()

masks = processor.extract_masks()
successful = 0
failed = 0

for i, (instance_id, mask) in enumerate(masks.items()):
    try:
        model = generator.generate_from_mask(mask)
        output_path = f'data/3d_models/model_{instance_id}.obj'
        generator.save_model(model, output_path)
        successful += 1
        logger.info(f"Processed {i+1}/{len(masks)}: {instance_id}")
    except Exception as e:
        failed += 1
        logger.error(f"Failed on {instance_id}: {str(e)}")

logger.info(f"Complete: {successful} successful, {failed} failed")
```
