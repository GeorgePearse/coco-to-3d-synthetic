# Synthetic Rendering

The Synthetic Rendering module creates realistic synthetic images from 3D models using various rendering techniques.

## Overview

This module provides tools to render 3D models from multiple viewpoints, with different lighting conditions and backgrounds, to generate diverse synthetic training data.

## Features

- Multi-viewpoint rendering
- Configurable lighting setups
- Background composition
- Camera control
- Post-processing effects

## Usage

### Basic Rendering

```python
from src.synthetic_rendering import SyntheticRenderer

renderer = SyntheticRenderer()

# Load 3D model
model = renderer.load_model('data/3d_models/model.obj')

# Render from default viewpoint
image = renderer.render(model)

# Save result
renderer.save_image(image, 'data/synthetic_images/render.png')
```

### Multi-Viewpoint Rendering

```python
# Render from multiple angles
for angle in range(0, 360, 45):
    image = renderer.render(
        model,
        camera_angle=angle,
        camera_distance=5.0,
        camera_height=2.0
    )
    renderer.save_image(image, f'output_{angle}.png')
```

### Lighting Configuration

```python
# Configure lighting
renderer.set_lighting(
    type='studio',  # or 'outdoor', 'dramatic', 'soft'
    intensity=1.2,
    color=(255, 255, 240)
)

# Multiple light sources
renderer.add_light(position=(5, 5, 5), intensity=1.0, type='point')
renderer.add_light(position=(-5, 5, 5), intensity=0.5, type='point')
```

### Background Composition

```python
# Solid color background
renderer.set_background(color=(255, 255, 255))

# Image background
renderer.set_background(image='backgrounds/outdoor.jpg')

# Random backgrounds
renderer.set_background(mode='random', dataset='backgrounds/')
```

## Rendering Modes

### Photorealistic Rendering

```python
renderer = SyntheticRenderer(mode='photorealistic')
image = renderer.render(model, samples=128, denoising=True)
```

**Use case**: High-quality visualization, evaluation datasets

### Fast Rendering

```python
renderer = SyntheticRenderer(mode='fast')
image = renderer.render(model, samples=16)
```

**Use case**: Quick previews, large-scale data generation

### Depth Rendering

```python
depth_map = renderer.render_depth(model)
```

**Use case**: Depth estimation training data

### Normal Mapping

```python
normal_map = renderer.render_normals(model)
```

**Use case**: Surface normal estimation

## Camera Control

### Manual Camera Positioning

```python
renderer.set_camera(
    position=(x, y, z),
    look_at=(0, 0, 0),
    up_vector=(0, 1, 0),
    fov=50
)
```

### Automatic Camera Placement

```python
# Orbit around object
positions = renderer.generate_orbit_positions(
    num_views=16,
    radius=5.0,
    height_variation=True
)

for i, pos in enumerate(positions):
    renderer.set_camera(position=pos)
    image = renderer.render(model)
    renderer.save_image(image, f'orbit_{i}.png')
```

## Batch Rendering

Process multiple models efficiently:

```python
models = [
    'data/3d_models/model1.obj',
    'data/3d_models/model2.obj',
    'data/3d_models/model3.obj'
]

# Batch render with parallel processing
images = renderer.batch_render(
    models,
    num_viewpoints=8,
    num_workers=4
)
```

## Post-Processing

Apply effects after rendering:

```python
# Add noise and blur for realism
image = renderer.render(model)
image = renderer.add_noise(image, intensity=0.02)
image = renderer.add_motion_blur(image, strength=0.5)
image = renderer.adjust_exposure(image, factor=1.2)
```

## Configuration

Configure rendering via `configs/synthetic_rendering.yaml`:

```yaml
synthetic_rendering:
  # Rendering settings
  resolution: [1920, 1080]
  samples: 64
  max_bounces: 8

  # Camera settings
  fov: 50
  camera_distance: 5.0
  num_viewpoints: 16

  # Lighting
  lighting_type: "studio"
  ambient_intensity: 0.3

  # Background
  background_mode: "random"
  background_dataset: "backgrounds/"

  # Post-processing
  denoise: true
  bloom: false
  motion_blur: false
```

## Output Formats

Supported image formats:

- PNG (lossless)
- JPEG (compressed)
- EXR (HDR, 32-bit)
- TIFF (high-quality)

```python
renderer.save_image(image, 'output.png', format='png')
renderer.save_image(image, 'output.exr', format='exr', bit_depth=32)
```

## Performance Optimization

### GPU Acceleration

```python
# Use GPU for rendering
renderer = SyntheticRenderer(device='cuda')
```

### Tiling for Large Images

```python
# Render large images in tiles
image = renderer.render_tiled(
    model,
    resolution=(4096, 4096),
    tile_size=512
)
```

### Caching

```python
# Cache loaded models
renderer.enable_model_cache(max_size=10)
```

## API Reference

For detailed API documentation, see the [Synthetic Rendering API Reference](../api/synthetic-rendering.md).

## Examples

Check out the [Examples](../examples.md) page for complete rendering pipelines and use cases.
