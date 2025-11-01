# 3D Generation

The 3D Generation module converts 2D instance segmentation masks into 3D mesh models.

## Overview

This module implements various techniques to reconstruct 3D models from 2D segmentation masks, including depth estimation, mesh generation, and texture mapping.

## Techniques

### 1. Depth Estimation

Estimate depth maps from 2D masks using various methods:

- **MiDaS**: Monocular depth estimation
- **Shape-from-Silhouette**: Classical computer vision approach
- **Learning-based**: Neural network depth prediction

### 2. Mesh Generation

Convert depth maps to 3D meshes:

- **Marching Cubes**: Isosurface extraction
- **Poisson Surface Reconstruction**: Smooth mesh generation
- **Delaunay Triangulation**: Point cloud to mesh

### 3. Texture Mapping

Apply textures to 3D models:

- **UV Mapping**: Project 2D image onto 3D surface
- **Color Transfer**: Extract colors from source images
- **Procedural Textures**: Generate synthetic textures

## Usage

### Basic 3D Generation

```python
from src.3d_generation import ModelGenerator

generator = ModelGenerator(method='midas')

# Generate 3D model from mask
model_3d = generator.generate_from_mask(mask)

# Save to file
generator.save_model(model_3d, 'data/3d_models/output.obj')
```

### Advanced Options

```python
# Configure generation parameters
generator = ModelGenerator(
    method='midas',
    resolution=512,
    smoothing=True,
    texture_mapping=True
)

# Generate with custom settings
model_3d = generator.generate_from_mask(
    mask,
    depth_scale=1.5,
    mesh_simplification=0.8
)
```

### Batch Processing

```python
# Process multiple masks
masks = [mask1, mask2, mask3]
models = generator.batch_generate(masks, num_workers=4)

for i, model in enumerate(models):
    generator.save_model(model, f'data/3d_models/model_{i}.obj')
```

## Depth Estimation Methods

### MiDaS

```python
generator = ModelGenerator(method='midas')
depth_map = generator.estimate_depth(mask)
```

**Pros**: High-quality depth estimation
**Cons**: Requires GPU, slower

### Shape-from-Silhouette

```python
generator = ModelGenerator(method='silhouette')
depth_map = generator.estimate_depth(mask)
```

**Pros**: Fast, CPU-friendly
**Cons**: Less accurate depth

## Mesh Formats

Supported output formats:

- **OBJ**: Wavefront OBJ (with MTL for materials)
- **PLY**: Polygon File Format
- **STL**: Stereolithography (3D printing)
- **GLTF**: GL Transmission Format (web-ready)

```python
# Export in different formats
generator.save_model(model, 'output.obj', format='obj')
generator.save_model(model, 'output.ply', format='ply')
generator.save_model(model, 'output.stl', format='stl')
```

## Optimization

### Mesh Simplification

Reduce polygon count while preserving shape:

```python
simplified_model = generator.simplify_mesh(
    model_3d,
    target_faces=1000,
    preserve_boundary=True
)
```

### Smoothing

Apply smoothing algorithms:

```python
smooth_model = generator.smooth_mesh(
    model_3d,
    iterations=10,
    method='laplacian'
)
```

## Configuration

Configure 3D generation via `configs/3d_generation.yaml`:

```yaml
3d_generation:
  # Depth estimation
  depth_estimator: "midas"
  depth_model: "DPT_Large"

  # Mesh generation
  mesh_resolution: 256
  marching_cubes_threshold: 0.5

  # Optimization
  simplification_ratio: 0.8
  smoothing_iterations: 5

  # Texture
  texture_resolution: 1024
  uv_unwrap_method: "smart_project"
```

## Performance Tips

- Use GPU for depth estimation when available
- Batch process multiple masks together
- Cache depth maps to avoid recomputation
- Adjust resolution based on your needs (lower = faster)

## API Reference

For detailed API documentation, see the [3D Generation API Reference](../api/3d-generation.md).

## Next Steps

After generating 3D models, move on to [Synthetic Rendering](synthetic-rendering.md) to create synthetic images.
