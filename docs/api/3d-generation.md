# 3D Generation API

API reference for the 3D Generation module.

## ModelGenerator

Main class for generating 3D models from 2D masks.

### Constructor

```python
ModelGenerator(
    method: str = 'midas',
    resolution: int = 256,
    device: str = 'cuda'
)
```

**Parameters:**

- `method` (str): Depth estimation method ('midas', 'silhouette')
- `resolution` (int): Mesh resolution
- `device` (str): Compute device ('cuda' or 'cpu')

**Example:**

```python
generator = ModelGenerator(method='midas', resolution=512)
```

### Methods

#### generate_from_mask()

Generate 3D model from 2D segmentation mask.

```python
def generate_from_mask(
    mask: np.ndarray,
    depth_scale: float = 1.0,
    mesh_simplification: float = 1.0
) -> Mesh3D
```

**Parameters:**

- `mask` (np.ndarray): Binary segmentation mask
- `depth_scale` (float): Scale factor for depth values
- `mesh_simplification` (float): Simplification ratio (0-1)

**Returns:** 3D mesh object

---

#### save_model()

Save 3D model to file.

```python
def save_model(
    model: Mesh3D,
    output_path: str,
    format: str = 'obj'
) -> None
```

**Parameters:**

- `model` (Mesh3D): 3D mesh to save
- `output_path` (str): Output file path
- `format` (str): File format ('obj', 'ply', 'stl', 'gltf')

---

#### batch_generate()

Generate multiple 3D models in batch.

```python
def batch_generate(
    masks: list,
    num_workers: int = 4
) -> list
```

**Parameters:**

- `masks` (list): List of binary masks
- `num_workers` (int): Number of parallel workers

**Returns:** List of 3D mesh objects

---

#### estimate_depth()

Estimate depth map from mask.

```python
def estimate_depth(mask: np.ndarray) -> np.ndarray
```

**Parameters:**

- `mask` (np.ndarray): Binary segmentation mask

**Returns:** Depth map as numpy array

---

#### simplify_mesh()

Reduce mesh polygon count.

```python
def simplify_mesh(
    model: Mesh3D,
    target_faces: int,
    preserve_boundary: bool = True
) -> Mesh3D
```

**Parameters:**

- `model` (Mesh3D): Input mesh
- `target_faces` (int): Target number of faces
- `preserve_boundary` (bool): Preserve mesh boundaries

**Returns:** Simplified mesh

---

#### smooth_mesh()

Apply smoothing to mesh.

```python
def smooth_mesh(
    model: Mesh3D,
    iterations: int = 10,
    method: str = 'laplacian'
) -> Mesh3D
```

**Parameters:**

- `model` (Mesh3D): Input mesh
- `iterations` (int): Number of smoothing iterations
- `method` (str): Smoothing method ('laplacian', 'taubin')

**Returns:** Smoothed mesh
