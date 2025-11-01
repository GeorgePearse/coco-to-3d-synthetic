# Synthetic Rendering API

API reference for the Synthetic Rendering module.

## SyntheticRenderer

Main class for rendering synthetic images from 3D models.

### Constructor

```python
SyntheticRenderer(
    mode: str = 'photorealistic',
    resolution: tuple = (1920, 1080),
    device: str = 'cuda'
)
```

**Parameters:**

- `mode` (str): Rendering mode ('photorealistic', 'fast', 'depth')
- `resolution` (tuple): Output image resolution (width, height)
- `device` (str): Compute device ('cuda' or 'cpu')

**Example:**

```python
renderer = SyntheticRenderer(mode='fast', resolution=(1024, 768))
```

### Methods

#### load_model()

Load 3D model from file.

```python
def load_model(model_path: str) -> Model3D
```

**Parameters:**

- `model_path` (str): Path to 3D model file

**Returns:** Loaded 3D model object

---

#### render()

Render 3D model to image.

```python
def render(
    model: Model3D,
    camera_angle: float = 0,
    camera_distance: float = 5.0,
    camera_height: float = 2.0,
    samples: int = 64
) -> np.ndarray
```

**Parameters:**

- `model` (Model3D): 3D model to render
- `camera_angle` (float): Camera rotation angle (degrees)
- `camera_distance` (float): Camera distance from object
- `camera_height` (float): Camera height
- `samples` (int): Number of rendering samples

**Returns:** Rendered image as numpy array

---

#### save_image()

Save rendered image to file.

```python
def save_image(
    image: np.ndarray,
    output_path: str,
    format: str = 'png'
) -> None
```

**Parameters:**

- `image` (np.ndarray): Image to save
- `output_path` (str): Output file path
- `format` (str): Image format ('png', 'jpg', 'exr', 'tiff')

---

#### set_camera()

Set camera position and orientation.

```python
def set_camera(
    position: tuple,
    look_at: tuple = (0, 0, 0),
    up_vector: tuple = (0, 1, 0),
    fov: float = 50
) -> None
```

**Parameters:**

- `position` (tuple): Camera position (x, y, z)
- `look_at` (tuple): Point camera is looking at
- `up_vector` (tuple): Camera up direction
- `fov` (float): Field of view in degrees

---

#### set_lighting()

Configure scene lighting.

```python
def set_lighting(
    type: str = 'studio',
    intensity: float = 1.0,
    color: tuple = (255, 255, 255)
) -> None
```

**Parameters:**

- `type` (str): Lighting preset ('studio', 'outdoor', 'dramatic', 'soft')
- `intensity` (float): Light intensity
- `color` (tuple): Light color (R, G, B)

---

#### set_background()

Set rendering background.

```python
def set_background(
    color: tuple = None,
    image: str = None,
    mode: str = 'solid'
) -> None
```

**Parameters:**

- `color` (tuple): Background color (R, G, B)
- `image` (str): Path to background image
- `mode` (str): Background mode ('solid', 'image', 'random')

---

#### batch_render()

Render multiple models in batch.

```python
def batch_render(
    model_paths: list,
    num_viewpoints: int = 8,
    num_workers: int = 4
) -> list
```

**Parameters:**

- `model_paths` (list): List of model file paths
- `num_viewpoints` (int): Number of viewpoints per model
- `num_workers` (int): Number of parallel workers

**Returns:** List of rendered images

---

#### render_depth()

Render depth map.

```python
def render_depth(model: Model3D) -> np.ndarray
```

**Parameters:**

- `model` (Model3D): 3D model to render

**Returns:** Depth map as numpy array

---

#### render_normals()

Render surface normals map.

```python
def render_normals(model: Model3D) -> np.ndarray
```

**Parameters:**

- `model` (Model3D): 3D model to render

**Returns:** Normal map as numpy array
