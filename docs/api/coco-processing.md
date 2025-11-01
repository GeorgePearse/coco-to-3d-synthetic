# COCO Processing API

API reference for the COCO Processing module.

## COCOProcessor

Main class for processing COCO annotation files.

### Constructor

```python
COCOProcessor(annotation_file: str, image_dir: str = None)
```

**Parameters:**

- `annotation_file` (str): Path to COCO annotation JSON file
- `image_dir` (str, optional): Directory containing images

**Example:**

```python
processor = COCOProcessor('annotations.json', 'images/')
```

### Methods

#### load_annotations()

Load and parse COCO annotations.

```python
def load_annotations() -> dict
```

**Returns:** Dictionary containing parsed annotations

---

#### extract_masks()

Extract instance segmentation masks.

```python
def extract_masks(
    category: str = None,
    image_id: int = None
) -> dict
```

**Parameters:**

- `category` (str, optional): Filter by category name
- `image_id` (int, optional): Filter by image ID

**Returns:** Dictionary mapping instance IDs to binary masks

---

#### get_categories()

Get list of all categories in the dataset.

```python
def get_categories() -> list
```

**Returns:** List of category dictionaries

---

#### filter_by_area()

Filter annotations by area.

```python
def filter_by_area(
    min_area: float = 0,
    max_area: float = float('inf')
) -> list
```

**Parameters:**

- `min_area` (float): Minimum area threshold
- `max_area` (float): Maximum area threshold

**Returns:** Filtered list of annotations

---

#### visualize_annotations()

Visualize annotations on image.

```python
def visualize_annotations(
    image_id: int,
    save_path: str = None,
    show: bool = True
) -> np.ndarray
```

**Parameters:**

- `image_id` (int): Image ID to visualize
- `save_path` (str, optional): Path to save visualization
- `show` (bool): Whether to display the image

**Returns:** Annotated image as numpy array
