# COCO Processing

The COCO Processing module handles loading, parsing, and extracting data from COCO-format annotation files.

## Overview

COCO (Common Objects in Context) is a widely-used dataset format for object detection, segmentation, and captioning tasks. This module provides tools to work with COCO-format data.

## Features

- Load COCO annotation JSON files
- Extract instance segmentation masks
- Filter by category or image
- Convert annotations to standard formats
- Validate dataset integrity

## Usage

### Basic Loading

```python
from src.coco_processing import COCOProcessor

processor = COCOProcessor('data/coco/annotations.json')
annotations = processor.load_annotations()
```

### Extracting Masks

```python
# Extract all instance masks
masks = processor.extract_masks()

# Extract masks for specific category
person_masks = processor.extract_masks(category='person')

# Extract masks for specific image
image_masks = processor.extract_masks(image_id=12345)
```

### Category Information

```python
# Get all categories
categories = processor.get_categories()

# Get instances by category
cars = processor.get_instances_by_category('car')
```

## COCO Format Structure

A typical COCO annotation file has the following structure:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1234.5,
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "person"
    }
  ]
}
```

## Advanced Features

### Filtering

```python
# Filter by minimum area
large_objects = processor.filter_by_area(min_area=1000)

# Filter by bounding box size
filtered = processor.filter_by_bbox(min_width=50, min_height=50)
```

### Visualization

```python
# Visualize annotations on image
processor.visualize_annotations(image_id=12345, save_path='output.png')

# Show mask overlay
processor.show_mask_overlay(annotation_id=67890)
```

## Configuration

Configure COCO processing via `configs/coco_processing.yaml`:

```yaml
coco_processing:
  # Filter settings
  min_area: 100
  max_area: 50000

  # Category filtering
  categories: ["person", "car", "bicycle"]

  # Image filtering
  min_image_width: 640
  min_image_height: 480

  # Preprocessing
  normalize_masks: true
  resize_to: [256, 256]
```

## API Reference

For detailed API documentation, see the [COCO Processing API Reference](../api/coco-processing.md).

## Next Steps

Once you have extracted masks from COCO annotations, proceed to [3D Generation](3d-generation.md) to create 3D models.
