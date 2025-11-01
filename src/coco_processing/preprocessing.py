"""
COCO Image Preprocessing for 3D Generation

This module provides preprocessing utilities for COCO dataset images,
specifically designed to prepare them for 3D reconstruction with OpenLRM.
Uses segmentation masks to properly isolate objects from backgrounds.

Example:
    >>> from src.coco_processing import COCOPreprocessor
    >>> preprocessor = COCOPreprocessor('annotations.json', 'images/')
    >>> result = preprocessor.extract_and_prepare(image_id=12345, output_path='output.png')
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg not available. Background removal will be disabled.")


class COCOPreprocessor:
    """
    Preprocesses COCO images for 3D reconstruction.

    This class handles extracting objects from COCO images using segmentation masks,
    removing backgrounds, centering objects, and preparing RGBA images suitable for
    OpenLRM 3D generation.

    Args:
        annotation_file (str): Path to COCO annotation JSON file
        image_dir (str): Directory containing COCO images
        enable_logging (bool): Enable detailed logging
    """

    def __init__(
        self,
        annotation_file: Union[str, Path],
        image_dir: Union[str, Path],
        enable_logging: bool = True
    ):
        """Initialize COCO preprocessor."""
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        self.logger = logging.getLogger(__name__)
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)

        # Load COCO annotations
        self.logger.info(f"Loading COCO annotations from {self.annotation_file}")
        self.coco = COCO(str(self.annotation_file))
        self.logger.info(
            f"Loaded {len(self.coco.getImgIds())} images "
            f"and {len(self.coco.getAnnIds())} annotations"
        )

    def get_segmentation_mask(
        self,
        annotation: Dict,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Convert COCO segmentation to binary mask.

        Args:
            annotation: COCO annotation dictionary
            image_shape: (height, width) of the image

        Returns:
            Binary mask as numpy array (height, width)
        """
        height, width = image_shape

        if 'segmentation' in annotation:
            seg = annotation['segmentation']

            if isinstance(seg, list):
                # Polygon format
                mask = np.zeros((height, width), dtype=np.uint8)
                img = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(img)

                for polygon in seg:
                    if len(polygon) >= 6:  # Valid polygon needs at least 3 points
                        # Convert flat list to list of tuples
                        points = [(polygon[i], polygon[i+1])
                                 for i in range(0, len(polygon), 2)]
                        draw.polygon(points, outline=1, fill=1)

                mask = np.array(img)

            elif isinstance(seg, dict):
                # RLE format
                if isinstance(seg['counts'], list):
                    # Uncompressed RLE
                    rle = coco_mask.frPyObjects([seg], height, width)[0]
                else:
                    # Compressed RLE
                    rle = seg
                mask = coco_mask.decode(rle)

            else:
                raise ValueError(f"Unknown segmentation format: {type(seg)}")

        else:
            # Fallback to bounding box if no segmentation
            self.logger.warning(
                f"No segmentation found for annotation {annotation.get('id')}, "
                "using bounding box"
            )
            bbox = annotation['bbox']
            x, y, w, h = [int(v) for v in bbox]
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 1

        return mask.astype(np.uint8)

    def extract_object_with_mask(
        self,
        image: Image.Image,
        mask: np.ndarray,
        apply_background_removal: bool = True
    ) -> Image.Image:
        """
        Extract object from image using segmentation mask.

        Args:
            image: Input PIL Image (RGB)
            mask: Binary segmentation mask
            apply_background_removal: Use rembg for additional bg removal

        Returns:
            RGBA image with transparent background
        """
        # Convert image to RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Create RGBA array
        img_array = np.array(image)

        # Apply mask to alpha channel
        mask_binary = (mask > 0).astype(np.uint8) * 255
        img_array[:, :, 3] = mask_binary

        # Convert back to PIL
        masked_image = Image.fromarray(img_array, 'RGBA')

        # Optional: Apply rembg for additional refinement
        if apply_background_removal and REMBG_AVAILABLE:
            try:
                # rembg works better with the masked image as input
                masked_image = remove(masked_image, alpha_matting=True)
            except Exception as e:
                self.logger.warning(f"rembg failed: {e}, using mask only")

        return masked_image

    def center_and_pad(
        self,
        image: Image.Image,
        target_size: int = 512,
        padding_percent: float = 0.1
    ) -> Image.Image:
        """
        Center object and pad to square image.

        Args:
            image: RGBA image with transparent background
            target_size: Target square size in pixels
            padding_percent: Padding around object (0.1 = 10% padding)

        Returns:
            Centered and padded RGBA image
        """
        # Get bounding box of non-transparent pixels
        bbox = image.getbbox()

        if bbox is None:
            # Image is completely transparent
            self.logger.warning("Image is completely transparent, returning blank")
            return Image.new('RGBA', (target_size, target_size), (255, 255, 255, 0))

        # Crop to object
        cropped = image.crop(bbox)
        width, height = cropped.size

        # Calculate size with padding
        max_dim = max(width, height)
        padded_dim = int(max_dim * (1 + 2 * padding_percent))

        # Create new square canvas
        square = Image.new('RGBA', (padded_dim, padded_dim), (255, 255, 255, 0))

        # Paste cropped image centered
        paste_x = (padded_dim - width) // 2
        paste_y = (padded_dim - height) // 2
        square.paste(cropped, (paste_x, paste_y))

        # Resize to target size
        final = square.resize((target_size, target_size), Image.Resampling.LANCZOS)

        return final

    def extract_and_prepare(
        self,
        image_id: Optional[int] = None,
        annotation_id: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        target_size: int = 512,
        padding_percent: float = 0.1,
        apply_background_removal: bool = False,
        return_metadata: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Dict]]:
        """
        Extract and prepare a COCO object for 3D generation.

        Args:
            image_id: COCO image ID (provide either image_id or annotation_id)
            annotation_id: COCO annotation ID
            output_path: Optional path to save prepared image
            target_size: Target square size in pixels
            padding_percent: Padding around object
            apply_background_removal: Use rembg for refinement
            return_metadata: Return metadata dict with image

        Returns:
            Prepared RGBA image, or (image, metadata) if return_metadata=True

        Raises:
            ValueError: If neither image_id nor annotation_id provided
        """
        # Get annotation
        if annotation_id is not None:
            ann = self.coco.loadAnns(annotation_id)[0]
            image_id = ann['image_id']
        elif image_id is not None:
            # Get largest annotation for this image
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            if not ann_ids:
                raise ValueError(f"No annotations found for image_id {image_id}")
            anns = self.coco.loadAnns(ann_ids)
            # Get annotation with largest area
            ann = max(anns, key=lambda a: a.get('area', 0))
        else:
            raise ValueError("Must provide either image_id or annotation_id")

        # Load image
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = self.image_dir / img_info['file_name']

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')

        # Get segmentation mask
        mask = self.get_segmentation_mask(
            ann,
            (img_info['height'], img_info['width'])
        )

        # Extract object using mask
        masked_image = self.extract_object_with_mask(
            image,
            mask,
            apply_background_removal=apply_background_removal
        )

        # Center and pad
        prepared_image = self.center_and_pad(
            masked_image,
            target_size=target_size,
            padding_percent=padding_percent
        )

        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            prepared_image.save(output_path, 'PNG')
            self.logger.info(f"Saved prepared image to {output_path}")

        # Prepare metadata
        if return_metadata:
            category = self.coco.loadCats(ann['category_id'])[0]
            metadata = {
                'image_id': image_id,
                'annotation_id': ann['id'],
                'category_id': ann['category_id'],
                'category_name': category['name'],
                'area': ann.get('area'),
                'bbox': ann.get('bbox'),
                'original_size': (img_info['width'], img_info['height']),
                'prepared_size': (target_size, target_size),
                'source_file': str(img_path)
            }
            return prepared_image, metadata

        return prepared_image

    def batch_extract_category(
        self,
        category_name: str,
        output_dir: Union[str, Path],
        max_images: Optional[int] = None,
        target_size: int = 512,
        apply_background_removal: bool = False
    ) -> List[Dict]:
        """
        Extract all instances of a specific category.

        Args:
            category_name: COCO category name (e.g., 'car', 'person')
            output_dir: Directory to save prepared images
            max_images: Maximum number of images to process
            target_size: Target image size
            apply_background_removal: Use rembg

        Returns:
            List of result dictionaries with metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get category ID
        cat_ids = self.coco.getCatIds(catNms=[category_name])
        if not cat_ids:
            raise ValueError(f"Category '{category_name}' not found")

        cat_id = cat_ids[0]

        # Get all annotations for this category
        ann_ids = self.coco.getAnnIds(catIds=cat_id)
        if max_images:
            ann_ids = ann_ids[:max_images]

        self.logger.info(
            f"Processing {len(ann_ids)} instances of category '{category_name}'"
        )

        results = []

        for i, ann_id in enumerate(ann_ids):
            try:
                output_path = output_dir / f"{category_name}_{ann_id}.png"

                prepared_img, metadata = self.extract_and_prepare(
                    annotation_id=ann_id,
                    output_path=output_path,
                    target_size=target_size,
                    apply_background_removal=apply_background_removal,
                    return_metadata=True
                )

                results.append({
                    'success': True,
                    'output_path': str(output_path),
                    **metadata
                })

                self.logger.info(
                    f"[{i+1}/{len(ann_ids)}] ✓ Processed annotation {ann_id}"
                )

            except Exception as e:
                self.logger.error(
                    f"[{i+1}/{len(ann_ids)}] ✗ Failed annotation {ann_id}: {e}"
                )
                results.append({
                    'success': False,
                    'annotation_id': ann_id,
                    'error': str(e)
                })

        successful = sum(1 for r in results if r['success'])
        self.logger.info(
            f"Batch complete: {successful}/{len(ann_ids)} successful"
        )

        return results


if __name__ == "__main__":
    # Example usage
    import argparse
    import json

    parser = argparse.ArgumentParser(description="COCO Image Preprocessing")
    parser.add_argument("annotations", type=str, help="COCO annotation file")
    parser.add_argument("images", type=str, help="COCO images directory")
    parser.add_argument("output", type=str, help="Output directory")
    parser.add_argument("--category", type=str, default="car",
                       help="Category to extract")
    parser.add_argument("--max-images", type=int, default=10,
                       help="Maximum images to process")
    parser.add_argument("--size", type=int, default=512,
                       help="Output image size")
    parser.add_argument("--remove-bg", action="store_true",
                       help="Apply rembg background removal")

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = COCOPreprocessor(args.annotations, args.images)

    # Process category
    results = preprocessor.batch_extract_category(
        category_name=args.category,
        output_dir=args.output,
        max_images=args.max_images,
        target_size=args.size,
        apply_background_removal=args.remove_bg
    )

    # Save results
    results_file = Path(args.output) / "preprocessing_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
