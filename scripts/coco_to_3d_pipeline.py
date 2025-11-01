#!/usr/bin/env python3
"""
Complete COCO to 3D Pipeline

This script processes COCO dataset images through the complete pipeline:
1. COCO preprocessing (segmentation-based extraction)
2. OpenLRM 3D generation
3. Mesh format conversion

Usage:
    python scripts/coco_to_3d_pipeline.py --config configs/coco_pipeline.yaml
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
import argparse
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coco_processing.preprocessing import COCOPreprocessor
from src.3d_generation.openlrm_generator import OpenLRMGenerator
from src.3d_generation.mesh_converter import MeshConverter


class COCOto3DPipeline:
    """Complete pipeline from COCO to 3D models."""

    def __init__(self, config_path: str):
        """Initialize pipeline from config file."""
        self.config = OmegaConf.load(config_path)
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.logger.info("Initializing pipeline components...")

        # COCO preprocessor
        self.preprocessor = COCOPreprocessor(
            annotation_file=self.config.coco_processing.annotation_file,
            image_dir=self.config.coco_processing.image_dir
        )

        # OpenLRM generator
        self.generator = OpenLRMGenerator(
            model_size=self.config.openlrm.model_size,
            model_variant=self.config.openlrm.model_variant,
            device=self.config.openlrm.device
        )

        # Mesh converter
        self.converter = MeshConverter()

        self.logger.info("Pipeline initialized successfully")

    def process_category(
        self,
        category_name: str,
        max_images: int = None
    ) -> List[Dict]:
        """Process all instances of a COCO category."""
        output_dir = Path(self.config.output.base_dir) / category_name
        output_dir.mkdir(parents=True, exist_ok=True)

        max_images = max_images or self.config.batch.max_images

        self.logger.info(f"Processing category: {category_name}")

        # Step 1: Preprocess COCO images
        self.logger.info("Step 1: Preprocessing COCO images...")
        prep_dir = output_dir / "preprocessed"
        preprocessing_results = self.preprocessor.batch_extract_category(
            category_name=category_name,
            output_dir=prep_dir,
            max_images=max_images,
            target_size=self.config.coco_processing.target_size,
            apply_background_removal=self.config.coco_processing.apply_background_removal
        )

        # Filter successful preprocessings
        successful_preps = [r for r in preprocessing_results if r['success']]
        self.logger.info(f"Preprocessed {len(successful_preps)} images")

        if not successful_preps:
            self.logger.error("No images successfully preprocessed")
            return []

        # Step 2: Generate 3D models
        self.logger.info("Step 2: Generating 3D models with OpenLRM...")
        meshes_dir = output_dir / "meshes"
        image_paths = [r['output_path'] for r in successful_preps]

        generation_results = self.generator.generate_batch(
            image_paths=image_paths,
            output_dir=meshes_dir,
            camera_distance=self.config.openlrm.camera_distance,
            export_video=self.config.openlrm.export_video,
            export_mesh=self.config.openlrm.export_mesh,
            create_subdirs=self.config.output.create_subdirs
        )

        # Filter successful generations
        successful_gens = [r for r in generation_results if r['success']]
        self.logger.info(f"Generated {len(successful_gens)} 3D models")

        if not successful_gens or not self.config.mesh_processing.export_formats:
            final_results = self._combine_results(preprocessing_results, generation_results)
            self._save_results(final_results, output_dir / "results.json")
            return final_results

        # Step 3: Convert mesh formats
        self.logger.info("Step 3: Converting mesh formats...")
        conversion_results = []

        for gen_result in successful_gens:
            if not gen_result.get('mesh'):
                continue

            mesh_path = Path(gen_result['mesh'])
            converted_dir = mesh_path.parent / "converted"

            try:
                formats = self.config.mesh_processing.export_formats
                outputs = self.converter.export_multiple_formats(
                    input_path=mesh_path,
                    output_dir=converted_dir,
                    formats=formats,
                    clean=self.config.mesh_processing.clean_mesh,
                    simplify_faces=(
                        self.config.mesh_processing.target_faces
                        if self.config.mesh_processing.simplify
                        else None
                    )
                )

                conversion_results.append({
                    'success': True,
                    'source_mesh': str(mesh_path),
                    'converted_formats': {fmt: str(path) for fmt, path in outputs.items()}
                })

            except Exception as e:
                self.logger.error(f"Conversion failed for {mesh_path}: {e}")
                conversion_results.append({
                    'success': False,
                    'source_mesh': str(mesh_path),
                    'error': str(e)
                })

        # Combine all results
        final_results = self._combine_results(
            preprocessing_results,
            generation_results,
            conversion_results
        )

        # Save results
        if self.config.output.save_metadata:
            self._save_results(final_results, output_dir / "results.json")

        return final_results

    def _combine_results(self, *result_lists) -> List[Dict]:
        """Combine results from different pipeline stages."""
        combined = []

        # Use preprocessing results as base
        for prep_result in result_lists[0]:
            combined_result = {'preprocessing': prep_result}

            # Find matching generation result
            if len(result_lists) > 1:
                prep_path = prep_result.get('output_path')
                gen_result = next(
                    (r for r in result_lists[1] if r.get('image_path') == prep_path),
                    None
                )
                if gen_result:
                    combined_result['generation'] = gen_result

                    # Find matching conversion result
                    if len(result_lists) > 2:
                        mesh_path = gen_result.get('mesh')
                        conv_result = next(
                            (r for r in result_lists[2] if r.get('source_mesh') == mesh_path),
                            None
                        )
                        if conv_result:
                            combined_result['conversion'] = conv_result

            combined.append(combined_result)

        return combined

    def _save_results(self, results: List[Dict], output_path: Path):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="COCO to 3D Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/coco_pipeline.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="COCO category to process (overrides config)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum images to process (overrides config)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize pipeline
    try:
        pipeline = COCOto3DPipeline(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1

    # Get configuration
    config = pipeline.config

    # Determine categories to process
    if args.category:
        categories = [args.category]
    elif config.coco_processing.get('categories'):
        categories = config.coco_processing.categories
    else:
        logger.error("No categories specified. Use --category or configure in YAML")
        return 1

    # Process each category
    all_results = {}

    for category in categories:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing category: {category}")
        logger.info(f"{'='*60}\n")

        try:
            results = pipeline.process_category(
                category_name=category,
                max_images=args.max_images
            )
            all_results[category] = results

            # Print summary
            successful = sum(
                1 for r in results
                if r.get('generation', {}).get('success', False)
            )
            logger.info(f"\n{category} Summary: {successful}/{len(results)} successful\n")

        except Exception as e:
            logger.error(f"Failed to process category {category}: {e}")
            continue

    logger.info("\n" + "="*60)
    logger.info("Pipeline complete!")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
