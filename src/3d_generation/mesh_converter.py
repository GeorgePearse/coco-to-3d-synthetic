"""
Mesh Format Conversion Utilities

This module provides utilities for converting between different 3D mesh formats
and applying post-processing operations like cleaning, simplification, and optimization.

Example:
    >>> from src.3d_generation import MeshConverter
    >>> converter = MeshConverter()
    >>> converter.convert('input.ply', 'output.obj')
    >>> converter.simplify_mesh('input.ply', 'simplified.obj', target_faces=10000)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import trimesh
import numpy as np


class MeshConverter:
    """
    Convert and process 3D mesh files.

    This class provides utilities for converting meshes between formats,
    cleaning mesh geometry, simplifying polygon counts, and exporting
    to various formats suitable for different applications.

    Supported Formats:
        - PLY: Native OpenLRM output format
        - OBJ: Wavefront OBJ (widely supported)
        - GLB/GLTF: GL Transmission Format (web, AR/VR)
        - STL: Stereolithography (3D printing)
        - DAE: COLLADA (animation, rigging)
        - OFF: Object File Format

    Args:
        enable_logging (bool): Enable detailed logging
    """

    SUPPORTED_FORMATS = ['ply', 'obj', 'glb', 'gltf', 'stl', 'dae', 'off']

    def __init__(self, enable_logging: bool = True):
        """Initialize mesh converter."""
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        self.logger = logging.getLogger(__name__)

    def load_mesh(self, mesh_path: Union[str, Path]) -> trimesh.Trimesh:
        """
        Load mesh from file.

        Args:
            mesh_path: Path to mesh file

        Returns:
            Loaded trimesh object

        Raises:
            FileNotFoundError: If mesh file doesn't exist
            ValueError: If mesh format not supported
        """
        mesh_path = Path(mesh_path)

        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        try:
            mesh = trimesh.load(str(mesh_path))
            self.logger.info(
                f"Loaded mesh: {mesh_path.name} "
                f"({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)"
            )
            return mesh

        except Exception as e:
            raise ValueError(f"Failed to load mesh: {e}")

    def clean_mesh(
        self,
        mesh: trimesh.Trimesh,
        merge_vertices: bool = True,
        remove_degenerate: bool = True,
        remove_duplicate_faces: bool = True,
        remove_infinite: bool = True,
        fix_normals: bool = True
    ) -> trimesh.Trimesh:
        """
        Clean and repair mesh geometry.

        Args:
            mesh: Input mesh
            merge_vertices: Merge duplicate vertices
            remove_degenerate: Remove degenerate faces (zero area)
            remove_duplicate_faces: Remove duplicate faces
            remove_infinite: Remove infinite or NaN values
            fix_normals: Recalculate and fix face normals

        Returns:
            Cleaned mesh
        """
        self.logger.info("Cleaning mesh...")

        if merge_vertices:
            mesh.merge_vertices()
            self.logger.debug("Merged duplicate vertices")

        if remove_degenerate:
            mesh.remove_degenerate_faces()
            self.logger.debug("Removed degenerate faces")

        if remove_duplicate_faces:
            mesh.remove_duplicate_faces()
            self.logger.debug("Removed duplicate faces")

        if remove_infinite:
            mesh.remove_infinite_values()
            self.logger.debug("Removed infinite values")

        if fix_normals:
            mesh.fix_normals()
            self.logger.debug("Fixed face normals")

        self.logger.info(
            f"Mesh cleaned: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        return mesh

    def simplify_mesh(
        self,
        mesh: trimesh.Trimesh,
        target_faces: Optional[int] = None,
        target_reduction: Optional[float] = None,
        preserve_border: bool = True
    ) -> trimesh.Trimesh:
        """
        Simplify mesh by reducing polygon count.

        Args:
            mesh: Input mesh
            target_faces: Target number of faces (provide either this or target_reduction)
            target_reduction: Reduction ratio (0.5 = reduce to 50% of faces)
            preserve_border: Preserve mesh boundaries

        Returns:
            Simplified mesh

        Raises:
            ValueError: If neither target_faces nor target_reduction provided
        """
        if target_faces is None and target_reduction is None:
            raise ValueError("Must provide either target_faces or target_reduction")

        original_faces = len(mesh.faces)

        if target_reduction is not None:
            target_faces = int(original_faces * target_reduction)

        self.logger.info(
            f"Simplifying mesh from {original_faces} to {target_faces} faces"
        )

        try:
            simplified = mesh.simplify_quadric_decimation(target_faces)

            self.logger.info(
                f"Simplified: {len(simplified.vertices)} vertices, "
                f"{len(simplified.faces)} faces "
                f"({len(simplified.faces)/original_faces*100:.1f}% of original)"
            )

            return simplified

        except Exception as e:
            self.logger.warning(f"Simplification failed: {e}, returning original mesh")
            return mesh

    def fill_holes(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Fill small holes in mesh.

        Args:
            mesh: Input mesh

        Returns:
            Mesh with holes filled
        """
        self.logger.info("Filling mesh holes...")

        try:
            mesh.fill_holes()
            self.logger.info("Holes filled successfully")
        except Exception as e:
            self.logger.warning(f"Hole filling failed: {e}")

        return mesh

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        clean: bool = True,
        simplify_faces: Optional[int] = None,
        fill_holes: bool = False
    ) -> Path:
        """
        Convert mesh to different format.

        Args:
            input_path: Input mesh file path
            output_path: Output mesh file path
            clean: Clean mesh before export
            simplify_faces: Optional target face count for simplification
            fill_holes: Fill small holes in mesh

        Returns:
            Path to output file

        Raises:
            ValueError: If output format not supported
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Check output format
        output_format = output_path.suffix[1:].lower()
        if output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported output format '{output_format}'. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        # Load mesh
        mesh = self.load_mesh(input_path)

        # Clean if requested
        if clean:
            mesh = self.clean_mesh(mesh)

        # Simplify if requested
        if simplify_faces:
            mesh = self.simplify_mesh(mesh, target_faces=simplify_faces)

        # Fill holes if requested
        if fill_holes:
            mesh = self.fill_holes(mesh)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export
        mesh.export(str(output_path))
        self.logger.info(f"Exported mesh to: {output_path}")

        return output_path

    def batch_convert(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        output_format: str,
        clean: bool = True,
        simplify_faces: Optional[int] = None
    ) -> List[Dict]:
        """
        Convert multiple meshes to a target format.

        Args:
            input_paths: List of input mesh paths
            output_dir: Output directory
            output_format: Target format ('obj', 'glb', etc.)
            clean: Clean meshes before export
            simplify_faces: Optional target face count

        Returns:
            List of result dictionaries
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, input_path in enumerate(input_paths):
            input_path = Path(input_path)

            try:
                output_path = output_dir / f"{input_path.stem}.{output_format}"

                self.convert(
                    input_path=input_path,
                    output_path=output_path,
                    clean=clean,
                    simplify_faces=simplify_faces
                )

                results.append({
                    'success': True,
                    'input': str(input_path),
                    'output': str(output_path),
                    'format': output_format
                })

                self.logger.info(f"[{i+1}/{len(input_paths)}] ✓ {input_path.name}")

            except Exception as e:
                self.logger.error(f"[{i+1}/{len(input_paths)}] ✗ {input_path.name}: {e}")
                results.append({
                    'success': False,
                    'input': str(input_path),
                    'error': str(e)
                })

        successful = sum(1 for r in results if r['success'])
        self.logger.info(f"Batch complete: {successful}/{len(input_paths)} successful")

        return results

    def export_multiple_formats(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        formats: List[str] = ['obj', 'glb', 'stl'],
        clean: bool = True,
        simplify_faces: Optional[int] = None
    ) -> Dict[str, Path]:
        """
        Export mesh to multiple formats.

        Args:
            input_path: Input mesh path
            output_dir: Output directory
            formats: List of output formats
            clean: Clean mesh before export
            simplify_faces: Optional target face count

        Returns:
            Dictionary mapping format to output path
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and process mesh once
        mesh = self.load_mesh(input_path)

        if clean:
            mesh = self.clean_mesh(mesh)

        if simplify_faces:
            mesh = self.simplify_mesh(mesh, target_faces=simplify_faces)

        # Export to all formats
        outputs = {}
        for fmt in formats:
            if fmt not in self.SUPPORTED_FORMATS:
                self.logger.warning(f"Skipping unsupported format: {fmt}")
                continue

            output_path = output_dir / f"{input_path.stem}.{fmt}"

            try:
                mesh.export(str(output_path))
                outputs[fmt] = output_path
                self.logger.info(f"Exported {fmt.upper()}: {output_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to export {fmt}: {e}")

        return outputs

    def get_mesh_info(self, mesh_path: Union[str, Path]) -> Dict:
        """
        Get information about a mesh file.

        Args:
            mesh_path: Path to mesh file

        Returns:
            Dictionary with mesh statistics
        """
        mesh = self.load_mesh(mesh_path)

        return {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'edges': len(mesh.edges),
            'is_watertight': mesh.is_watertight,
            'is_winding_consistent': mesh.is_winding_consistent,
            'bounds': mesh.bounds.tolist(),
            'center_mass': mesh.center_mass.tolist(),
            'volume': float(mesh.volume) if mesh.is_watertight else None,
            'area': float(mesh.area),
            'euler_number': mesh.euler_number
        }


if __name__ == "__main__":
    # Example usage
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Mesh Format Converter")
    parser.add_argument("input", type=str, help="Input mesh file")
    parser.add_argument("output", type=str, help="Output mesh file or directory")
    parser.add_argument("--formats", nargs='+', default=['obj', 'glb', 'stl'],
                       help="Output formats (for batch export)")
    parser.add_argument("--clean", action="store_true", default=True,
                       help="Clean mesh before export")
    parser.add_argument("--simplify", type=int,
                       help="Target face count for simplification")
    parser.add_argument("--info", action="store_true",
                       help="Print mesh information and exit")
    parser.add_argument("--batch", action="store_true",
                       help="Export to multiple formats")

    args = parser.parse_args()

    converter = MeshConverter()

    if args.info:
        # Print mesh info
        info = converter.get_mesh_info(args.input)
        print("\nMesh Information:")
        print(json.dumps(info, indent=2))

    elif args.batch:
        # Export to multiple formats
        outputs = converter.export_multiple_formats(
            input_path=args.input,
            output_dir=args.output,
            formats=args.formats,
            clean=args.clean,
            simplify_faces=args.simplify
        )

        print("\nExported formats:")
        for fmt, path in outputs.items():
            print(f"  {fmt.upper()}: {path}")

    else:
        # Single conversion
        output = converter.convert(
            input_path=args.input,
            output_path=args.output,
            clean=args.clean,
            simplify_faces=args.simplify
        )

        print(f"\n✓ Converted: {output}")
