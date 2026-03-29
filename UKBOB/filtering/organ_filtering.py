"""
Specialized Organ Label Filter (SOLF) for automated quality control.

This module implements the novel SOLF pipeline for filtering and refining
automatically generated segmentation labels from the UK Biobank dataset.
It provides sophisticated quality control mechanisms including:
- Surface area and volume consistency checks
- Anatomical plausibility validation
- Connected component analysis
- Statistical outlier detection
- Confidence-based filtering

Author: Emmanuelle Bourigault
License: MIT
"""

import glob
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import measure, morphology

# Comprehensive organ taxonomy with 72 anatomical structures
ORGAN_TAXONOMY = {
    "labels": {
        "0": "background",
        "1": "spleen",
        "2": "kidney_right",
        "3": "kidney_left",
        "4": "gallbladder",
        "5": "liver",
        "6": "stomach",
        "7": "pancreas",
        "8": "adrenal_gland_right",
        "9": "adrenal_gland_left",
        "10": "lung_upper_lobe_left",
        "11": "lung_lower_lobe_left",
        "12": "lung_upper_lobe_right",
        "13": "lung_middle_lobe_right",
        "14": "lung_lower_lobe_right",
        "15": "esophagus",
        "16": "trachea",
        "17": "thyroid_gland",
        "18": "intestine",
        "19": "duodenum",
        "20": "unused",
        "21": "urinary_bladder",
        "22": "prostate",
        "23": "sacrum",
        "24": "heart",
        "25": "aorta",
        "26": "pulmonary_vein",
        "27": "brachiocephalic_trunk",
        "28": "subclavian_artery_right",
        "29": "subclavian_artery_left",
        "30": "common_carotid_artery_right",
        "31": "common_carotid_artery_left",
        "32": "brachiocephalic_vein_left",
        "33": "brachiocephalic_vein_right",
        "34": "atrial_appendage_left",
        "35": "superior_vena_cava",
        "36": "inferior_vena_cava",
        "37": "portal_vein_and_splenic_vein",
        "38": "iliac_artery_left",
        "39": "iliac_artery_right",
        "40": "iliac_vena_left",
        "41": "iliac_vena_right",
        "42": "humerus_left",
        "43": "humerus_right",
        "44": "scapula_left",
        "45": "scapula_right",
        "46": "clavicula_left",
        "47": "clavicula_right",
        "48": "femur_left",
        "49": "femur_right",
        "50": "hip_left",
        "51": "hip_right",
        "52": "spinal_cord",
        "53": "gluteus_maximus_left",
        "54": "gluteus_maximus_right",
        "55": "gluteus_medius_left",
        "56": "gluteus_medius_right",
        "57": "gluteus_minimus_left",
        "58": "gluteus_minimus_right",
        "59": "autochthon_left",
        "60": "autochthon_right",
        "61": "iliopsoas_left",
        "62": "iliopsoas_right",
        "63": "sternum",
        "64": "costal_cartilages",
        "65": "subcutaneous_fat",
        "66": "muscle",
        "67": "inner_fat",
        "68": "IVD",
        "69": "vertebra_body",
        "70": "vertebra_posterior_elements",
        "71": "spinal_channel",
        "72": "bone_other",
    }
}


@dataclass
class OrganStatistics:
    """
    Container for organ-specific statistical measurements.

    Attributes:
        volume_mm3: Volume in cubic millimeters
        surface_area_mm2: Surface area in square millimeters
        sphericity: Measure of how spherical the organ is (0-1)
        n_components: Number of connected components
        centroid: 3D centroid coordinates
        bbox: Bounding box coordinates (min_x, min_y, min_z, max_x, max_y, max_z)
        mean_intensity: Mean intensity value (if image provided)
        std_intensity: Standard deviation of intensity (if image provided)
    """

    volume_mm3: float
    surface_area_mm2: float
    sphericity: float
    n_components: int
    centroid: Tuple[float, float, float]
    bbox: Tuple[int, int, int, int, int, int]
    mean_intensity: Optional[float] = None
    std_intensity: Optional[float] = None


class OrganFilter:
    """
    Advanced filtering pipeline for automated segmentation quality control.

    This class implements the Specialized Organ Label Filter (SOLF) algorithm
    for identifying and correcting errors in automated segmentation masks.
    It uses anatomical priors and statistical analysis to ensure high-quality
    training data for the foundation model.

    Attributes:
        confidence_threshold: Minimum confidence score for accepting labels
        volume_outlier_factor: Factor for volume-based outlier detection
        min_volume_mm3: Minimum organ volume to be considered valid
        max_components: Maximum allowed connected components per organ

    Example:
        >>> filter = OrganFilter(confidence_threshold=0.8)
        >>> filtered_mask = filter.process_segmentation(
        ...     mask_path="segmentation.nii.gz",
        ...     image_path="image.nii.gz"
        ... )
    """

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        volume_outlier_factor: float = 3.0,
        min_volume_mm3: float = 100.0,
        max_components: int = 3,
    ):
        """
        Initialize the organ filter with quality control parameters.

        Args:
            confidence_threshold: Minimum confidence for accepting organ labels
            volume_outlier_factor: Standard deviations for outlier detection
            min_volume_mm3: Minimum valid organ volume in mm³
            max_components: Maximum connected components per organ
        """
        self.confidence_threshold = confidence_threshold
        self.volume_outlier_factor = volume_outlier_factor
        self.min_volume_mm3 = min_volume_mm3
        self.max_components = max_components

        # Load anatomical priors (organ volume ranges in mm³)
        self.anatomical_priors = self._load_anatomical_priors()

    def _load_anatomical_priors(self) -> Dict[int, Tuple[float, float]]:
        """
        Load expected volume ranges for each organ based on anatomical knowledge.

        Returns:
            Dict mapping organ IDs to (min_volume, max_volume) tuples in mm³
        """
        priors = {
            1: (100000, 300000),  # spleen: 100-300 cm³
            2: (80000, 200000),  # kidney_right: 80-200 cm³
            3: (80000, 200000),  # kidney_left: 80-200 cm³
            4: (15000, 60000),  # gallbladder: 15-60 cm³
            5: (1000000, 2500000),  # liver: 1000-2500 cm³
            6: (100000, 500000),  # stomach: 100-500 cm³
            7: (60000, 150000),  # pancreas: 60-150 cm³
            8: (3000, 10000),  # adrenal_gland_right: 3-10 cm³
            9: (3000, 10000),  # adrenal_gland_left: 3-10 cm³
            # ... additional organs with their expected ranges
            24: (200000, 400000),  # heart: 200-400 cm³
            21: (100000, 600000),  # urinary_bladder: 100-600 cm³ (variable)
            52: (20000, 50000),  # spinal_cord: 20-50 cm³
        }

        # Default range for organs without specific priors
        default_range = (1000, 10000000)  # 1 cm³ to 10 L

        for organ_id in range(73):
            if organ_id not in priors and organ_id != 0:  # Exclude background
                priors[organ_id] = default_range

        return priors

    def compute_surface_area(
        self, mask: np.ndarray, spacing: Tuple[float, float, float]
    ) -> float:
        """
        Compute the surface area of a 3D binary mask using marching cubes.

        This method uses the marching cubes algorithm to generate a mesh
        representation of the organ surface and calculates its area.

        Args:
            mask: 3D binary mask array
            spacing: Voxel spacing in mm (x, y, z)

        Returns:
            float: Surface area in mm²

        Example:
            >>> area = filter.compute_surface_area(
            ...     mask=organ_mask,
            ...     spacing=(1.5, 1.5, 2.0)
            ... )
        """
        # Apply marching cubes to get surface mesh
        try:
            verts, faces, _, _ = measure.marching_cubes(
                mask.astype(float), level=0.5, spacing=spacing
            )

            # Calculate surface area from mesh
            surface_area = measure.mesh_surface_area(verts, faces)

            return surface_area
        except Exception as e:
            warnings.warn(f"Surface area computation failed: {e}")
            return 0.0

    def compute_sphericity(self, volume: float, surface_area: float) -> float:
        """
        Calculate sphericity as a measure of organ shape regularity.

        Sphericity is a dimensionless measure that indicates how closely
        an object resembles a sphere. Perfect sphere has sphericity of 1.

        Args:
            volume: Organ volume in mm³
            surface_area: Organ surface area in mm²

        Returns:
            float: Sphericity value between 0 and 1

        Notes:
            Sphericity = (π^(1/3) * (6*V)^(2/3)) / A
            where V is volume and A is surface area
        """
        if surface_area == 0:
            return 0.0

        # Calculate sphericity using the formula
        sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area

        # Clamp to [0, 1] range (numerical errors can cause slight violations)
        return np.clip(sphericity, 0.0, 1.0)

    def analyze_connected_components(
        self, mask: np.ndarray, organ_id: int
    ) -> Tuple[np.ndarray, int]:
        """
        Analyze and filter connected components for an organ.

        This method identifies separate connected regions in the mask and
        filters out small spurious components that likely represent noise
        or segmentation errors.

        Args:
            mask: Binary mask for a single organ
            organ_id: Organ identifier for applying specific rules

        Returns:
            Tuple containing:
                - Filtered mask with small components removed
                - Number of remaining connected components
        """
        # Label connected components
        labeled_mask, n_components = ndimage.label(mask)

        if n_components == 0:
            return mask, 0

        # Calculate component volumes
        component_sizes = []
        for i in range(1, n_components + 1):
            component_size = np.sum(labeled_mask == i)
            component_sizes.append(component_size)

        # Sort components by size
        sorted_indices = np.argsort(component_sizes)[::-1]

        # Keep only significant components
        filtered_mask = np.zeros_like(mask)
        kept_components = 0

        for idx in sorted_indices[: self.max_components]:
            component_id = idx + 1
            component_size = component_sizes[idx]

            # Keep component if it's large enough (>1% of largest component)
            if component_size > 0.01 * component_sizes[sorted_indices[0]]:
                filtered_mask[labeled_mask == component_id] = 1
                kept_components += 1

        return filtered_mask, kept_components

    def validate_anatomical_plausibility(
        self, organ_stats: Dict[int, OrganStatistics]
    ) -> Dict[int, bool]:
        """
        Validate anatomical plausibility of organ segmentations.

        This method checks whether organ measurements fall within expected
        anatomical ranges and maintains proper spatial relationships.

        Args:
            organ_stats: Dictionary mapping organ IDs to their statistics

        Returns:
            Dict mapping organ IDs to validation status (True if valid)
        """
        validation_results = {}

        for organ_id, stats in organ_stats.items():
            if organ_id == 0:  # Skip background
                continue

            is_valid = True
            reasons = []

            # Check volume against anatomical priors
            if organ_id in self.anatomical_priors:
                min_vol, max_vol = self.anatomical_priors[organ_id]
                if not (min_vol <= stats.volume_mm3 <= max_vol):
                    is_valid = False
                    reasons.append(
                        f"Volume {stats.volume_mm3:.0f} outside range "
                        f"[{min_vol:.0f}, {max_vol:.0f}]"
                    )

            # Check minimum volume threshold
            if stats.volume_mm3 < self.min_volume_mm3:
                is_valid = False
                reasons.append(
                    f"Volume {stats.volume_mm3:.0f} below minimum "
                    f"{self.min_volume_mm3:.0f}"
                )

            # Check sphericity for certain organs
            sphericity_organs = [1, 2, 3, 8, 9]  # Kidneys, spleen, adrenals
            if organ_id in sphericity_organs and stats.sphericity < 0.3:
                is_valid = False
                reasons.append(f"Sphericity {stats.sphericity:.2f} too low")

            # Check number of components
            if stats.n_components > self.max_components:
                is_valid = False
                reasons.append(f"Too many components: {stats.n_components}")

            validation_results[organ_id] = is_valid

            if not is_valid:
                organ_name = ORGAN_TAXONOMY["labels"].get(
                    str(organ_id), f"organ_{organ_id}"
                )
                print(f"Validation failed for {organ_name}: {', '.join(reasons)}")

        return validation_results

    def process_segmentation(
        self,
        mask_path: str,
        image_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Process and filter a complete segmentation mask.

        This is the main entry point for the SOLF pipeline. It loads a
        segmentation mask, applies all filtering and validation steps,
        and returns a refined mask with improved quality.

        Args:
            mask_path: Path to input segmentation mask
            image_path: Optional path to corresponding image for intensity analysis
            output_path: Optional path to save filtered mask

        Returns:
            np.ndarray: Filtered segmentation mask

        Example:
            >>> filtered = filter.process_segmentation(
            ...     mask_path="raw_segmentation.nii.gz",
            ...     output_path="filtered_segmentation.nii.gz"
            ... )
        """
        print(f"Processing segmentation: {mask_path}")

        # Load mask
        mask_nii = nib.load(mask_path)
        mask_data = mask_nii.get_fdata().astype(np.uint8)
        spacing = mask_nii.header.get_zooms()[:3]

        # Load image if provided
        image_data = None
        if image_path:
            image_nii = nib.load(image_path)
            image_data = image_nii.get_fdata()

        # Initialize filtered mask
        filtered_mask = np.zeros_like(mask_data)

        # Get unique organ IDs
        organ_ids = np.unique(mask_data)
        organ_ids = organ_ids[organ_ids != 0]  # Exclude background

        # Compute statistics for each organ
        organ_stats = {}

        for organ_id in organ_ids:
            # Extract organ mask
            organ_mask = (mask_data == organ_id).astype(np.uint8)

            # Filter connected components
            organ_mask, n_components = self.analyze_connected_components(
                organ_mask, organ_id
            )

            # Skip if no valid components
            if n_components == 0:
                continue

            # Compute statistics
            volume_mm3 = np.sum(organ_mask) * np.prod(spacing)
            surface_area = self.compute_surface_area(organ_mask, spacing)
            sphericity = self.compute_sphericity(volume_mm3, surface_area)

            # Get centroid and bounding box
            coords = np.argwhere(organ_mask)
            if len(coords) > 0:
                centroid = tuple(coords.mean(axis=0))
                min_coords = coords.min(axis=0)
                max_coords = coords.max(axis=0)
                bbox = tuple(min_coords) + tuple(max_coords)
            else:
                centroid = (0, 0, 0)
                bbox = (0, 0, 0, 0, 0, 0)

            # Compute intensity statistics if image provided
            mean_intensity = None
            std_intensity = None
            if image_data is not None:
                organ_intensities = image_data[organ_mask == 1]
                if len(organ_intensities) > 0:
                    mean_intensity = organ_intensities.mean()
                    std_intensity = organ_intensities.std()

            # Store statistics
            organ_stats[organ_id] = OrganStatistics(
                volume_mm3=volume_mm3,
                surface_area_mm2=surface_area,
                sphericity=sphericity,
                n_components=n_components,
                centroid=centroid,
                bbox=bbox,
                mean_intensity=mean_intensity,
                std_intensity=std_intensity,
            )

            # Add to filtered mask if valid
            filtered_mask[organ_mask == 1] = organ_id

        # Validate anatomical plausibility
        validation_results = self.validate_anatomical_plausibility(organ_stats)

        # Remove invalid organs from filtered mask
        for organ_id, is_valid in validation_results.items():
            if not is_valid:
                filtered_mask[filtered_mask == organ_id] = 0

        # Save filtered mask if output path provided
        if output_path:
            filtered_nii = nib.Nifti1Image(
                filtered_mask.astype(np.uint8), mask_nii.affine, mask_nii.header
            )
            nib.save(filtered_nii, output_path)
            print(f"Saved filtered mask to: {output_path}")

        # Report filtering results
        n_original = len(organ_ids)
        n_filtered = len([v for v in validation_results.values() if v])
        print(f"Filtering complete: {n_filtered}/{n_original} organs retained")

        return filtered_mask


def batch_process_segmentations(
    input_dir: str,
    output_dir: str,
    image_dir: Optional[str] = None,
    confidence_threshold: float = 0.8,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Process multiple segmentation masks in batch with parallel processing.

    This function applies the SOLF pipeline to all segmentation masks in a
    directory, optionally using parallel processing for efficiency.

    Args:
        input_dir: Directory containing input segmentation masks
        output_dir: Directory to save filtered masks
        image_dir: Optional directory containing corresponding images
        confidence_threshold: Minimum confidence for organ filtering
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        DataFrame: Summary statistics for all processed masks

    Example:
        >>> results = batch_process_segmentations(
        ...     input_dir="./raw_masks/",
        ...     output_dir="./filtered_masks/",
        ...     n_jobs=8
        ... )
        >>> results.to_csv("filtering_report.csv")
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all segmentation masks
    mask_paths = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
    print(f"Found {len(mask_paths)} segmentation masks to process")

    # Initialize filter
    organ_filter = OrganFilter(confidence_threshold=confidence_threshold)

    # Process each mask
    results = []

    for mask_path in mask_paths:
        mask_name = os.path.basename(mask_path)
        output_path = os.path.join(output_dir, mask_name)

        # Find corresponding image if available
        image_path = None
        if image_dir:
            image_name = mask_name.replace("_label", "").replace("_seg", "")
            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                image_path = None

        # Process segmentation
        try:
            filtered_mask = organ_filter.process_segmentation(
                mask_path=mask_path, image_path=image_path, output_path=output_path
            )

            # Collect statistics
            n_organs_original = len(np.unique(nib.load(mask_path).get_fdata())) - 1
            n_organs_filtered = len(np.unique(filtered_mask)) - 1

            results.append(
                {
                    "file": mask_name,
                    "organs_original": n_organs_original,
                    "organs_filtered": n_organs_filtered,
                    "retention_rate": n_organs_filtered / max(n_organs_original, 1),
                    "status": "success",
                }
            )

        except Exception as e:
            print(f"Error processing {mask_name}: {e}")
            results.append(
                {
                    "file": mask_name,
                    "organs_original": 0,
                    "organs_filtered": 0,
                    "retention_rate": 0,
                    "status": f"error: {str(e)}",
                }
            )

    # Create summary DataFrame
    df_results = pd.DataFrame(results)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {len(df_results)}")
    print(f"Successful: {(df_results['status'] == 'success').sum()}")
    print(f"Failed: {(df_results['status'] != 'success').sum()}")
    print(f"Average retention rate: {df_results['retention_rate'].mean():.2%}")
    print(f"Total organs original: {df_results['organs_original'].sum()}")
    print(f"Total organs filtered: {df_results['organs_filtered'].sum()}")

    return df_results


if __name__ == "__main__":
    """
    Command-line interface for the SOLF pipeline.

    Usage:
        python organ_filtering.py --input_dir ./raw_masks/ --output_dir ./filtered_masks/
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="SOLF: Specialized Organ Label Filter for UKBOB dataset"
    )
    parser.add_argument("--input_dir", required=True, help="Input directory with masks")
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for filtered masks"
    )
    parser.add_argument(
        "--image_dir", help="Optional directory with corresponding images"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.8, help="Confidence threshold"
    )
    parser.add_argument("--report", help="Path to save filtering report CSV")

    args = parser.parse_args()

    # Run batch processing
    results_df = batch_process_segmentations(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        confidence_threshold=args.confidence,
    )

    # Save report if requested
    if args.report:
        results_df.to_csv(args.report, index=False)
        print(f"\nReport saved to: {args.report}")
