"""
Data loading and preprocessing utilities for 3D medical image segmentation.

This module provides comprehensive data handling functionality including:
- Custom data loaders for large-scale 3D medical imaging datasets
- Advanced preprocessing and augmentation pipelines
- Distributed sampling strategies for multi-GPU training
- Memory-efficient caching mechanisms for large datasets
- Custom transforms for medical image processing

Author: Emmanuelle Bourigault
License: MIT
"""

import math
import os
from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.distributed as dist

from monai import data, transforms
from monai.data import load_decathlon_datalist, CacheDataset, SmartCacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandAffined, ToTensord, MapTransform, ResizeWithPadOrCropd,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd
)


class AbdomenSliceSelector(MapTransform):
    """
    Custom transform for selecting anatomically relevant slices from 3D volumes.
    
    This transform implements intelligent slice selection based on anatomical
    content, reducing computational overhead by focusing on slices containing
    relevant structures. It's particularly useful for abdominal imaging where
    many superior and inferior slices may contain only background.
    
    Attributes:
        keys (List[str]): Keys of data dictionary to apply transform to
        min_annotation (int): Minimum number of annotated voxels per slice
        
    Example:
        >>> selector = AbdomenSliceSelector(
        ...     keys=["image", "label"],
        ...     min_annotation=100
        ... )
        >>> filtered_data = selector(data_dict)
    """
    
    def __init__(self, keys: List[str], min_annotation: int = 1):
        """
        Initialize the slice selector transform.
        
        Args:
            keys: List of keys to apply the transform (typically ["image", "label"])
            min_annotation: Minimum sum of label values for slice retention.
                           Higher values result in more aggressive filtering.
        """
        super().__init__(keys)
        self.min_annotation = min_annotation
    
    def __call__(self, data: Dict) -> Dict:
        """
        Apply slice selection to input data.
        
        This method analyzes the label volume to identify slices containing
        anatomical structures and filters both image and label volumes accordingly.
        
        Args:
            data: Dictionary containing image and label tensors
            
        Returns:
            Dict: Filtered data with selected slices
            
        Raises:
            ValueError: If label key is missing or tensor shapes are incorrect
        """
        d = dict(data)
        
        # Validate input data
        if "label" not in d:
            raise ValueError(
                "AbdomenSliceSelector requires a 'label' key in the data dictionary."
            )
        
        label_arr = d["label"]
        if label_arr.ndim != 4:
            raise ValueError(
                f"Expected label tensor of shape (C, H, W, D), but got {label_arr.shape}."
            )
        
        # Compute annotation density per slice
        # Sum over channels and spatial dimensions (H, W) for each slice
        annotation_sum = label_arr.sum(axis=(0, 1, 2))
        
        # Select slices meeting annotation threshold
        selected_indices = np.where(annotation_sum >= self.min_annotation)[0]
        
        # Fallback to all slices if no slices meet criteria
        if len(selected_indices) == 0:
            print(f"Warning: No slices meet annotation threshold {self.min_annotation}, "
                  f"using all {label_arr.shape[3]} slices")
            selected_indices = np.arange(label_arr.shape[3])
        else:
            print(f"Selected {len(selected_indices)}/{label_arr.shape[3]} slices "
                  f"with annotation >= {self.min_annotation}")
        
        # Apply selection to all specified keys
        for key in self.keys:
            arr = d[key]
            if arr.ndim != 4:
                raise ValueError(
                    f"Expected tensor of shape (C, H, W, D) for key '{key}', "
                    f"but got {arr.shape}."
                )
            d[key] = arr[..., selected_indices]
        
        return d


class DistributedSampler(Sampler):
    """
    Custom distributed sampler for multi-GPU training with improved load balancing.
    
    This sampler ensures even distribution of data across multiple GPUs while
    maintaining reproducibility and supporting both training and validation modes.
    It implements smart padding strategies to handle datasets that don't divide
    evenly across GPUs.
    
    Attributes:
        dataset: Dataset to sample from
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process
        shuffle: Whether to shuffle indices each epoch
        make_even: Whether to pad dataset for even distribution
        
    Example:
        >>> sampler = DistributedSampler(
        ...     dataset=train_dataset,
        ...     num_replicas=8,
        ...     rank=0,
        ...     shuffle=True
        ... )
        >>> dataloader = DataLoader(dataset, sampler=sampler)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        make_even: bool = True
    ):
        """
        Initialize the distributed sampler.
        
        Args:
            dataset: Dataset to sample from
            num_replicas: Number of distributed processes (auto-detected if None)
            rank: Current process rank (auto-detected if None)
            shuffle: Enable shuffling for training mode
            make_even: Pad dataset to ensure even distribution across GPUs
            
        Raises:
            RuntimeError: If distributed package is not available
        """
        # Auto-detect distributed settings if not provided
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package not available")
            num_replicas = dist.get_world_size()
        
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package not available")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.make_even = make_even
        
        # Calculate samples per GPU
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        
        # Track valid sample count for accurate metric aggregation
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])
        
        print(f"Rank {self.rank}: Handling {self.valid_length} samples "
              f"from total dataset of {len(self.dataset)}")
    
    def __iter__(self):
        """
        Generate indices for the current epoch.
        
        Returns:
            Iterator over indices assigned to current rank
        """
        # Deterministic shuffling based on epoch seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Pad dataset for even distribution if needed
        if self.make_even:
            padding_size = self.total_size - len(indices)
            
            if padding_size > 0:
                if padding_size < len(indices):
                    # Repeat first samples for small padding
                    indices += indices[:padding_size]
                else:
                    # Random sampling for large padding
                    extra_ids = np.random.RandomState(self.epoch).randint(
                        low=0, high=len(indices), size=padding_size
                    )
                    indices += [indices[idx] for idx in extra_ids]
            
            assert len(indices) == self.total_size, \
                f"Index count {len(indices)} != total_size {self.total_size}"
        
        # Select subset for current rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        
        return iter(indices)
    
    def __len__(self):
        """Return the number of samples for current rank."""
        return self.num_samples
    
    def set_epoch(self, epoch: int):
        """
        Set epoch for shuffling randomization.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch


def get_train_transforms(args: Any) -> Compose:
    """
    Create comprehensive training data augmentation pipeline.
    
    This function assembles a complete preprocessing and augmentation pipeline
    for training, including intensity normalization, spatial transformations,
    and various augmentation techniques proven effective for medical imaging.
    
    Args:
        args: Configuration arguments containing transform parameters
        
    Returns:
        Compose: MONAI Compose transform with full augmentation pipeline
    """
    train_transforms = Compose([
        # Load and ensure correct format
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Reorient to standard orientation (RAS+)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # Resample to target spacing
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest"),
        ),
        
        # Intensity normalization
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        
        # Remove background
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # Random crop with balanced sampling
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=1,  # Positive sample ratio
            neg=1,  # Negative sample ratio
            num_samples=4,  # Number of samples per image
            image_key="image",
            image_threshold=0,
        ),
        
        # Spatial augmentations
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=args.RandFlipd_prob,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=args.RandFlipd_prob,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=args.RandFlipd_prob,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=args.RandRotate90d_prob,
            max_k=3,
        ),
        
        # Intensity augmentations
        RandScaleIntensityd(
            keys=["image"],
            factors=0.1,
            prob=args.RandScaleIntensityd_prob,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.1,
            prob=args.RandShiftIntensityd_prob,
        ),
        
        # Convert to tensor
        ToTensord(keys=["image", "label"]),
    ])
    
    return train_transforms


def get_val_transforms(args: Any) -> Compose:
    """
    Create validation/inference preprocessing pipeline.
    
    This function creates a minimal preprocessing pipeline for validation
    and inference, excluding augmentations to ensure reproducibility.
    
    Args:
        args: Configuration arguments containing transform parameters
        
    Returns:
        Compose: MONAI Compose transform for validation
    """
    val_transforms = Compose([
        # Load and ensure correct format
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Reorient to standard orientation
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # Resample to target spacing
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest"),
        ),
        
        # Intensity normalization
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        
        # Convert to tensor
        ToTensord(keys=["image", "label"]),
    ])
    
    return val_transforms


def get_loader(args: Any) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized data loaders for training and validation.
    
    This function handles the complete data loading pipeline including:
    - Loading dataset splits from JSON
    - Creating appropriate transforms
    - Setting up caching strategies
    - Configuring distributed sampling
    - Creating DataLoader instances
    
    Args:
        args: Configuration arguments containing all data loading parameters
        
    Returns:
        Tuple containing:
            - Training DataLoader
            - Validation DataLoader
            
    Raises:
        FileNotFoundError: If dataset JSON file is not found
        ValueError: If dataset format is invalid
    """
    # Load dataset configuration
    data_dir = args.data_dir
    json_path = os.path.join(data_dir, args.json_list)
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")
    
    print(f"Loading dataset from: {json_path}")
    datalist = load_decathlon_datalist(json_path, True, "training", base_dir=data_dir)
    val_datalist = load_decathlon_datalist(json_path, True, "validation", base_dir=data_dir)
    
    print(f"Found {len(datalist)} training samples and {len(val_datalist)} validation samples")
    
    # Create transforms
    train_transforms = get_train_transforms(args)
    val_transforms = get_val_transforms(args)
    
    # Create datasets with appropriate caching strategy
    if args.use_normal_dataset:
        # Standard dataset without caching (for debugging)
        train_ds = data.Dataset(data=datalist, transform=train_transforms)
        val_ds = data.Dataset(data=val_datalist, transform=val_transforms)
        print("Using standard Dataset (no caching)")
    else:
        # Smart caching for efficient memory usage
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,  # Cache replacement rate
            cache_num=24,  # Number of items to cache
        )
        val_ds = CacheDataset(
            data=val_datalist,
            transform=val_transforms,
            cache_num=6,  # Cache all validation data
        )
        print("Using SmartCacheDataset for training and CacheDataset for validation")
    
    # Setup distributed sampling if needed
    train_sampler = None
    val_sampler = None
    
    if args.distributed:
        train_sampler = DistributedSampler(
            dataset=train_ds,
            shuffle=True,
            make_even=True
        )
        val_sampler = DistributedSampler(
            dataset=val_ds,
            shuffle=False,
            make_even=False
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # Use batch_size=1 for validation to handle varying sizes
        shuffle=False,
        num_workers=args.workers,
        sampler=val_sampler,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
    )
    
    return train_loader, val_loader


def create_data_splits(
    data_dir: str,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
) -> Dict:
    """
    Create train/validation/test splits from a data directory.
    
    This utility function automatically splits a dataset into training,
    validation, and test sets based on specified ratios, maintaining
    reproducibility through seeding.
    
    Args:
        data_dir: Directory containing image and label subdirectories
        split_ratios: Tuple of (train, val, test) ratios (must sum to 1.0)
        seed: Random seed for reproducible splits
        
    Returns:
        Dict: Dataset configuration in Decathlon format
        
    Example:
        >>> splits = create_data_splits(
        ...     data_dir="./data/UKBOB/",
        ...     split_ratios=(0.8, 0.1, 0.1)
        ... )
        >>> save_json(splits, "./dataset_splits.json")
    """
    if not math.isclose(sum(split_ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
    
    # Find all image files
    image_dir = Path(data_dir) / "images"
    label_dir = Path(data_dir) / "labels"
    
    image_files = sorted(image_dir.glob("*.nii.gz"))
    
    # Create file pairs
    data_pairs = []
    for img_path in image_files:
        label_name = img_path.name.replace("_T1", "_label").replace("_T2", "_label")
        label_path = label_dir / label_name
        
        if label_path.exists():
            data_pairs.append({
                "image": str(img_path.relative_to(data_dir)),
                "label": str(label_path.relative_to(data_dir))
            })
    
    # Perform split
    np.random.seed(seed)
    np.random.shuffle(data_pairs)
    
    n_total = len(data_pairs)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    
    dataset_json = {
        "training": data_pairs[:n_train],
        "validation": data_pairs[n_train:n_train + n_val],
        "test": data_pairs[n_train + n_val:],
        "labels": {
            str(i): f"organ_{i}" for i in range(72)
        },
        "tensorImageSize": "3D"
    }
    
    print(f"Created splits: Train={n_train}, Val={n_val}, Test={n_total-n_train-n_val}")
    
    return dataset_json