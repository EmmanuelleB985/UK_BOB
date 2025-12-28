"""
Main training script for Swin-BOB: 3D Medical Image Segmentation on UK Biobank Dataset

This module provides the main entry point for training the Swin-UNetr model on large-scale
3D MRI data from the UK Biobank. It supports both single-GPU and distributed training,
with various optimization strategies and data augmentation techniques.

Author: Emmanuelle Bourigault
License: MIT
"""

import argparse
import os
from functools import partial
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all training configurations
        including model architecture, data paths, optimization parameters, and
        distributed training settings.
    """
    parser = argparse.ArgumentParser(
        description="Swin-BOB: Advanced 3D Medical Image Segmentation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model checkpoint arguments
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--logdir", default="test", type=str,
                        help="Directory name for tensorboard logs (created under ./runs/)")
    parser.add_argument("--pretrained_dir", default="./pretrained_models/", type=str,
                        help="Directory containing pretrained model weights")
    parser.add_argument("--pretrained_model_name", default="", type=str,
                        help="Name of pretrained model file to load")
    parser.add_argument("--save_checkpoint", action="store_true",
                        help="Enable checkpoint saving during training")
    
    # Dataset configuration
    parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str,
                        help="Root directory containing the dataset")
    parser.add_argument("--json_list", default="dataset_0.json", type=str,
                        help="JSON file containing dataset split information")
    parser.add_argument("--use_normal_dataset", action="store_true",
                        help="Use standard MONAI Dataset class instead of cached dataset")
    
    # Training hyperparameters
    parser.add_argument("--max_epochs", default=5000, type=int,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Batch size for training")
    parser.add_argument("--sw_batch_size", default=4, type=int,
                        help="Batch size for sliding window inference during validation")
    parser.add_argument("--val_every", default=100, type=int,
                        help="Validation frequency (run validation every N epochs)")
    parser.add_argument("--workers", default=2, type=int,
                        help="Number of data loading workers per GPU")
    
    # Optimization parameters
    parser.add_argument("--optim_lr", default=1e-4, type=float,
                        help="Initial learning rate for optimizer")
    parser.add_argument("--optim_name", default="adamw", type=str,
                        choices=["adam", "adamw", "sgd"],
                        help="Optimization algorithm to use")
    parser.add_argument("--reg_weight", default=1e-5, type=float,
                        help="L2 regularization weight (weight decay)")
    parser.add_argument("--momentum", default=0.99, type=float,
                        help="Momentum factor for SGD optimizer")
    parser.add_argument("--noamp", action="store_true",
                        help="Disable automatic mixed precision training")
    
    # Learning rate scheduling
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str,
                        choices=["warmup_cosine", "cosine_anneal", "none"],
                        help="Learning rate scheduling strategy")
    parser.add_argument("--warmup_epochs", default=50, type=int,
                        help="Number of warmup epochs for warmup_cosine scheduler")
    
    # Model architecture parameters
    parser.add_argument("--feature_size", default=48, type=int,
                        help="Feature dimension for Swin-UNetr backbone")
    parser.add_argument("--in_channels", default=1, type=int,
                        help="Number of input channels (e.g., 1 for single modality MRI)")
    parser.add_argument("--out_channels", default=72, type=int,
                        help="Number of output channels (segmentation classes)")
    parser.add_argument("--spatial_dims", default=3, type=int,
                        help="Spatial dimensions of input data (2D or 3D)")
    parser.add_argument("--dropout_rate", default=0.0, type=float,
                        help="Dropout rate for model regularization")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float,
                        help="Stochastic depth rate for drop path regularization")
    parser.add_argument("--norm_name", default="instance", type=str,
                        choices=["instance", "batch", "layer"],
                        help="Normalization layer type")
    
    # Data preprocessing parameters
    parser.add_argument("--a_min", default=-175.0, type=float,
                        help="Minimum HU value for intensity scaling")
    parser.add_argument("--a_max", default=250.0, type=float,
                        help="Maximum HU value for intensity scaling")
    parser.add_argument("--b_min", default=0.0, type=float,
                        help="Minimum output value for intensity scaling")
    parser.add_argument("--b_max", default=1.0, type=float,
                        help="Maximum output value for intensity scaling")
    parser.add_argument("--space_x", default=1.5, type=float,
                        help="Target voxel spacing in x direction (mm)")
    parser.add_argument("--space_y", default=1.5, type=float,
                        help="Target voxel spacing in y direction (mm)")
    parser.add_argument("--space_z", default=2.0, type=float,
                        help="Target voxel spacing in z direction (mm)")
    parser.add_argument("--roi_x", default=96, type=int,
                        help="ROI size in x direction for random cropping")
    parser.add_argument("--roi_y", default=96, type=int,
                        help="ROI size in y direction for random cropping")
    parser.add_argument("--roi_z", default=96, type=int,
                        help="ROI size in z direction for random cropping")
    
    # Data augmentation parameters
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float,
                        help="Probability for random flip augmentation")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float,
                        help="Probability for random 90-degree rotation augmentation")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float,
                        help="Probability for random intensity scaling augmentation")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float,
                        help="Probability for random intensity shifting augmentation")
    
    # Inference parameters
    parser.add_argument("--infer_overlap", default=0.5, type=float,
                        help="Overlap ratio for sliding window inference (0.0 to 1.0)")
    
    # Distributed training parameters
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training across multiple GPUs")
    parser.add_argument("--world_size", default=1, type=int,
                        help="Number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int,
                        help="Node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str,
                        help="URL for distributed training process group initialization")
    parser.add_argument("--dist-backend", default="nccl", type=str,
                        help="Distributed backend (nccl recommended for GPUs)")
    
    # Advanced training options
    parser.add_argument("--use_checkpoint", action="store_true",
                        help="Enable gradient checkpointing to reduce memory usage")
    parser.add_argument("--use_ssl_pretrained", action="store_true",
                        help="Use self-supervised pretrained weights for initialization")
    parser.add_argument("--resume_ckpt", action="store_true",
                        help="Resume training from pretrained checkpoint")
    
    # Loss function parameters
    parser.add_argument("--squared_dice", action="store_true",
                        help="Use squared Dice loss formulation")
    parser.add_argument("--smooth_dr", default=1e-6, type=float,
                        help="Smoothing constant for Dice loss denominator")
    parser.add_argument("--smooth_nr", default=0.0, type=float,
                        help="Smoothing constant for Dice loss numerator")
    
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    
    return args


def initialize_model(args: argparse.Namespace) -> torch.nn.Module:
    """
    Initialize the Swin-UNetr model with specified configuration.
    
    This function creates a Swin-UNetr model instance and optionally loads
    pretrained weights from either supervised or self-supervised pretraining.
    
    Args:
        args: Parsed command-line arguments containing model configuration
        
    Returns:
        torch.nn.Module: Initialized Swin-UNetr model ready for training
        
    Raises:
        ValueError: If self-supervised weights are requested but not available
    """
    print(f"Initializing Swin-UNetr model with feature_size={args.feature_size}")
    
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,  # Note: Using 0.0 instead of args.dropout_rate for stability
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
    )
    
    # Load pretrained checkpoint if specified
    if args.resume_ckpt and args.pretrained_model_name:
        checkpoint_path = os.path.join(args.pretrained_dir, args.pretrained_model_name)
        print(f"Loading pretrained weights from: {checkpoint_path}")
        model_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        model.load_state_dict(model_dict)
        print("Successfully loaded pretrained weights")
    
    # Load self-supervised pretrained weights if specified
    if args.use_ssl_pretrained:
        try:
            ssl_checkpoint_path = "./pretrained_models/model_swinvit.pt"
            print(f"Loading SSL pretrained weights from: {ssl_checkpoint_path}")
            model_dict = torch.load(ssl_checkpoint_path, map_location="cpu")
            state_dict = model_dict["state_dict"]
            
            # Clean state dict keys for compatibility
            state_dict = clean_state_dict_keys(state_dict)
            
            model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded self-supervised Swin-UNetr backbone weights")
        except (FileNotFoundError, KeyError, ValueError) as e:
            raise ValueError(f"Failed to load self-supervised weights: {str(e)}")
    
    # Calculate and display model size
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {pytorch_total_params:,}")
    
    return model


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Clean state dictionary keys for compatibility with model architecture.
    
    This function handles common naming mismatches between saved checkpoints
    and current model architecture, such as 'module.' prefixes from DataParallel
    or different naming conventions.
    
    Args:
        state_dict: State dictionary with potentially incompatible keys
        
    Returns:
        Dict[str, torch.Tensor]: Cleaned state dictionary with compatible keys
    """
    # Remove 'module.' prefix from DataParallel/DistributedDataParallel
    if "module." in list(state_dict.keys())[0]:
        print("Removing 'module.' prefix from state dict keys")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    
    # Handle different naming conventions for Swin-ViT backbone
    if "swin_vit" in list(state_dict.keys())[0]:
        print("Renaming 'swin_vit' to 'swinViT' in state dict keys")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
    
    return state_dict


def setup_optimizer_and_scheduler(
    model: torch.nn.Module,
    args: argparse.Namespace
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Setup optimizer and learning rate scheduler based on configuration.
    
    Args:
        model: Neural network model to optimize
        args: Configuration arguments containing optimizer settings
        
    Returns:
        Tuple containing:
            - Optimizer instance (Adam, AdamW, or SGD)
            - Optional learning rate scheduler instance
            
    Raises:
        ValueError: If unsupported optimizer name is specified
    """
    print(f"Setting up {args.optim_name} optimizer with lr={args.optim_lr}")
    
    # Create optimizer based on specified algorithm
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.optim_lr,
            weight_decay=args.reg_weight
        )
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.optim_lr,
            weight_decay=args.reg_weight
        )
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.optim_lr,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.reg_weight
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optim_name}")
    
    # Create learning rate scheduler if specified
    scheduler = None
    if args.lrschedule == "warmup_cosine":
        print(f"Using warmup cosine annealing scheduler with {args.warmup_epochs} warmup epochs")
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        print("Using cosine annealing scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs
        )
    
    return optimizer, scheduler


def main_worker(gpu: int, args: argparse.Namespace) -> float:
    """
    Main worker function for both single-GPU and distributed training.
    
    This function handles the complete training pipeline including:
    - Distributed process group initialization
    - Model initialization and checkpoint loading
    - Data loader creation
    - Optimizer and scheduler setup
    - Training loop execution
    
    Args:
        gpu: GPU index to use for training
        args: Parsed command-line arguments
        
    Returns:
        float: Best validation accuracy achieved during training
    """
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    
    # Initialize distributed training if enabled
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        print(f"Initialized process group: rank={args.rank}, world_size={args.world_size}")
    
    # Set CUDA device
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    
    # Load data
    args.test_mode = False
    train_loader, val_loader = get_loader(args)
    
    if args.rank == 0:
        print(f"Training configuration:")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Max epochs: {args.max_epochs}")
        print(f"  - Training samples: {len(train_loader.dataset)}")
        print(f"  - Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model = initialize_model(args)
    
    # Setup loss function and metrics
    if args.squared_dice:
        dice_loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    
    post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)
    dice_acc = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN,
        get_not_nans=True
    )
    
    # Setup sliding window inference
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    
    # Load checkpoint if specified
    best_acc = 0
    start_epoch = 0
    
    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        
        # Load model state
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        
        # Restore training state
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        
        print(f"Resumed from epoch {start_epoch} with best_acc={best_acc:.4f}")
    
    # Move model to GPU
    model.cuda(args.gpu)
    
    # Setup distributed training wrapper if needed
    if args.distributed:
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu
        )
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
    
    # Start training
    accuracy = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    
    return accuracy


def main():
    """
    Main entry point for the training script.
    
    This function parses arguments and launches either single-GPU
    or distributed multi-GPU training based on the configuration.
    """
    args = parse_arguments()
    
    # Setup distributed training if enabled
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print(f"Launching distributed training on {args.ngpus_per_node} GPUs")
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        print("Starting single-GPU training")
        main_worker(gpu=0, args=args)


if __name__ == "__main__":
    main()