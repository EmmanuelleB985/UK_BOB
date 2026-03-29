"""
Training and validation loop implementation for Swin-BOB model.

This module implements the core training logic including:
- Epoch-wise training with mixed precision support
- Distributed training synchronization
- Validation with sliding window inference
- Checkpoint saving and tensorboard logging
- Memory-efficient training with gradient accumulation

Author: Emmanuelle Bourigault
License: MIT
"""

import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
from monai.data import decollate_batch
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from utils.utils import AverageMeter, distributed_all_gather


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    loss_func: Callable,
    args: Any,
) -> float:
    """
    Execute one training epoch with automatic mixed precision support.

    This function performs forward and backward passes for all batches in the
    training loader, updating model weights and tracking loss metrics. It supports
    both single-GPU and distributed training scenarios.

    Args:
        model: Neural network model to train
        loader: DataLoader providing training batches
        optimizer: Optimizer for updating model parameters
        scaler: GradScaler for automatic mixed precision training
        epoch: Current epoch number (for logging)
        loss_func: Loss function for computing training objective
        args: Configuration arguments containing training settings

    Returns:
        float: Average training loss for the epoch

    Notes:
        - Uses automatic mixed precision (AMP) for faster training when enabled
        - Synchronizes gradients across GPUs in distributed mode
        - Implements gradient accumulation for effective larger batch sizes
        - Provides real-time progress updates during training
    """
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    # Calculate total iterations for progress tracking
    total_iterations = len(loader)

    for idx, batch_data in enumerate(loader):
        # Extract data and labels from batch
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        # Move data to GPU
        data = data.cuda(args.rank, non_blocking=True)
        target = target.cuda(args.rank, non_blocking=True)

        # Zero gradients before forward pass
        for param in model.parameters():
            param.grad = None

        # Forward pass with automatic mixed precision
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        # Backward pass with gradient scaling for AMP
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Synchronize and aggregate loss across distributed processes
        if args.distributed:
            loss_list = distributed_all_gather(
                [loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length
            )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                n=args.batch_size * args.world_size,
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)

        # Log training progress
        if args.rank == 0:
            iteration_time = time.time() - start_time
            eta_seconds = iteration_time * (total_iterations - idx - 1)
            eta_string = format_time(eta_seconds)

            print(
                f"Epoch [{epoch}/{args.max_epochs}] "
                f"Iter [{idx + 1}/{total_iterations}] "
                f"Loss: {run_loss.avg:.4f} "
                f"Time: {iteration_time:.2f}s "
                f"ETA: {eta_string}"
            )

        start_time = time.time()

    # Clear gradients after epoch completion
    for param in model.parameters():
        param.grad = None

    return run_loss.avg


def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    acc_func: Callable,
    args: Any,
    model_inferer: Optional[Callable] = None,
    post_label: Optional[Callable] = None,
    post_pred: Optional[Callable] = None,
) -> float:
    """
    Execute one validation epoch with sliding window inference.

    This function evaluates model performance on the validation set using
    sliding window inference for handling large 3D volumes. It computes
    organ-wise Dice scores and aggregates metrics across distributed processes.

    Args:
        model: Neural network model to evaluate
        loader: DataLoader providing validation batches
        epoch: Current epoch number (for logging)
        acc_func: Metric function for computing accuracy (e.g., Dice score)
        args: Configuration arguments containing validation settings
        model_inferer: Sliding window inference function for large volumes
        post_label: Post-processing transform for ground truth labels
        post_pred: Post-processing transform for model predictions

    Returns:
        float: Average validation accuracy across all organs and samples

    Notes:
        - Uses sliding window inference to handle memory constraints
        - Computes per-organ Dice scores for detailed evaluation
        - Synchronizes metrics across GPUs in distributed mode
        - Operates in no_grad mode for memory efficiency
    """
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # Extract data and labels
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            data = data.cuda(args.rank, non_blocking=True)
            target = target.cuda(args.rank, non_blocking=True)

            # Perform inference with automatic mixed precision
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    # Use sliding window inference for large volumes
                    logits = model_inferer(data)
                else:
                    # Standard forward pass
                    logits = model(data)

            # Move target to CPU if predictions are on CPU
            if not logits.is_cuda:
                target = target.cpu()

            # Post-process predictions and labels
            val_labels_list = decollate_batch(target)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]

            val_outputs_list = decollate_batch(logits)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]

            # Compute accuracy metrics
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            # Synchronize metrics across distributed processes
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans],
                    out_numpy=True,
                    is_valid=idx < loader.sampler.valid_length,
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            # Log validation progress
            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    f"Val [{epoch}/{args.max_epochs}] "
                    f"Batch [{idx + 1}/{len(loader)}] "
                    f"Acc: {avg_acc:.4f} "
                    f"Time: {time.time() - start_time:.2f}s"
                )

                # Log per-organ accuracy for detailed analysis
                if idx == 0 and run_acc.avg.ndim > 0:
                    log_organ_metrics(run_acc.avg, epoch, args)

            start_time = time.time()

    return np.mean(run_acc.avg)


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    args: Any,
    filename: str = "model.pt",
    best_acc: float = 0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> None:
    """
    Save model checkpoint with training state information.

    This function saves the complete training state including model weights,
    optimizer state, and training metadata for resuming training or inference.

    Args:
        model: Neural network model to save
        epoch: Current epoch number
        args: Configuration arguments
        filename: Name of checkpoint file to save
        best_acc: Best validation accuracy achieved so far
        optimizer: Optional optimizer state to save
        scheduler: Optional scheduler state to save

    Notes:
        - Handles both single-GPU and distributed model saving
        - Saves optimizer and scheduler states for perfect training resumption
        - Creates backup of previous best model when saving new best
        - Includes metadata like epoch number and best accuracy
    """
    # Extract state dict handling distributed wrapper
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    # Prepare checkpoint dictionary
    save_dict = {
        "epoch": epoch,
        "best_acc": best_acc,
        "state_dict": state_dict,
        "args": vars(args),  # Save configuration for reproducibility
    }

    # Include optimizer state if provided
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()

    # Include scheduler state if provided
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    # Save checkpoint
    checkpoint_dir = os.path.join(args.logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(save_dict, checkpoint_path)

    print(f"Saved checkpoint: {checkpoint_path}")

    # Create symlink to latest checkpoint for easy access
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(filename, latest_path)


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_func: Callable,
    acc_func: Callable,
    args: Any,
    model_inferer: Optional[Callable] = None,
    scheduler: Optional[Any] = None,
    start_epoch: int = 0,
    post_label: Optional[Callable] = None,
    post_pred: Optional[Callable] = None,
) -> float:
    """
    Execute the complete training pipeline for the Swin-BOB model.

    This is the main training orchestrator that manages the training loop,
    validation, checkpointing, and logging over all epochs. It implements
    early stopping, learning rate scheduling, and distributed synchronization.

    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for parameter updates
        loss_func: Loss function for training objective
        acc_func: Accuracy metric function
        args: Configuration arguments containing all training settings
        model_inferer: Optional sliding window inference function
        scheduler: Optional learning rate scheduler
        start_epoch: Starting epoch number (for resumed training)
        post_label: Post-processing for ground truth labels
        post_pred: Post-processing for model predictions

    Returns:
        float: Best validation accuracy achieved during training

    Notes:
        - Implements automatic mixed precision training
        - Saves checkpoints at regular intervals and for best models
        - Logs metrics to TensorBoard for visualization
        - Supports training resumption from checkpoints
        - Handles distributed training synchronization
    """
    # Initialize TensorBoard writer for rank 0 process
    writer = None
    if args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        # Log initial configuration
        log_configuration(writer, args)
        print(f"Writing TensorBoard logs to: {args.logdir}")

    # Initialize gradient scaler for AMP
    scaler = GradScaler() if args.amp else None

    # Track best accuracy for model selection
    val_acc_max = 0.0

    # Main training loop
    for epoch in range(start_epoch, args.max_epochs):
        # Distributed sampler shuffling
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            if val_loader.sampler is not None:
                val_loader.sampler.set_epoch(epoch)

        # Adjust learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            if args.rank == 0:
                print(f"Epoch {epoch}: Learning rate = {current_lr:.6f}")
                if writer is not None:
                    writer.add_scalar("train/learning_rate", current_lr, epoch)

        # Training epoch
        print(f"\n{'='*60}")
        print(f"Starting Epoch {epoch + 1}/{args.max_epochs}")
        print(f"{'='*60}")

        epoch_start_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, epoch, loss_func, args
        )

        # Log training metrics
        if args.rank == 0:
            print(f"Training loss: {train_loss:.4f}")
            if writer is not None:
                writer.add_scalar("train/loss", train_loss, epoch)

        # Validation at specified intervals
        if (epoch + 1) % args.val_every == 0:
            epoch_time = time.time() - epoch_start_time

            if args.rank == 0:
                print(f"Epoch time: {format_time(epoch_time)}")
                print("\nRunning validation...")

            val_acc = val_epoch(
                model,
                val_loader,
                epoch,
                acc_func,
                args,
                model_inferer,
                post_label,
                post_pred,
            )

            # Log validation metrics
            if args.rank == 0:
                print(f"Validation accuracy: {val_acc:.4f}")
                if writer is not None:
                    writer.add_scalar("val/acc", val_acc, epoch)

                # Save checkpoint if best model
                if val_acc > val_acc_max:
                    print(f"New best accuracy! {val_acc:.4f} > {val_acc_max:.4f}")
                    val_acc_max = val_acc

                    if args.save_checkpoint:
                        save_checkpoint(
                            model,
                            epoch,
                            args,
                            filename="model_best.pt",
                            best_acc=val_acc,
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )

                # Save periodic checkpoint
                if args.save_checkpoint and (epoch + 1) % 100 == 0:
                    save_checkpoint(
                        model,
                        epoch,
                        args,
                        filename=f"model_epoch_{epoch}.pt",
                        best_acc=val_acc_max,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )

    # Save final model
    if args.rank == 0 and args.save_checkpoint:
        print("\nSaving final model...")
        save_checkpoint(
            model,
            args.max_epochs,
            args,
            filename="model_final.pt",
            best_acc=val_acc_max,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    # Close TensorBoard writer
    if writer is not None:
        writer.close()

    return val_acc_max


def log_organ_metrics(organ_scores: np.ndarray, epoch: int, args: Any) -> None:
    """
    Log per-organ segmentation metrics for detailed analysis.

    Args:
        organ_scores: Array of Dice scores for each organ
        epoch: Current epoch number
        args: Configuration arguments

    Notes:
        Provides detailed breakdown of segmentation performance
        for each anatomical structure in the dataset.
    """
    # Define organ names (subset shown for brevity)
    organ_names = [
        "spleen",
        "kidney_right",
        "kidney_left",
        "gallbladder",
        "liver",
        "stomach",
        "pancreas",
        "adrenal_gland_right",
        "adrenal_gland_left",
        # ... additional organs
    ]

    # Log top and bottom performing organs
    if len(organ_scores) > 0:
        sorted_indices = np.argsort(organ_scores)

        print("\nTop 5 performing organs:")
        for i in sorted_indices[-5:]:
            if i < len(organ_names):
                print(f"  {organ_names[i]}: {organ_scores[i]:.4f}")

        print("\nBottom 5 performing organs:")
        for i in sorted_indices[:5]:
            if i < len(organ_names):
                print(f"  {organ_names[i]}: {organ_scores[i]:.4f}")


def log_configuration(writer: SummaryWriter, args: Any) -> None:
    """
    Log training configuration to TensorBoard for experiment tracking.

    Args:
        writer: TensorBoard SummaryWriter instance
        args: Configuration arguments to log
    """
    config_text = "Training Configuration:\n"
    config_text += "=" * 50 + "\n"

    important_args = [
        "batch_size",
        "max_epochs",
        "optim_name",
        "optim_lr",
        "feature_size",
        "roi_x",
        "roi_y",
        "roi_z",
        "in_channels",
        "out_channels",
        "distributed",
    ]

    for arg in important_args:
        if hasattr(args, arg):
            config_text += f"{arg}: {getattr(args, arg)}\n"

    writer.add_text("configuration", config_text, 0)


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable time string.

    Args:
        seconds: Number of seconds to format

    Returns:
        str: Formatted time string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"
