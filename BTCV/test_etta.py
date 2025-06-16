# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio
from skimage import exposure, measure
from utils.data_utils import get_loader
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
import SimpleITK as sitk
import torch.nn.functional as F

class EntropyAdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, tune_affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.tune_affine = tune_affine
        
        if tune_affine and elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            with torch.no_grad():
                self.weight.copy_(self.ln.weight)
                self.bias.copy_(self.ln.bias)
    
    def forward(self, x):
        if self.training and self.tune_affine:
            dims = tuple(range(-len(self.ln.normalized_shape), 0))
            mean = x.mean(dims, keepdim=True)
            var = x.var(dims, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + self.ln.eps)
            return self.weight * x_norm + self.bias
        else:
            return self.ln(x)

class EntropyAdaptiveInstanceNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, tune_affine=True):
        super().__init__()
        self.inst_norm = nn.InstanceNorm3d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.tune_affine = tune_affine
        
        if tune_affine and affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
            with torch.no_grad():
                self.weight.copy_(self.inst_norm.weight)
                self.bias.copy_(self.inst_norm.bias)
    
    def forward(self, x):
        if self.training and self.tune_affine:
            mean = x.mean([2, 3, 4], keepdim=True)
            var = x.var([2, 3, 4], keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + self.inst_norm.eps)
            return self.weight[None, :, None, None, None] * x_norm + self.bias[None, :, None, None, None]
        else:
            return self.inst_norm(x)

class EntropyAdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, tune_affine=True):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
        self.tune_affine = tune_affine
        
        if tune_affine and affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
            with torch.no_grad():
                self.weight.copy_(self.group_norm.weight)
                self.bias.copy_(self.group_norm.bias)
    
    def forward(self, x):
        if self.training and self.tune_affine:
            N, C, D, H, W = x.shape
            G = self.group_norm.num_groups
            x_reshaped = x.view(N, G, C // G, D, H, W)
            mean = x_reshaped.mean([2, 3, 4, 5], keepdim=True)
            var = x_reshaped.var([2, 3, 4, 5], keepdim=True, unbiased=False)
            x_norm = (x_reshaped - mean) / torch.sqrt(var + self.group_norm.eps)
            x_norm = x_norm.view(N, C, D, H, W)
            return self.weight[None, :, None, None, None] * x_norm + self.bias[None, :, None, None, None]
        else:
            return self.group_norm(x)

class SwinUNETRWithTTA(nn.Module):
    def __init__(self, original_model, num_tta_passes=3, lr=0.01):
        super().__init__()
        self.model = original_model
        self.num_tta_passes = num_tta_passes
        self.lr = lr
        
        self._replace_norm_layers()
        self.affine_params = []
        for m in self.model.modules():
            if isinstance(m, (EntropyAdaptiveLayerNorm, EntropyAdaptiveInstanceNorm3d, EntropyAdaptiveGroupNorm)):
                if hasattr(m, 'weight') and hasattr(m, 'bias') and m.tune_affine:
                    self.affine_params.append({'params': [m.weight, m.bias]})
        
        if self.affine_params:
            self.tta_optimizer = torch.optim.SGD(self.affine_params, lr=lr)
        else:
            self.tta_optimizer = None
    
    def _replace_norm_layers(self):
        """Replace normalization layers with entropy-adaptive versions"""
        def replace_module(module):
            for name, child in module.named_children():
                if isinstance(child, nn.LayerNorm):
                    new_norm = EntropyAdaptiveLayerNorm(
                        normalized_shape=child.normalized_shape,
                        eps=child.eps,
                        elementwise_affine=child.elementwise_affine,
                        tune_affine=True
                    )
                    setattr(module, name, new_norm)
                elif isinstance(child, nn.InstanceNorm3d):
                    new_norm = EntropyAdaptiveInstanceNorm3d(
                        num_features=child.num_features,
                        eps=child.eps,
                        momentum=child.momentum,
                        affine=child.affine,
                        tune_affine=True
                    )
                    setattr(module, name, new_norm)
                elif isinstance(child, nn.GroupNorm):
                    new_norm = EntropyAdaptiveGroupNorm(
                        num_groups=child.num_groups,
                        num_channels=child.num_channels,
                        eps=child.eps,
                        affine=child.affine,
                        tune_affine=True
                    )
                    setattr(module, name, new_norm)
                elif len(list(child.children())) > 0:
                    replace_module(child)
        
        replace_module(self.model)
    
    def forward(self, x):
        return self.model(x)

def resample_3d(img, target_shape, mode='nearest'):
    input_image = sitk.GetImageFromArray(img)
    original_size = input_image.GetSize()
    original_spacing = input_image.GetSpacing()
    
    new_size = [int(target_shape[2]), int(target_shape[1]), int(target_shape[0])]
    new_spacing = [
        original_spacing[0] * (original_size[0] / new_size[0]),
        original_spacing[1] * (original_size[1] / new_size[1]),
        original_spacing[2] * (original_size[2] / new_size[2])
    ]
    
    interpolator = sitk.sitkNearestNeighbor if mode == 'nearest' else sitk.sitkLinear
    
    resampled_image = sitk.Resample(
        input_image,
        new_size,
        sitk.Transform(),
        interpolator,
        input_image.GetOrigin(),
        new_spacing,
        input_image.GetDirection(),
        0,
        input_image.GetPixelID()
    )
    
    return sitk.GetArrayFromImage(resampled_image)

def dice(pred, gt):
    """Calculate Dice coefficient between two numpy arrays"""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    
    if union == 0:  # Handle empty masks
        return 1.0 if np.array_equal(pred, gt) else 0.0
    
    return (2.0 * intersection) / union

def postprocess_segmentation(segmentation):
    """Keep only largest connected component for each class"""
    cleaned = np.zeros_like(segmentation)
    
    for class_id in np.unique(segmentation):
        if class_id == 0:  # Skip background
            continue
            
        # Create binary mask for current class
        mask = (segmentation == class_id).astype(np.uint8)
        
        # Label connected components
        labeled = measure.label(mask, connectivity=3)
        regions = measure.regionprops(labeled)
        
        if len(regions) == 0:
            continue
            
        # Select largest component
        largest_region = max(regions, key=lambda x: x.area)
        cleaned[labeled == largest_region.label] = class_id
    return cleaned

def create_comparison_gif(scan, pred, gt, output_path, num_frames=100, 
                         downsample=2, duration=0.5):
    """Create annotated comparison GIF with animation"""
    try:
        # Validate dimensions
        assert scan.shape == pred.shape == gt.shape, f"Dimension mismatch: Scan{scan.shape} Pred{pred.shape} GT{gt.shape}"
        
        # Preprocess scan
        scan = exposure.rescale_intensity(scan, in_range=(np.percentile(scan, 1), np.percentile(scan, 99)))
        scan = (scan * 255).astype(np.uint8)
        
        z_size = scan.shape[-1]
        z_indices = np.clip(np.linspace(0, z_size-1, num_frames).astype(int), 0, z_size-1)
        
        # Downsample if requested
        if downsample > 1:
            scan = scan[::downsample, ::downsample, :]
            pred = pred[::downsample, ::downsample, :]
            gt = gt[::downsample, ::downsample, :]
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        fig.subplots_adjust(wspace=0.01)
        cmap = plt.cm.get_cmap('hsv', 14)
        annotation_params = {
            'fontsize': 10,
            'color': 'white',
            'bbox': dict(facecolor='black', alpha=0.5, edgecolor='none')
        }
        frames = []
        for z in z_indices:
            ax1.clear()
            ax2.clear()
            
            try:
                # Rotate slices 90 degrees clockwise
                # Prediction plot
                ax1.imshow(np.rot90(scan[:, :, z], k=1), cmap='gray', origin='lower')
                pred_mask = np.ma.masked_where(np.rot90(pred[:, :, z] == 0, k=1), np.rot90(pred[:, :, z], k=1))
                ax1.imshow(pred_mask, cmap=cmap, alpha=0.4, vmin=0, vmax=13)
                ax1.text(0.05, 0.95, 'Prediction', transform=ax1.transAxes, **annotation_params)
                
                # Ground truth plot
                ax2.imshow(np.rot90(scan[:, :, z], k=1), cmap='gray', origin='lower')
                gt_mask = np.ma.masked_where(np.rot90(gt[:, :, z] == 0, k=1), np.rot90(gt[:, :, z], k=1))
                ax2.imshow(gt_mask, cmap=cmap, alpha=0.4, vmin=0, vmax=13)
                ax2.text(0.05, 0.95, 'Ground Truth', transform=ax2.transAxes, **annotation_params)
                # Slice number
                fig.suptitle(f'Slice {z+1}/{z_size}', y=0.98, **annotation_params)
            except IndexError:
                continue
            # Remove axes
            for ax in [ax1, ax2]:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            # Convert to numpy array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(img)
        plt.close(fig)
        
        # Save here
        if frames:
            imageio.mimsave(
                output_path,
                frames,
                duration=duration,
                fps=10,
                subrectangles=True
            )
            print(f"Saved GIF: {output_path}")
        else:
            print(f"No valid frames for {output_path}")
            
    except Exception as e:
        print(f"GIF creation failed: {str(e)}")

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--pretrained_dir", default="./pretrained_models/", type=str)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str)
parser.add_argument("--exp_name", default="test1", type=str)
parser.add_argument("--json_list", default="dataset_0.json", type=str)
parser.add_argument("--pretrained_model_name", default="pretrained.pt", type=str)
parser.add_argument("--feature_size", default=48, type=int)
parser.add_argument("--infer_overlap", default=0.5, type=float)
parser.add_argument("--in_channels", default=1, type=int)
parser.add_argument("--out_channels", default=14, type=int)
parser.add_argument("--a_min", default=-175.0, type=float)
parser.add_argument("--a_max", default=250.0, type=float)
parser.add_argument("--b_min", default=0.0, type=float)
parser.add_argument("--b_max", default=1.0, type=float)
parser.add_argument("--space_x", default=1.5, type=float)
parser.add_argument("--space_y", default=1.5, type=float)
parser.add_argument("--space_z", default=2.0, type=float)
parser.add_argument("--roi_x", default=96, type=int)
parser.add_argument("--roi_y", default=96, type=int)
parser.add_argument("--roi_z", default=96, type=int)
parser.add_argument("--dropout_rate", default=0.0, type=float)
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--RandFlipd_prob", default=0.2, type=float)
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float)
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float)
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float)
parser.add_argument("--spatial_dims", default=3, type=int)
parser.add_argument("--use_checkpoint", action="store_true")
parser.add_argument("--generate_gifs", action="store_true")
parser.add_argument("--gif_frames", default=100, type=int)
parser.add_argument("--gif_downsample", default=2, type=int)
parser.add_argument("--gif_duration", default=0.5, type=float)

def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    os.makedirs(output_directory, exist_ok=True)
    
    val_loader = get_loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = SwinUNETR(
        img_size=96,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
    )
    
    # Load weights
    model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, args.pretrained_model_name))["state_dict"])
    tta_model = SwinUNETRWithTTA(model).to(device)
    
    with torch.no_grad():
        dice_list_case = []
        for batch in val_loader:
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].cpu().numpy()[0, 0]
            img_name = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])
            
            # Inference with TTA
            tta_model.eval()
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, tta_model,
                overlap=args.infer_overlap, mode="gaussian"
            )
            
            # TTA optimization
            if tta_model.tta_optimizer:
                tta_model.train()
                for tta_step in range(tta_model.num_tta_passes):
                    tta_model.tta_optimizer.zero_grad()
                    outputs = sliding_window_inference(
                        val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, tta_model,
                        overlap=args.infer_overlap, mode="gaussian"
                    )
                    probs = F.softmax(outputs, dim=1)
                    entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=1))
                    entropy_loss.backward()
                    tta_model.tta_optimizer.step()
            
            # Final prediction
            tta_model.eval()
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, tta_model,
                overlap=args.infer_overlap, mode="gaussian"
            )
            
            # Process outputs
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()[0]
            val_outputs = np.argmax(val_outputs, axis=0)
            val_outputs = postprocess_segmentation(val_outputs)  # Remove artifacts
            
            # Resample to original label space
            _, _, h, w, d = batch["label"].shape
            val_outputs = resample_3d(val_outputs, (h, w, d), mode='nearest')
            
            # Save NIFTI
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            nib.save(nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                     os.path.join(output_directory, img_name))
            
            # Generate GIF
            if args.generate_gifs:
                try:
                    original_scan = batch["image"][0,0].cpu().numpy()
                    original_scan = resample_3d(original_scan, (h, w, d), mode='linear')
                    
                    gif_dir = os.path.join(output_directory, "gifs")
                    os.makedirs(gif_dir, exist_ok=True)
                    gif_path = os.path.join(gif_dir, f"{os.path.splitext(img_name)[0]}_comparison.gif")
                    
                    create_comparison_gif(
                        original_scan,
                        val_outputs,
                        val_labels,
                        gif_path,
                        num_frames=args.gif_frames,
                        downsample=args.gif_downsample,
                        duration=args.gif_duration
                    )
                except Exception as e:
                    print(f"GIF failed for {img_name}: {str(e)}")
            
            # Calculate Dice
            dice_scores = [dice(val_outputs == i, val_labels == i) for i in range(1, 14)]
            mean_dice = np.mean(dice_scores)
            dice_list_case.appenpd(mean_dice)
            print(f"{img_name} Mean Dice: {mean_dice:.4f}")
        
        print(f"\nOverall Mean Dice: {np.mean(dice_list_case):.4f}")

if __name__ == "__main__":
    main()
