import argparse
import os
from functools import partial
import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

parser = argparse.ArgumentParser(description="Swin UNETR")
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=1, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="model_final.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.6, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument(
    "--pretrained_dir",
    default="./pretrained_models/",
    type=str,
    help="pretrained checkpoint directory",
)


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./BRATS23/outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    test_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = SwinUNETR(
        img_size=128,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch["image"].cuda()
            
            # Check which keys are available in the batch
            if "image_meta_dict" in batch:
                # Use original metadata structure
                affine = batch["image_meta_dict"]["original_affine"][0].numpy()
                num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")[1]
            else:
                # Fallback if metadata isn't available
                print("Warning: No metadata found in batch. Using default values.")
                affine = np.eye(4)  # Default identity affine
                num = str(i).zfill(3)  # Use batch index as fallback
                
            img_name = "BraTS2023_" + num + ".nii.gz"
            print("Inference on case {}".format(img_name))
            
            # Run model inference
            prob = torch.sigmoid(model_inferer_test(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4
            
            # Save the segmentation prediction
            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))
            
            # Save the original image
            orig_image = image[0].detach().cpu().numpy()
            for ch_idx, modality in enumerate(["t1", "t1ce", "t2", "flair"]):
                # Save each modality separately
                nib.save(
                    nib.Nifti1Image(orig_image[ch_idx].astype(np.float32), affine),
                    os.path.join(output_directory, f"BraTS2023_{num}_{modality}.nii.gz")
                )
            
            # Save the ground truth if available
            if "label" in batch:
                gt = batch["label"][0].detach().cpu().numpy()
                gt_out = np.zeros((gt.shape[1], gt.shape[2], gt.shape[3]))
                # Convert multi-channel back to single label map
                # Assuming channels are [ET, TC, WT] or similar BraTS encoding
                gt_out[gt[1] == 1] = 2  # TC
                gt_out[gt[0] == 1] = 1  # ET
                gt_out[gt[2] == 1] = 4  # WT
                
                nib.save(
                    nib.Nifti1Image(gt_out.astype(np.uint8), affine),
                    os.path.join(output_directory, f"BraTS2023_{num}_gt.nii.gz")
                )
        
        print("Finished inference!")

if __name__ == "__main__":
    main()
