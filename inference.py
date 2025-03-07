import os
import copy
import json
import shutil
import matplotlib.pyplot as plt
from monai.transforms import (
    AddChanneld,
    Compose,
    CenterScaleCropd,
    LoadImaged,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    EnsureChannelFirstd,
    NormalizeIntensityd
    Resized,
    Spacingd,
    EnsureTyped,
)
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from PIL import Image
import nibabel as nib
import numpy as np
import torch
import warnings
import argparse
import pydicom


warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task_form', type=str, required=True, help="task form")
parser.add_argument('-o', '--model_out_channels', type=int, required=True, help="model out channels")
parser.add_argument('-w', '--model_weights_path', type=str, required=True, help="model weights path")
parser.add_argument('-d', '--data_path', type=str, required=True, help="data path")
parser.add_argument('-c', '--crop', type=str, required=True, help="crop")
parser.add_argument('-p', '--prediction_path', type=str, required=True, help="prediction path")
parser.add_argument('-v', '--visualization_path', type=str, required=True, help="visualization path")
args = parser.parse_args()

task_form = args.task_form
model_out_channels = args.model_out_channels
model_weights_path = args.model_weights_path
data_path = args.data_path
crop = args.crop
prediction_path = args.prediction_path
visualization_path = args.visualization_path

def nifti2dicom(array):
    """
    array: input slice
    file_dir: path to saved slices
    index: slice index 
    """
    dicom_file = pydicom.dcmread(dicom_temp_path)
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.PixelData = arr.tobytes()
    return dicom_file


test_transforms = {
    '3D Segmentation BTCV': [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(
            keys=["image"],
            spatial_size=(256, 256, 256),
            mode=("trilinear"),
        ),
        EnsureTyped(keys=["image"], device=device, track_meta=False),
    ],
    '3D Segmentation AMOS': [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(
            keys=["image"],
            spatial_size=(256, 256, 256),
            mode=("trilinear"),
        ),
        EnsureTyped(keys=["image"], device=device, track_meta=True),
    ],
}
test_transforms = test_transforms[task_form]
if crop == 'On':
    test_transforms.insert(2, 
        CenterScaleCropd(
            keys=["image"],
            roi_scale=(0.66,0.66,1.0),
        ))
data = Compose(test_transforms)({
    "image": data_path
})

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=model_out_channels,
    feature_size=48,
    use_checkpoint=True,
)
print('Loading model')
model.load_state_dict(torch.load(model_weights_path))
model.to(device)
model.eval()

with torch.no_grad():
    print('Prediction')
    test_inputs = torch.unsqueeze(data["image"], 1).to(device)
    test_outputs = sliding_window_inference(
        test_inputs, (96, 96, 96), 4, model, overlap=0.8
    )

    test_inputs = test_inputs.cpu().numpy()
    test_outputs = torch.argmax(test_outputs, dim=1).detach().cpu().numpy()

slice_rate = 0.5
slice_num = int(test_inputs.shape[-1]*slice_rate)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(test_inputs[0, 0, :, :, slice_num], cmap="gray")
ax1.set_title('Image')
ax2.imshow(test_outputs[0, :, :, slice_num])
ax2.set_title(f'Predict')
plt.savefig(visualization_path, bbox_inches='tight')


test_outputs = np.round(test_outputs[0]/model_out_channels*255, 0)
print('Label classes:', np.unique(test_outputs) )


nii_path = os.path.join(prediction_path, 'mask.nii.gz')
test_outputs = nib.Nifti1Image(test_outputs, affine=np.eye(4))
nib.save(test_outputs, nii_path)


dicom_temp_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dcmimage.dcm')

image_path = os.path.join(prediction_path, 'image')
dicom_path = os.path.join(prediction_path, 'dicom')
os.makedirs(image_path)
os.makedirs(dicom_path)
nifti_array = test_outputs.get_fdata()
for i in range(test_outputs.shape[-1]):
    slice = nifti_array[:, :, i]
    img = Image.fromarray(np.uint8(slice))
    img.save(os.path.join(prediction_path, 'image', f'{str(i).zfill(5)}.jpg'))
    dicom_file = nifti2dicom(nifti_array[:, :, i])
    dicom_file.save_as(os.path.join(prediction_path, 'dicom', f'{str(i).zfill(5)}.dcm'))