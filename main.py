import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirst,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    RandRotate90d,
    Resize,
    Resized,
    EnsureTyped,
    Flipd,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    SpatialPad,
    ResizeWithPadOrCrop,
    SpatialPadd,
    DivisiblePadd
    
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
import torch
import warnings
import matplotlib.pyplot as plt
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.util import montage
import json
import pprint


warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dir_list = "./TotalVibeSegmentator/F/"        
imgs = []
for i in os.listdir(dir_list):
    imgs.append(i)

seg_path = dir_list
img_path = "./spinenet-ukbb-stitching/UKBB_stitched/" #path to the stitched images from UK Biobank

img_names = []
img_labels = []

for el in imgs:
    img_names.append(img_path + el.replace('.nii.gz','_in.nii.gz')) # input image in-phase
    img_labels.append(seg_path + el)
    
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_names, img_labels)]

num_samples = 2
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        DivisiblePadd(keys=["image", "label"],k=64),        
        Resized(
            keys=["image", "label"],
            spatial_size=(128, 128, 128),
            mode=("trilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
           prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(32, 32, 32),
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0),
        Resized(
            keys=["image", "label"],
            spatial_size=(96, 96, 96),
            mode=("trilinear", "nearest")),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
  
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        DivisiblePadd(keys=["image", "label"],
            k=64),
        
        Resized(
            keys=["image", "label"],
            spatial_size=(128, 128, 128),
            mode=("trilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(
            keys=["image", "label"],
            spatial_size=(96, 96, 96),
            mode=("trilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        DivisiblePadd(keys=["image", "label"],
            k=64),
        
        Resized(
            keys=["image", "label"],
            spatial_size=(128, 128, 128),
            mode=("trilinear", "nearest"),
        ),

        #ScaleIntensityRanged(
        #    keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        #),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(
            keys=["image", "label"],
            spatial_size=(96, 96, 96),
            mode=("trilinear", "nearest")),

        EnsureTyped(keys=["image"], device=device, track_meta=False),
    ]
)

dir_list = "./Output/"        
imgs = []
for i in os.listdir(dir_list):
    imgs.append(i)

seg_path = dir_list
img_path = "./spinenet-ukbb-stitching/UKBB_stitched/"

img_names = []
img_labels = []

for el in imgs:
    img_names.append(img_path + el.replace('.nii.gz','_F.nii.gz'))
    img_labels.append(seg_path + el)
    
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_names, img_labels)]
train_files, val_files = data_dicts[:int(0.9*len(data_dicts))], data_dicts[int(0.99*len(data_dicts)):]


dataset_json = {
    "labels":{
    "0": "background",
    "1": "spleen",
    "2": "kidney_right",
    "3": "kidney_left",
    "4": "gallbladder",
    "5": "liver",
    "6": "stomach",
    "7": "pancreas",
    "8":  "adrenal_gland_right",
    "9":  "adrenal_gland_left",
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
    },
    "tensorImageSize": "3D",
    "training": [],
    "validation": []
}

for i in range(len(train_files)):
    dataset_json["training"].append(train_files[i])
for j in range(len(val_files)):
    dataset_json["validation"].append(val_files[j])

datasets = './Swin-UNETR/data/dataset.json'

with open(datasets, 'w') as outfile:
    json.dump(dataset_json, outfile)

pprint.pprint(dataset_json)


train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_num=16,
    cache_rate=1.0,
    num_workers=1,
)
train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=2, shuffle=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=4, cache_rate=1.0, num_workers=0)#6
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)


set_track_meta(False)

case_num = 1
slice_num = 64

img = val_ds[case_num]["image"]
label = val_ds[case_num]["label"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img[0, :, :, slice_num].detach().cpu(), cmap="gray")
ax1.set_title(f'Image shape: {img.shape}')
ax2.imshow(label[0, :, :, slice_num].detach().cpu())
ax2.set_title(f'Label shape: {label.shape}')
plt.show()
plt.savefig("./results/test.png")


model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=73, #72 classes + background
    feature_size=48,
    use_checkpoint=True,
).to(device)


weight = torch.load("./data/model_swin.pt")
model.load_from(weights=weight)


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 1.0)
            )
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), "/work/emmanuelle/Swin-UNETR/data/best_metric_model.pth")
                print(f"\nModel Saved. Best Average Dice: {dice_val_best} Current Average Dice: {dice_val}")
            else:
                print(f"\nModel Not Saved. Best Average Dice: {dice_val_best} Current Average Dice: {dice_val}")
        global_step += 1
    return global_step, dice_val_best, global_step_best


torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()

max_iterations = 30000
eval_num = 500
post_label = AsDiscrete(to_onehot=73) # class n (72 classes + background)
post_pred = AsDiscrete(argmax=True, to_onehot=73) # class n (#72 classes + background)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
model.load_state_dict(torch.load("./data/best_metric_model.pth"))

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.show()
plt.savefig("./results/loss.png")

#eval 
case_num = 1
slice_num = 50

model.load_state_dict(torch.load("./data/best_metric_model.pth"))
model.eval()
with torch.no_grad():
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    val_inputs = torch.unsqueeze(img, 1).cuda()
    val_labels = torch.unsqueeze(label, 1).cuda()
    val_outputs = sliding_window_inference(
        val_inputs, (96, 96, 96), 4, model, overlap=0.8
    )
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_num], cmap="gray")
    ax1.set_title('Image')
    ax2.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_num])
    ax2.set_title(f'Label')
    ax3.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_num])
    ax3.set_title(f'Predict')
    plt.show()
    plt.savefig("./results/prediction.png")