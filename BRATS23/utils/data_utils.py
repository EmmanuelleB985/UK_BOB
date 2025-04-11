import json
import math
import os
import numpy as np
import torch
from monai import data, transforms
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Randomizable, RandomizableTrait, RandomizableTransform, Transform
import time 
import random 

def worker_init_fn(worker_id):
    """
    Worker initialization function to ensure reproducible data loading.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class ConvertToMultiChannelBasedOnBratsClasses(Transform):
    """
    Convert labels to multi channels based on brats23 classes:
    label 1 is the necrotic tumor core (NCR)
    label 2 is the peritumoral edema (ED)
    label 3 is the enhancing tumor (ET)
    The possible classes are:
    - TC (Tumor core): NCR + ET (labels 1+3)
    - WT (Whole tumor): NCR + ED + ET (labels 1+2+3)
    - ET (Enhancing tumor): just ET (label 3)
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    def __init__(self, keys="label"):
        self.keys = keys
    
    def __call__(self, data):
        d = dict(data)
        # Process the image corresponding to the specified key
        img = d[self.keys]
        
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        
        # Create the multi-channel output based on BraTS 2023 labels
        # TC: tumor core (labels 1+3)
        # WT: whole tumor (labels 1+2+3)
        # ET: enhancing tumor (label 3)
        result = [
            (img == 1) | (img == 3),  # TC: Tumor Core (NCR + ET)
            (img == 1) | (img == 2) | (img == 3),  # WT: Whole Tumor (NCR + ED + ET)
            img == 3  # ET: Enhancing Tumor
        ]
        
        d[self.keys] = torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)
        return d


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)
    
    # Custom transform to preserve metadata
    class SaveMetadataTransform(transforms.MapTransform):
        def __init__(self, keys):
            super().__init__(keys)
        
        def __call__(self, data):
            # Copy metadata we need before it's potentially lost
            if "image_meta_dict" in data:
                data["meta_dict"] = {
                    "filename_or_obj": data["image_meta_dict"]["filename_or_obj"],
                    "original_affine": data["image_meta_dict"]["original_affine"]
                }
            return data
    
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            SaveMetadataTransform(keys=["image"]),  # Save metadata immediately after loading
            ConvertToMultiChannelBasedOnBratsClasses(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            SaveMetadataTransform(keys=["image"]),  # Save metadata immediately after loading
            ConvertToMultiChannelBasedOnBratsClasses(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            SaveMetadataTransform(keys=["image"]),  # Save metadata immediately after loading
            ConvertToMultiChannelBasedOnBratsClasses(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    
    if args.test_mode:
        val_ds = data.Dataset(data=validation_files, transform=test_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True if args.workers > 0 else False,
            worker_init_fn=worker_init_fn,
        )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]
    
    return loader
