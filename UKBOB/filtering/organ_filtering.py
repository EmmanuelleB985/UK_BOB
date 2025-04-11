import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd

# ======== Define the dataset JSON with 72 classes ========
dataset_json = {
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
    },
    "tensorImageSize": "3D",
    "training": [],
    "validation": []
}

#--Surface Area-----
def compute_surface_area(mask):
        """
        Compute the surface area of a 3D mask from a NumPy 3D tensor.
        
        Parameters:
        - mask: A 3D NumPy array of shape (nx, ny, nz) with binary values (0 or 1),
                where 1 indicates the object and 0 indicates the background.
        
        Returns:
        - surface_area: The total surface area (number of exposed faces).
        """
        # Pad the mask with a layer of zeros on all sides to handle boundary conditions
        padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        
        # Initialize surface area
        surface_area = 0
        
        # Compute exposed faces in each direction
        # For each direction, an exposed face occurs where the voxel is 1
        # and the neighbor in that direction is 0 (background)
        
        # Negative x-direction
        surface_area += np.sum((mask == 1) & (padded_mask[:-2, 1:-1, 1:-1] == 0))
        
        # Positive x-direction
        surface_area += np.sum((mask == 1) & (padded_mask[2:, 1:-1, 1:-1] == 0))
        
        # Negative y-direction
        surface_area += np.sum((mask == 1) & (padded_mask[1:-1, :-2, 1:-1] == 0))
        
        # Positive y-direction
        surface_area += np.sum((mask == 1) & (padded_mask[1:-1, 2:, 1:-1] == 0))
        
        # Negative z-direction
        surface_area += np.sum((mask == 1) & (padded_mask[1:-1, 1:-1, :-2] == 0))
        
        # Positive z-direction
        surface_area += np.sum((mask == 1) & (padded_mask[1:-1, 1:-1, 2:] == 0))
        
        return surface_area

# ======== Specify the folder containing NIfTI files ========
input_folder = './data'
output_csv = "./all_features.csv"

# Get list of all nifti files in the folder
nifti_files = glob.glob(os.path.join(input_folder, "*.nii.gz"))

# Initialize a list to store features for all images.
# Each element is a list: [Filename, Label, Volume, Surface_Area, Eccentricity, Sphericity]
all_features = []

# Loop over each file in the folder
for nifti_file in nifti_files[:10]:  # top ten
    print(f"Processing: {nifti_file}")
    try:
        img = nib.load(nifti_file)
    except Exception as e:
        print(f"Error loading {nifti_file}: {e}")
        continue

    data = img.get_fdata()
    affine = img.affine
    voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))

    # Identify unique labels (excluding background: label 0)
    labels = np.unique(data)
    labels = labels[labels != 0]

    # Loop over each label in the current image
    for label in labels:
        # Create binary mask for the current label
        mask = data == label

        # --- Volume ---
        volume = np.sum(mask) * voxel_volume

        # --- Surface Area ---
        surface_area = compute_surface_area(mask)

        # --- Eccentricity ---
        coords = np.column_stack(np.nonzero(mask))
        coords_phys = nib.affines.apply_affine(affine, coords)
        if coords_phys.shape[0] >= 3:
            cov = np.cov(coords_phys, rowvar=False)
            eigvals, _ = np.linalg.eig(cov)
            eigvals = np.sort(eigvals)[::-1]
            eccentricity = np.sqrt(1 - eigvals[-1] / eigvals[0]) if eigvals[0] > 0 else np.nan
        else:
            eccentricity = np.nan
    
        # --- Sphericity ---
        if surface_area and surface_area > 0:
            sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface_area
        else:
            sphericity = np.nan

        base_filename = os.path.basename(nifti_file)
        all_features.append([base_filename, int(label), volume, surface_area, eccentricity, sphericity])

# Convert the list of features to a DataFrame with appropriate column names.
df_features = pd.DataFrame(all_features, 
                           columns=['Filename', 'Label', 'Volume', 'Surface_Area', 'Eccentricity', 'Sphericity'])
print("Extracted Features:")
print(df_features)

# Save the combined features to a CSV file.
df_features.to_csv(output_csv, index=False, header=True)
print(f"Features saved to: {output_csv}")
