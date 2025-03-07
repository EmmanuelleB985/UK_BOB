# UKBOB: One Billion MRI Masks for Generalizable 3D Medical Image Segmentation

Welcome to our UKBOB repository. Please see below installation of dependencies and datasets. 
We provide training and inference code for reproducibility. 


# Installing Dependencies
Create the conda environment and activate it
```
conda env create -f environment.yml
conda activate swin_bob
```

# Pre-trained Models

We will provide pre-trained weights for Swin-UNETR backbone trained on more than 50k 3D MRI with filtered labels from the UK Biobank. 

# Datasets

The following datasets were used for pre-training.


# Training

To pre-train a `Swin-B0B` with a single gpu:

```bash
python main.py

``
To evaluate `Swin-BoB` run:

```bash
python inference.py
```

To evaluate `Swin-BoB` with ETTA, run 

```bash
python etta.py
```

# License
MIT License.


# Citations
This work is based on Swin-UNetr and InTEnt. We thank the authors of these works, please consider citing them:

```

@article{hatamizadeh2022swin,
  title={Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images},
  author={Hatamizadeh, Ali and Nath, Vishwesh and Tang, Yucheng and Yang, Dong and Roth, Holger and Xu, Daguang},
  journal={arXiv preprint arXiv:2201.01266},
  year={2022}
}

@inproceedings{Dong2024MedicalIS,
  title={Medical Image Segmentation with InTEnt: Integrated Entropy Weighting for Single Image Test-Time Adaptation},
  author={Haoyu Dong and N. Konz and Han Gu and Maciej A. Mazurowski},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267682146}
}
```

# References

[1]: Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. and Xu, D., 2022. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv preprint arXiv:2201.01266.

[2]: Dong, H., Konz, N., Gu, H., & Mazurowski, M.A. (2024). Medical Image Segmentation with InTEnt: Integrated Entropy Weighting for Single Image Test-Time Adaptation. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 5046-5055.


