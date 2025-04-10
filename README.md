# UKBOB: One Billion MRI Masks for Generalizable 3D Medical Image Segmentation

Welcome to our UKBOB repository. Please see below installation of dependencies and datasets. 
We provide training and inference code for reproducibility. 

# Installing Dependencies
Create the conda environment and activate it
```
conda env create -f environment.yml
conda activate swin_bob
```

# Datasets and Preprocessing 

We use the initial segmentation labels from [2] that we filter with our custom _Specialized Organ Label Filter (SOLF)_.

```bash
python filtering/organ_filtering.py
```

For out-of-domain evaluation, we use the **BTCV** and **BRATS23** datasets where we preserve the train-val-test splits. 
Please download these public datasets and associated json files for [BRATS](https://www.synapse.org/Synapse:syn51156910/wiki/627000) and [BTCV](https://www.synapse.org/Synapse:syn3193805/wiki/217789).


# Pre-trained Models

We will provide pre-trained weights for Swin-UNETR backbone trained on more than 50k 3D MRI with filtered labels from the UK Biobank. 
In the meantime, we provide weights for our segmentation model with ETTA on **BTCV** [here]() and **BRATS** [here]().
Please download the weights and follow the instructions below to run inference and visualise the outputs.


# Training

To pre-train a `Swin-B0B` with a single gpu:

```bash
python main.py

```
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

@article{graf2024totalvibesegmentator,
  title={TotalVibeSegmentator:  Full Body MRI Segmentation for the NAKO and UK Biobank },
  author={Graf, Robert and Platzek, Paul-S{\"o}ren and Riedel, Evamaria Olga and Ramsch{\"u}tz, Constanze and Starck, Sophie and M{\"o}ller, Hendrik Kristian and Atad, Matan and V{\"o}lzke, Henry and B{\"u}low, Robin and Schmidt, Carsten Oliver and others},
  journal={arXiv preprint arXiv:2406.00125},
  year={2024}
}

@inproceedings{Dong2024MedicalIS,
  title={Medical Image Segmentation with InTEnt: Integrated Entropy Weighting for Single Image Test-Time Adaptation},
  author={Haoyu Dong and N. Konz and Han Gu and Maciej A. Mazurowski},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267682146}
}
```

# References

[1]: Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. and Xu, D., (2022). Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv preprint arXiv:2201.01266.

[2]: Graf, R., Platzek, P., Riedel, E.O., Ramschutz, C., Starck, S., Moller, H.K., Atad, M., Vőlzke, H., Bulow, R., Schmidt, C.O., Rudebusch, J., Jung, M., Reisert, M., Weiss, J., Loffler, M., Bamberg, F., Wiestler, B., Paetzold, J.C., Rueckert, D., & Kirschke, J.S. (2024). TotalVibeSegmentator: Full Body MRI Segmentation for the NAKO and UK Biobank.

[3]: Dong, H., Konz, N., Gu, H., & Mazurowski, M.A. (2024). Medical Image Segmentation with InTEnt: Integrated Entropy Weighting for Single Image Test-Time Adaptation. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 5046-5055.


