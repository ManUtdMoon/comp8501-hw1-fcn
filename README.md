# COMP8501-HW1-FCN: Fully Convolutional Networks based on ResNet
Dongjie Yu (3030102216)

## Project Structure
```
├── config (Shared and specific hyper-parameters, managed by hydra)
│   ├── model (Specific, mainly model names)
│   │   ├── ....yaml
│   └── train.yaml (Shared)
├── data (training set, training logs, checkpoints, plots)
│   ├── outputs
│   │   ├── ...
│   └── VOCdevkit (training data)
│       └── VOC2012
├── dataset.py (load dataset, define transformations)
├── models.py (FCN models based on ResNet)
├── model_utils.py (Basic blocks for models)
├── plot_curves.py (plot training curves)
├── predict.py (Make segmentation predictions on test set)
├── predict_reference.py (load torchvision model to make predictions)
├── README.md
├── train.py (main training script)
└── utils.py (helper functions and variables such as logger, metrics, etc.)
```

## Downloading Dataset
- Create a sub-directory called `data` under the root directory.
- Download the VOC2012 dataset from [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and extract it under the `data` directory. Then the directory structure is as follows:
```
└── data
    └── VOCdevkit
        └── VOC2012
            ├── Annotations
            ├── ImageSets
            ├── JPEGImages
            ├── SegmentationClass
            └── SegmentationObject
```

## Installing environment
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```bash
mamba env create -f conda_env.yaml
```

but you can use conda as well: 
```bash
conda env create -f conda_env.yaml
```

Then activate the environment:
```bash
conda activate fcn
```

## Train
Checkout the model your like in `config/model` and run the following command to start training:
```bash
python train.py model=resnet50_pretrained_fcn16s
```
Here I recommend using ImageNet pretrained ResNet50 as the backbone to accelerate training and get better results. Therefore, only ones with `pretrained` in their names are recommended.

## Training Results
After training, the checkpoints, training logs, will be saved in the `data/outputs` directory. You can read the csv files for an overview of the training process.

### Training curves
Run the following command to visualize the **training curves** of my results:
```bash
python plot_curves.py -t data/outputs/2024.03.01/00.57.43_train_fcn_seg_resnet50_pretrained_fcn16s
```
which will be saved to the `./plot` sub-directory.

### Segmentation Inference
Run the following command to make **inferences** on the first 40 images in test set:
```bash
python predict.py -c data/outputs/2024.03.01/00.57.43_train_fcn_seg_resnet50_pretrained_fcn16s -d cpu
```
which will be saved to the `./predict` sub-directory.

Then the whole training results directory structure is as follows:
```
└── 2024.03.01
    └── 00.57.43_train_fcn_seg_resnet50_pretrained_fcn16s
        ├── checkpoints
        │   ├── .ckpt
        ├── plot
        │   ├── .png
        ├── predict
        │   └── epoch=0021-acc=0.7765
        │       ├── .png
        ├── train.log
        ├── train_log.csv
        └── val_log.csv
```

## Acknowledgements
Following repos are referred to:
- [DevikalyanDas](https://github.com/DevikalyanDas/Semantic-Segmentation-of-pascal-voc-dataset-using-Pytorch)
- [wkentaro/pytorch-fcn](https://github.com/wkentaro/pytorch-fcn)
- [pytorch/vision](https://github.com/pytorch/vision/tree/main/references/segmentation)
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)

References include but are not limited to:
- Borrowed functions, classes, and scripts from the above repos
- The overall project structure.