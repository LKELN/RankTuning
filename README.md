# Ranktuning
This is the official repository for the paper "Ranktuning: Improving Visual Plcae Recognition Preformance with Little Cost".

## Getting Started

This repo follows the framework of [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) for training and evaluation. You can refer to [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) to prepare test datasets.

The test dataset should be organized in a directory tree as such:

```
├── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```

Before training, you should download the pre-trained foundation model DINOv2(ViT-B/14) [HERE](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) and SALAD [HERE](https://github.com/serizba/salad). 

## Train
To train the model on MSLS
```
python3 train.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=MSLS --foundation_model_path=/path/to/pre-trained/dinov2_vitb14_pretrain.pth --resume=/path/to/pre-trained/SALAD.pth --negs_num_per_query=4 --pos_num_per_query=4
```

Further finetuning on Pitts30k
```
python3 train.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=Pitts30k --foundation_model_path=/path/to/pre-trained/dinov2_vitb14_pretrain.pth --resume=/path/to/resume/RankTuning_msls.pth --negs_num_per_query=2 --pos_num_per_query=2
```


## Test

To evaluate the trained model on Pitts30k/MSLS:

```
python3 eval.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=Pitts30k --resume=/path/to/resume/RankTuning_MSLS.pth or RankTuning_Pitts30k.pth
```

[SALAD](https://github.com/serizba/salad)

[Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)

[DINOv2](https://github.com/facebookresearch/dinov2)

