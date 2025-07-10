## News
- [x] [2025.7.10] **FSVLM model parameters have been released** [Model](https://pan.baidu.com/s/1sX8mhJ3NftS9w1lmFczKvg?pwd=8888). 
- [x] [2025.5.15] 《FarmSeg_VLM: A farmland remote sensing image segmentation method considering vision-language alignment》 received by **ISPRS Journal of Photogrammetry and Remote Sensing** [Paper](https://doi.org/10.1016/j.isprsjprs.2025.05.010). The associated code will be made publicly available upon completion of organization and finalization.
- [x] [2025.5.9] 《FarmSeg_VLM: A farmland remote sensing image segmentation method considering vision-language alignment》 received by **ISPRS Journal of Photogrammetry and Remote Sensing** and is currently in the proof stage. The associated code will be made publicly available upon completion of organization and finalization.
- [x] [2025.4.25] 《A large-scale image-text dataset benchmark for farmland segmentation》 received as preprint by **Earth System Science Data** [Paper](https://essd.copernicus.org/preprints/essd-2025-184/essd-2025-184.pdf) [Datasets](https://doi.org/10.5281/zenodo.15099885)
- [x] [2025.3] 《FSVLM: A Vision-Language Model for Remote Sensing Farmland Segmentation》 received by **IEEE TGRS** [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10851315) [code](https://github.com/WithoutOcean/FSVLM-A-Vision-Language-Model-for-Remote-Sensing-Farmland-Segmentation).


**FSVLM: A Vision-Language Model for Remote Sensing Farmland Segmentation[[Paper](https://ieeexplore.ieee.org/document/10851315)]** <br />
[Haiyang Wu](https://scholar.google.com/citations?user=VzEne7oAAAAJ&hl=zh-CN),
[Zhuofei Du],
[Dandan Zhong],
[Yuze Wang](https://scholar.google.com/citations?hl=zh-CN&user=XptRvWUAAAAJ),
[Chao Tao](https://scholar.google.com/citations?hl=zh-CN&user=g0Dvu7IAAAAJ)<br />

## Abstract
Existing visual deep learning paradigms, which are based on labels, struggle to capture the intricate interrelationships between farmland and its surrounding environment and fail to account for temporal variations associated with the phenological cycle. These limitations lead to omissions and confusion in the recognition process, greatly impacting the accuracy and efficiency of farmland recognition. Language can accurately depict the spatial attributes of farmland, profoundly express the unique phenological landscapes of farmland that change with seasons and growth stages, and express the intricate interactions between these changes and environmental factors. This capability can address the deficiencies of label-based visual deep learning in understanding the complex features of farmland. This study explored, for the first time, the application of language-guided vision-language models (VLMs) for farmland segmentation. First, as current VLM research lacks a dedicated farmland image text (FIT)pair dataset, this study constructed an FIT dataset in two steps. Step 1, designed a semi-automatic text description annotation framework for farmland images based on 12 key factors influencing farmland segmentation. Step 2, used the framework to construct the FIT dataset. Then, a VLM for farmland segmentation (FSVLM) was designed by combining a semantic segmentation model with a multimodal large language model (LLM). Comparative experiments demonstrated that the proposed method outperforms existing farmland segmentation methods in both generalization and segmentation accuracy. In addition, a series of ablation experiments were conducted to examine the impacts of language descriptions of different semantic levels on the model’s farmland information extraction performance.For more details, please refer to the [paper](https://ieeexplore.ieee.org/document/10851315).


## Installation
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Training
### Training Data Preparation

1. farmland image-text datasets: [FarmSeg-VL](https://doi.org/10.5281/zenodo.15099885)

2. segmentation datasets: [LoveDA](https://github.com/Junjue-Wang/LoveDA?tab=readme-ov-file)

Download them from the above links, and organize them as follows.
```
├── dataset
│   ├── train
│   │   ├── img
│   │   └── lbl
|   │   └── json 
│   ├── test
│   │   ├── img
│   │   └── lbl
|   │   └── json 
│   ├── val
│   │   ├── img
│   │   └── lbl
|   │   └── json 
```

Replace the data path with the root directory of your data.
```
    parser.add_argument("--val_dataset", default="dyour data root path", type=str)
    parser.add_argument("--dataset_dir", default="your data root path", type=str)
```

### Pre-trained weights

#### LLaVA
We can directly use the LLaVA full weights `liuhaotian/llava-llama-2-13b-chat-lightning-preview`.

#### SAM ViT-H weights
Download SAM ViT-H pre-trained weights from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

### Training
```
python train_ds.py 
```


### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```

For example:
```
CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version="./LLaVA/LLaVA-Lightning-7B-v1-1" \
  --weight="FSVLM-7b/pytorch_model.bin" \
  --save_path="./FSVLM-7B"
```

## Inference 
```
python test.py --version='your weight'
```

## Citation 
If you find this project useful in your research, please consider citing:

```
@ARTICLE{10851315,
  author={Wu, Haiyang and Du, Zhuofei and Zhong, Dandan and Wang, Yuze and Tao, Chao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={FSVLM: A Vision-Language Model for Remote Sensing Farmland Segmentation}, 
  year={2025},
  volume={63},
  number={},
  pages={1-13},
  keywords={Image segmentation;Remote sensing;Accuracy;Annotations;Visualization;Deep learning;Crops;Semantics;Semantic segmentation;Data models;Deep learning;farmland segmentation;image-text dataset;semantic segmentation;vision-language model (VLM)},
  doi={10.1109/TGRS.2025.3532960}}

@article{WU2025423,
title = {FarmSeg_VLM: A farmland remote sensing image segmentation method considering vision-language alignment},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {225},
pages = {423-439},
year = {2025},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.05.010},
url = {https://www.sciencedirect.com/science/article/pii/S0924271625001911},
author = {Haiyang Wu and Weiliang Mu and Dandan Zhong and Zhuofei Du and Haifeng Li and Chao Tao},
}

@Article{essd-2025-184,
AUTHOR = {Tao, C. and Zhong, D. and Mu, W. and Du, Z. and Wu, H.},
TITLE = {A large-scale image-text dataset benchmark for farmland segmentation},
JOURNAL = {Earth System Science Data Discussions},
VOLUME = {2025},
YEAR = {2025},
PAGES = {1--29},
URL = {https://essd.copernicus.org/preprints/essd-2025-184/},
DOI = {10.5194/essd-2025-184}
}
```

## Acknowledgement
-  This work is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA), [SAM](https://github.com/facebookresearch/segment-anything) and [LISA](https://github.com/dvlab-research/LISA). 
