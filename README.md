 ---

<div align="center"> 

# Iterative Loop Combining Active Learning and  <br>Semi-Supervised Learning for Domain <br> Adaptive Semantic Segmentation
by [Licong Guan](https://licongguan.github.io/), Xue Yuan

[![Paper](http://img.shields.io/badge/paper-arxiv.2301.13361-B31B1B.svg)](https://arxiv.org/abs/2301.13361)

</div>

<!-- <div align="center"> <img width="70%" src="./img/Performance_comparison.png"></div> -->

This repository provides the official code for the paper [Iterative Loop Combining Active Learning and  Semi-Supervised Learning for Domain  Adaptive Semantic Segmentation](https://arxiv.org/abs/2301.13361).


> **Abstract** 
> Semantic segmentation is an important technique
for environment perception in intelligent transportation systems.
> With the rapid development of convolutional neural networks
(CNNs), road scene analysis can usually achieve satisfactory
results in the source domain. 
> However, guaranteeing good generalization
to different target domain scenarios remains a significant
challenge. Recently, semi-supervised learning and active
learning have been proposed to alleviate this problem. 
> Semisupervised learning can improve model accuracy with massive
unlabeled data, but some pseudo labels containing noise would
be generated with limited or imbalanced training data. 
> And there will be suboptimal models if human guidance is absent.
Active learning can select more effective data to intervene, while
the model accuracy can not be improved because the massive
unlabeled data are not used. 
> And the probability of querying suboptimal
samples will increase when the domain difference is too
large, increasing annotation cost. 
> This paper proposes an iterative loop method Combining Active learning and Semi-Supervised learning (CASS) for domain adaptive semantic segmentation. 
> The method first uses semi-supervised to learn massive unlabeled
data to improve model accuracy and provide more accurate
selection models for active learning. 
> Secondly, combined with the predictive uncertainty sample selection strategy of active learning, manual intervention is used to correct the pseudo-labels.
> Finally, flexible iterative loops achieve the best performance with minimal labeling cost.
>Extensive experiments show that our method establishes state-of-the-art performance on tasks of GTAV→Cityscapes, SYNTHIA→Cityscapes, improving by 4.9% mIoU and 5.2% mIoU, compared to the previous best method, respectively. 

![image](./img/pipeline.png)

<!-- ![image](./img/Performance_comparison.png) -->

For more information on CASS, please check our **[[Paper](https://arxiv.org/pdf/2301.13361.pdf)]**.

## Usage

### Prerequisites
- Python 3.6.9
- Pytorch 1.8.1
- torchvision 0.9.1

Step-by-step installation

```bash
git clone https://github.com/licongguan/CASS.git && cd CASS
conda create -n cass python=3.6.9
conda activate cass
pip install -r requirements.txt
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

### Data Preparation

- Download [The Cityscapes Dataset](https://www.cityscapes-dataset.com/), [The GTAV Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/), and [The SYNTHIA Dataset](https://synthia-dataset.net/)

**First, the data folder should be structured as follows:**

```
├── datasets/
│   ├── cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── gtav/
|   |   ├── images/
|   |   ├── labels/
│   └──	synthia/
|   |   ├── RGB/
|   |   ├── LABELS/
│   └──	
```

**Second, generate ```_labelTrainIds.png``` for the cityscapes dataset:**

```bash
pip install cityscpaesscripts
pip install cityscpaesscripts[gui]
export CITYSCAPES_DATASET='/path_to_cityscapes'
csCreateTrainIdLabelImgs
```

**Final, rename the gtav and synthia files for  by running:**

```bash
python cass/datasets/rename.py
```

### Prepare Pretrained Backbone

Before training, please download ResNet101 pretrained on ImageNet-1K from one of the following:
  - [Google Drive](https://drive.google.com/file/d/1fkOA3WSM4FjqBw3EtQIaI5FgEtPMAky9/view?usp=share_link)
  - [Baidu Drive](https://pan.baidu.com/s/1qWYbpi29U-UOiElFNtWy8g) Fetch Code: cass

After that, modify ```model_urls``` in ```cass/models/resnet.py``` to ```</path/to/resnet101.pth>```


###  Model Zoo
#### GTAV to Cityscapes

We have put our model checkpoints here [[Google Drive](https://drive.google.com/drive/folders/1M674BpVWY7laWAdDkoHvVRh-36-zO1nM?usp=share_link)] [[百度网盘](https://pan.baidu.com/s/1s6KA33yA1Z3qgc9o1XffAQ)] (提取码`cass`).

| Method                      | Net | budget | mIoU | Chekpoint | Where in Our [Paper](https://arxiv.org/abs/2301.13361) |
| --------------------------- | --------- | --------- | --------- | --------- | ----------- |
| CASS                     | V3+     | 1%     | 70.0     | [Google Drive](https://drive.google.com/file/d/12GJptsIQbbtNqZS8sryI_BI0ToGz84Qc/view?usp=share_link)/[BaiDu](https://pan.baidu.com/s/1dhW8gk4G3AYOuLab2x8kRg)     | Table1     |
| CASS                     | V3+     | 2.2%  | 75.0     | [Google Drive](https://drive.google.com/file/d/1E_HxxlJseg2F_aaJG4yeb3mXgI8qQRrN/view?usp=share_link)/[BaiDu](https://pan.baidu.com/s/1b7G5k064EYoA3HaMdxKnAQ)     | Table1     |
| CASS                     | V3+     | 5.0%  | 76.1     | [Google Drive](https://drive.google.com/file/d/19Aooa71riTeYA70-ZwSjqY8DUeb2GKnl/view?usp=share_link)/[BaiDu](https://pan.baidu.com/s/15HCyAbIsP6Rhpw_mhON71Q)     | Table1     |


#### SYNTHIA to Cityscapes

| Method                      | Net | budget | mIoU | Chekpoint | Where in Our [Paper](https://arxiv.org/abs/2301.13361) |
| --------------------------- | --------- | --------- | --------- | --------- | ----------- |
| CASS                     | V3+     | 1%     | 73.2     | [Google Drive](https://drive.google.com/file/d/1JGQr-yt4R8jOPLNipQ-zDV2Rvi7uvnEq/view?usp=share_link)/[BaiDu](https://pan.baidu.com/s/1v7LrMxvAY9yt2YXMosu6Nw)     | Table2     |
| CASS                     | V3+     | 2.2%  | 76.0     | [Google Drive](https://drive.google.com/file/d/1XGOhS8wy_gw3fcZUKzfSc-TJNAzUgslq/view?usp=share_link)/[BaiDu](https://pan.baidu.com/s/1Xe07TxGuE4qR7Rh8kioKlg)     | Table2     |
| CASS                     | V3+     | 5.0%  | 76.6     | [Google Drive](https://drive.google.com/file/d/1bLYuFnOAnmsv-_fbMIKymFhhS4kO27Dp/view?usp=share_link)/[BaiDu](https://pan.baidu.com/s/17oTkh1srH7hb1xv8fy3tDQ)     | Table2     |


### CASS Training

**We provide the training scripts  using  Multiple GPU.**

```bash
# training for GTAV to Cityscapes
# use GTAV 2000 labeled images and Cityscpaes 30(1%) labeled images
cd  experiments/gtav2cityscapes/1.0%
# use torch.distributed.launch
sh train.sh <num_gpu> <port>

```

### CASS Testing

```bash
sh eval.sh
```

## Acknowledgement

This project is based on the following open-source projects: [U<sup>2</sup>PL](https://github.com/Haochen-Wang409/U2PL) and [RIPU](https://github.com/BIT-DA/RIPU). We thank their authors for making the source code publically available.


## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@article{guan2023iterative,
  title={Iterative Loop Combining Active Learning and  Semi-Supervised Learning for Domain  Adaptive Semantic Segmentation},
  author={Guan, Licong and Yuan, Xue},
  journal={arXiv preprint arXiv:2301.13361},
  year={2023}
}
```

## Contact

If you have any problem about our code, feel free to contact

- [lcguan941@bjtu.edu.cn](lcguan941@bjtu.edu.cn)

or describe your problem in Issues.

