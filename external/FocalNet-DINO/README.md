# FocalNet for Object Detection with DINO

This repo contains the code for reproducing object detection results of our [FocalNets](https://arxiv.org/abs/2203.11926). It is based on [DINO](https://github.com/IDEA-Research/DINO).

## Installation

Please follow [DINO's instruction](https://github.com/IDEA-Research/DINO) for installation.

## Training

* Train on COCO with FocalNet-L with 3 focal levels:

```
python -m torch.distributed.launch --nproc_per_node={ngpus} main.py --config_file config/DINO/DINO_4scale_focalnet_fl3.py --coco_path {coco_path} --output_dir {output_dir}
```

* Train on COCO with 5scale DINO and FocalNet-L with 4 focal levels:

```
python -m torch.distributed.launch --nproc_per_node={ngpus} main.py --config_file config/DINO/DINO_5scale_focalnet_fl4.py --coco_path {coco_path} --output_dir {output_dir}
```

## Model Zoos

FocalNet-DINO pretrained with Object365:

| Backbone | Method | Pretrained Data | COCO minival mAP (w/o tta) | Download
| :---: | :---: | :---: | :---: | :---: | 
Swin-L | DINO | Object365 | 63.1 | - |
FocalNet-L | DINO | Object365 | 63.5 | [in21k ckpt](https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384_fl4.pth)/[o365 ckpt](https://projects4jw.blob.core.windows.net/focalnet/release/detection/focalnet_large_fl4_pretrained_on_o365.pth)/[coco ckpt](https://projects4jw.blob.core.windows.net/focalnet/release/detection/focalnet_large_fl4_o365_finetuned_on_coco.pth)

## Related Links

Thanks to the authors of DINO, the DINO models trained with FocalNets as the backbones can be found here:
> **FocalNet-L + DINO**: [DINO + FocalNet-L](https://github.com/IDEA-Research/detrex/tree/main/projects/dino)

All pretrained models on imagenet-1k or imagenet-21k are provided in:

> **Focal Modulation Networks**: [Focal Modulation Networks Model Zoo](https://github.com/microsoft/FocalNet).

## Citation

If you find this repo useful to your project, please consider to cite it with following bib:

    @misc{yang2022focalnet,  
      author = {Yang, Jianwei and Li, Chunyuan and Dai, Xiyang and Yuan, Lu and Gao, Jianfeng},
      title = {Focal Modulation Networks},
      publisher = {arXiv},
      year = {2022},
    }

and also:

    @misc{zhang2022dino,
          title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
          author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
          year={2022},
          eprint={2203.03605},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

