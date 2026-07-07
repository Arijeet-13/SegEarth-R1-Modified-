# Dataset🚀

Our model is trained separately on three datasets. You can download and organize these datasets in the following structure:

* [EarthReason](https://huggingface.co/datasets/earth-insights/EarthReason)
* [RRSIS-D](https://drive.google.com/drive/folders/1Xqi3Am2Vgm4a5tHqiV9tfaqKNovcuK3A)
* [RefSegRS](https://huggingface.co/datasets/JessicaYuan/RefSegRS)

```
data_path/
├── rs_reason_seg/
│   └── RSReasonSeg/
│       ├── test/
│       ├── train/
│       ├── val/
├── rs_ref_seg/
│   ├── RefSegRS/
│   │   ├── images/
│   │   ├── masks/
│   │   ├── output_phrase_test.txt
│   │   ├── output_phrase_train.txt
│   │   └── output_phrase_val.txt
│   └── RRSIS-D/
│       ├── images/
│       └── rrssid/
```

# Pretrained Weights📂

You can download the pre-trained weights of [Phi-1.5](https://huggingface.co/susnato/phi-1_5_dev) and [Mask2Former](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_384_bs16_50ep/model_final_9d7f02.pkl) from these links, and place them in the `pre_trained` folder according to the following structure:

```
pre_trained/
├── phi-1_5_dev/
│   └── ...
├── Swin_base/
│   └── model.pkl
```
