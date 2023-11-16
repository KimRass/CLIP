# CLIP (Radford et al., 2021) implementation from scratch in PyTorch
- [Learning Transferable Visual Models From Natural Language Supervision](https://github.com/KimRass/CLIP/blob/main/papers/learning_transferable_visual_models_from_natural_language_supervision.pdf)
## Pretrained Model
- CLIP trained on Flickr8k + Flickr30k for 200 epochs
    - [clip_flickr.pth](https://drive.google.com/file/d/1BEKphn5BULRIMYJr5JT5_p2W8sYzJKHO/view?usp=drive_link)
## Linear Classification on ImageNet1k (mini) Dataset
```bash
# e.g.,
python3 linear_classification.py\
    --ckpt_path="../clip_flickr.pth"\
    --data_dir="../imagenet-mini/"\
    --n_epochs=64\
    --batch_size=128\
    --n_cpus=4 # Optional
```
- Top-5 accuracy on validation set: 5.8%
## Zero-shot Classification on ImageNet1k (mini) Dataset
```bash
# e.g.,
python3 zero_shot_classification.py\
    --ckpt_path="../clip_flickr.pth"\
    --data_dir="../imagenet-mini/"\
    --batch_size=16\
    --n_cpus=4\ # Optional
    --max_len=128\ # Optional
    --k=10 # Optional
```
- Top-10 accuracy on train + validation set: 3.0%
