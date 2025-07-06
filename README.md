# SPOT
### [Paper](https://arxiv.org/abs/2503.06471)
> [**Online Dense Point Tracking with Streaming Memory**](https://arxiv.org/abs/2503.06471)            
> Qiaole Dong, Yanwei Fu        
> **ICCV 2025**

## Requirements

```Shell
conda create --name spot python=3.9
conda activate spot
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install flash-attn --no-build-isolation
pip install cupy==12.3.0
pip install tqdm matplotlib einops einshape scipy timm lmdb av mediapy tensorboard numpy
```

## Models
We provide pretrained model under ckpts directory. The default path of the model for evaluation is:
```Shell
├── ckpts
    ├── spot.pth
```

## Demos
Run the following command:
```shell
python demo.py --ckpt_path ckpts/spot.pth --visualization_modes overlay_mask_stripes --video_path demo_input_images/color --mask_path demo_input_images/mask/00000.png --save_mode image --vis_dir demo_vis
```

## Reference
If you found our paper helpful, please consider citing:
```bibtex
@inproceedings{dong2025online,
  title={Online Dense Point Tracking with Streaming Memory},
  author={Dong, Qiaole and Fu, Yanwei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

## Acknowledgement

Thanks to previous open-sourced repo: 
* [DOT](https://github.com/16lemoing/dot)