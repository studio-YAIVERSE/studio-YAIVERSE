## Introduction

Our branch 'taps3d' contains the GET3D training code + TAPS3D training code.
Please note that we encountered difficulties while implementing the TAPS3D code. Due to the lack of some details in the paper, we were unable to reproduce the results.
Unfortunately, mode collapse occurs when running the TAPS3D training code.
We hope that our code can be used as a reference until the official code is released.

--- 

## How to use this code.

### GET3D

: just follow official repository.

<br>

### TAPS3D

#### Prepare data

extract and save CLIP feature embedding for training

given a dataset_path structured as the following,

```python
.
└── dataset_path/
    ├── camera/
    │   ├── object_class_1/
    │   │   ├── object_name_1/
    │   │   │   ├── elevation.npy
    │   │   │   └── rotation.npy
    │   │   ├── object_name_2
    │   │   └── ...
    │   ├── object_class_2
    │   └── ...
    └── img/
        ├── object_class_1/
        │   ├── object_name_1/
        │   │   ├── 000.png
        │   │   ├── 001.png
        │   │   └── ...
        │   ├── obejct_name_2
        │   └── ...
        ├── object_class_2
        └── ...
```

 to generate pseudo captions and its corresponding clip features, run

```bash
 python caption_gen.py --path {img path} --camera_path {camera path} --data_camera_mode {camera mode} --split all --clip_patch 32 --caption_embedding_path {path/to/save_captions} --noun_vocab_path {path/to/noun_vocabs}
 
(example)
python caption_gen.py --path /opt/myspace/data/shapenet_chair_result/img --camera_path /opt/myspace/data/shapenet_chair_result/camera --data_ca
mera_mode shapenet_chair --split all --clip_patch 32 --caption_embedding_path /opt/myspace/data/shapenet_chair_result/caption_embedding --noun_vocab_path ./noun_vocab.txt
```

which will output a caption_embedding directory of the following structure 

```python

caption_embedding/
    ├── object_class_1/
    │   ├── object_name_1/
    │   │   ├── caption_16.npy
    │   │   ├── catpion_16.txt
    │   │   ├── caption_32.npy
    │   │   └── caption_32.txt
    │   ├── object_name_2
    │   └── ...
    ├── object_class_2
    └── ...

```

Note that unlike noun vocabulary, adj_vocabs.txt is provided for you. To use custom adjective vocabulary, just edit adj_vocabs.txt.

<br>

#### Training : TAPS3D

: add `--taps3d True` and `--resume_pretrain /PATH/TO/GET3D.ckpt` into original trainig code.

```
(Example)

python train_3d.py --outdir ./chair_TAPS3D --data /opt/myspace/data/shapenet_chair_result/img --camera_path /opt/myspace/data/shapenet_chair_result/camera --gpus 2 --batch 8 --batch-gpu 4 --gamma 40 --data_camera_mode shapenet_chair --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --taps3d True --resume_pretrain ./pretrained_model/shapenet_car.pt
```



