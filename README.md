<br>
<p align="center"><a href="https://github.com/studio-YAIVERSE"><img width=60% src="https://raw.githubusercontent.com/studio-YAIVERSE/studio-YAIVRSE/master/assets/logo_flat.png" alt="logo"></a></p>
<h4 align="center">
 <a href="https://github.com/studio-YAIVERSE/studio-YAIVERSE">AI Repo</a> &nbsp;&nbsp; | &nbsp;&nbsp; <a href="https://github.com/studio-YAIVERSE/Frontend">Frontend Repo</a> &nbsp;&nbsp; | &nbsp;&nbsp; <a href="https://github.com/studio-YAIVERSE/Backend">Backend Repo</a>
</h4>
<br>

# Studio-YAIverse

### "Text-guided 3D synthesis by GET3D + NADA"

|                                                     |                                             |
|-----------------------------------------------------|---------------------------------------------|
| <b>Car &rarr; Police</b>                            | <b>Car &rarr; Sketch</b>                    |
| ![car_policecar](assets/nada_car_police.gif)        | ![car_sketch](assets/nada_car_sketch.gif)   |
| <b>Motorbike &rarr; Tiger</b>                       | <b>Shoe &rarr; Mossy</b>                    |
| ![motorbike_tiger](assets/nada_motorbike_tiger.gif) | ![shoes_mossy](assets/nada_shoes_mossy.gif) |

> You can make any other interesting stylish 3D object!

<br>

---

## Requirements

### Git clone repo

- **On GET3D Repository**: You can just clone this repo into your own computer

```bash
git clone https://github.com/nv-tlabs/GET3D.git && cd GET3D
git clone https://github.com/studio-YAIVERSE/studio-YAIVERSE.git && cd studio-YAIVERSE
```

- **Without GET3D Repository**: Use GitHub submodule, so you can clone GET3D repository automatically.

```bash
git clone --recursive https://github.com/studio-YAIVERSE/studio-YAIVERSE.git
cd studio-YAIVERSE
# if you are updating an existing checkout
git submodule sync && git submodule update --init --recursive
```

<br>

### Environment setup with Docker or Anaconda

- Docker container

```
cd docker
docker build -f Dockerfile -t studio-yaiverse:v1 .
cd ..
```

- Anaconda environment

```
conda create -n get3d python=3.8
conda activate get3d
pip install -r requirements.txt
```

<br>

### Download checkpoints

For GET3D + NADA, you need pretrained model's checkpoint. You can set downloaded ckpt path at yaml file.

- Car, Chair, Table, Motorbike &rarr; [link](https://github.com/nv-tlabs/GET3D/tree/master/pretrained_model)

- Fruits, Shoe &rarr; [link](https://huggingface.co/datasets/allenai/objaverse/discussions/1#63c0441bd9e14fd8875cec97)

<br>

And finally the directory hierarchy is configured as,

```
GET3D
├── studio-YAIVERSE (= Your Working Directory)
|	├── assets
|	├── docker
|	├── experiments
|	|	├── *.yaml
|	|	└── ....
|	├── legacy
|	├── results
|	|	├── default
|	|	|	  ├── checkpoint
|	|	|	  ├── sample
|	|	|	  └── default_date.log
|	|	└── ....
|	├── scripts
|	|	├── README.md
|	|	├── clip_save.py
|	|	├── image_to_video.py
|	|	└── render_shapenet_multigpu.py
|	├── sample_img
|	├── LICENSE
|	├── README.md
|	├── clip_loss.py
|	├── dist_util.py
|	├── functional.py
|	├── model_engine.py
|	├── nada.py
|	├── requirements.txt
|	├── text_templates.py
|	└── train_nada.py
├── 3dgan_data_split
├── data
└── ....
```

or,

```
studio-YAIVERSE (= Your Working Directory)
├── GET3D (submodule)
|	├── 3dgan_data_split
|	├── data
|	└── ....
├── assets
├── docker
├── experiments
|	├── *.yaml
|	└── ....
├── legacy
├── results
|	├── default
|	|	  ├── checkpoint
|	|	  ├── sample
|	|	  └── default_date.log
|	└── ....
├── scripts
|	├── README.md
|	├── clip_save.py
|	├── image_to_video.py
|	└── render_shapenet_multigpu.py
├── sample_img
├── LICENSE
├── README.md
├── clip_loss.py
├── dist_util.py
├── functional.py
├── model_engine.py
├── nada.py
├── requirements.txt
├── text_templates.py
└── train_nada.py
```

<br>

---

## Train

### Train code

If you want to train the code, please refer to the training script below.

```
$ # working directory: studio-YAIVERSE
$ python train_nada.py --config_path='experiments/{}.yaml' --name='{}' --suppress

optional arguments
	--config_path             select yaml file to run (in experiments folder)
	--name                    choose any name you want for log file name (optional)
	--suppress                store only latest & best pkl file

EX)
$ python train_nada.py --config_path='experiments/car_police_example.yaml' --name='car_police' --suppress
```

<br>

### Trainable Parameters

When you open yaml file, you could see many trainable parameters and configs.

Among them, below are some important parameters you could change as you conduct an experiment.

We provide some yaml files as [examples](./experiments).

<br>

**Global Config**

|          | Default Setting | Detailed explanation                                                                                                                                                                      |
|----------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| batch    | 3               | Setting the batch number less than 3 resulted unfavorable results in most of the experiments. However, you could change this value to some other value that fits well to your experiments |
| iter_1st | 1               | For most of the cases, 1 was enough to generate 3d object you want. You could increase this value to see more changes in the generated objects                                            |
| iter_2nd | 30              | For most of the cases, since model converges after iter_1st, 1 was enough to generate 3d object you want. You could increase this value to see more changes in the generated objects      |

<br>

**GET3D config**

|         | Default Setting | Detailed explanation                                                                       |
|---------|-----------------|--------------------------------------------------------------------------------------------|
| n_views | 12              | You can change this value that fits your GPU memory. According to Paper, set n_views >= 16 |

<br>

**NADA config**

|                         |  Default Setting  |  Detailed explanation                                                                                                                                                                                                                                                                   |
|-------------------------|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| lr                      | 0.002             | For most of the experiments lr 0.002 was suitable. However, you can change this value that fits your task.                                                                                                                                                                              |
| auto_layer_k            | 20 , 30           | auto_layer_k means the number of trainable layers during adaptation of GET3D. We empirically found some following tips. For texture changes + slight shape changes, setting the auto_layer_k to 20 was suitable. For only texture changes, setting the auto_layer_k to 30 was suitable. |
| source text             | pretrained object | For most of the experiments, we simply set source text to pretrained object. However, we found out that giving some text prompt to this variable showed some improvements in some cases. EX) 3D object car                                                                              |
| target text             | target object     | For most of the experiments, we simply set target text to target object. However, we found out that giving some text prompt to this variable showed some improvements in some cases. EX) 3D render in the style of Pixar                                                                |
| gradient_clip_threshold | -1                | For most of the experiments, not using gradient_clip(set as -1) was suitable. However, if the task requires some major changes in shape, using gradient clip was helpful.                                                                                                               |

<br>

---

## Inference

### Changing GET3D Checkpoint

* You may have to change GET3D model's checkpoint to apprpriate one, by `clip_loss` value of given prompt text or image.
* Function `match_checkpoint`, in [`legacy/backend_input_checkpoint_map.py`](./legacy/backend_input_checkpoint_map.py), returns best combination of source and target text. You can parse best checkpoint from it, like given example code:
   ```python
   ...
   get3d_generator: GeneratorDMTETMesh
   
   from legacy.backend_input_checkpoint_map import match_checkpoint, parse_checkpoint_path
   src, trg = match_checkpoint(
       input_text,
       input_image_path,
       embedding_group_pt_path,
       device
   )
   checkpoint_path = parse_checkpoint_path(src, trg, args.checkpoint_root))
   get3d_generator.load_state_dict(torch.load(checkpoint_path))
   ...
   # Your inference code with GET3D...
   ```
* `embedding_group_pt_path` is the path of "embedding_group_pt", which could be built from  [`legacy/generate_embedding_group_pt.py`](./legacy/generate_embedding_group_pt.py) script.

<br>

### (Note) Full inference instruction of backend

* You can see end-to-end inference code - from text or image to 3d binary files - in backend's `studio_YAIVERSE.apps.main.pytorch` package. See [here](https://github.com/studio-YAIVERSE/Backend/blob/master/studio_YAIVERSE/apps/main/pytorch/__init__.py).
* Since `pytorch` package is implemented with related import method, you can copy `pytorch` folder and just harness it, like given example:
   ```python
   from pytorch import init, inference
   
   YOUR_CONFIG: dict  # Write your config refer to init.__doc__
   init(YOUR_CONFIG)
   
   result = inference("object_name", "your_text_prompt")
   
   with open("result.glb", "wb") as fp:
       fp.write(result.file.getvalue())  # result.file: BytesIO
   
   with open("thumbnail.png", "wb") as fp:
       fp.write(result.thumbnail.getvalue())  # result.thumbnail: BytesIO
   ```

<br>

### Our checkpoint for inference

* We provide our checkpoint for some text templates, in backend repository. See [checkpoint release](https://github.com/studio-YAIVERSE/Backend/releases/tag/1.0.0).

<br>

---

## Appendix

### CLIP util
* We provide `scripts/clip_save.py` to avoid 'connection reset by peer' error from CLIP library, which accidentally stops the runtime.

1. Do `python scripts/clip_save.py`, and then you can get `clip-cnn.pt` / `clip-vit-b-16.pt` / `clip-vit-b-32.pt`
2. Change `clip.load()` argument as follows (Note that this is used at `clip_loss.py`)
   - `clip.load('RN50')` &rarr; `clip.load('/PATH/TO/clip-cnn.pt')`
   - `clip.load('ViT-B/16')` &rarr; `clip.load('/PATH/TO/clip-vit-b-16.pt')`
   - `clip.load('ViT-B/32')` &rarr; `clip.load('/PATH/TO/clip-vit-b-32.pt')`

### Image to Video

* We provide `scripts/image_to_video.py` to convert image sequences into a video.
* Do `python scripts/image_to_video.py -i /PATH/TO/INPUT_IMAGE_DIR -o /PATH/TO/OUTPUT_VIDEO_FILE`
* This code is used for visualizing our result like Introduction section.

### Rendering ShapeNet with Multi-GPU

* Rendering ShapeNet is required for training GET3D model with new category.
* We provide `scripts/render_shapenet_multigpu.py` to render ShapeNet with multi-GPU.
* This code is same with `GET3D/render_shapenet_data/render_all.py`, but makes blender rendering faster.
* Copy this script's content to `GET3D/render_shapenet_data/render_all.py` and run it.

<br>

---

## Note

* Please note that this is not official code by GET3D authors. There may be differences in detail from the original.
* [`TAPS3D`](https://github.com/studio-YAIVERSE/studio-YAIVERSE/tree/TAPS3D) branch contains our incomplete trial of GET3D training code + TAPS3D training code.
* If you have any question for our team project, don't hesitate to leave an issue or email to [minsu1206@yonsei.ac.kr](mailto:minsu1206@yonsei.ac.kr). Thanks.
