## studio-YAIVERSE Appendix Scripts

### CLIP util
* We provide `clip_save.py` to avoid 'connection reset by peer' error from CLIP library, which accidentally stops the runtime.

1. Do `clip_save.py`, and then you can get `clip-cnn.pt` / `clip-vit-b-16.pt` / `clip-vit-b-32.pt`
2. Change `clip.load()` argument as follows (Note that this is used at `clip_loss.py`)
   - `clip.load('RN50')` &rarr; `clip.load('/PATH/TO/clip-cnn.pt')`
   - `clip.load('ViT-B/16')` &rarr; `clip.load('/PATH/TO/clip-vit-b-16.pt')`
   - `clip.load('ViT-B/32')` &rarr; `clip.load('/PATH/TO/clip-vit-b-32.pt')`

### Image to Video

* We provide `image_to_video.py` to convert image sequences into a video.
* Do `python image_to_video.py -i /PATH/TO/INPUT_IMAGE_DIR -o /PATH/TO/OUTPUT_VIDEO_FILE`
* This code is used for visualizing our result like Introduction section.

### Rendering ShapeNet with Multi-GPU

* Rendering ShapeNet is required for training GET3D model with new category.
* We provide `render_shapenet_multigpu.py` to render ShapeNet with multi-GPU.
* This code is same with `GET3D/render_shapenet_data/render_all.py`, but makes blender rendering faster.
* Copy this script's content to `GET3D/render_shapenet_data/render_all.py` and run it.
