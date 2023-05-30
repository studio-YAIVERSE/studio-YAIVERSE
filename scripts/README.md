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
