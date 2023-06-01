#!/usr/bin/env python3
# Orig: GET3D/render_shapenet_data/render_all.py
"""
Script Description
    - Render all the 3D models in the given ShapeNet dataset folder.
    - This script is Multi-GPU version of GET3D/render_shapenet_data/render_all.py
Usage
    - python render_all.py \
        --save_folder PATH_TO_SAVE_IMAGE \
        --dataset_folder PATH_TO_3D_OBJ \
        --blender_root PATH_TO_BLENDER \
        --num_gpus NUM_GPUS
Author
    - Dongha Kim
"""
import os
import sys
import shlex
import argparse
import subprocess
import multiprocessing
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--save_folder', type=str, default='./tmp',
    help='path for saving rendered image')
parser.add_argument(
    '--dataset_folder', type=str, default='./tmp',
    help='path for downloaded 3d dataset folder')
parser.add_argument(
    '--blender_root', type=str, default='./tmp',
    help='path for blender')
parser.add_argument(
    '--num_gpus', type=int, default=1,
    help='number of gpus to use')
parser.add_argument(
    '--script_path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'render_shapenet.py'),
    help='path for `render_shapenet.py` script used in blender'
)
args = parser.parse_args()

synset_scale_list = [
    ('02958343', 0.9, 24),  # Car
    ('03001627', 0.7, 24),  # Chair
    ('03790512', 0.9, 100),   # Motorbike
    # ('02924116', 1.1, 24),  # bus
    # ('02691156', 0.9, 24),  # airplane
    # ('04530566', 1.1, 24),  # watercraft
    # ('04256520', 0.9, 24),  # sofa
    # ('03691459', 0.7, 40),  # loudspeaker
    # ('04401088', 1.2, 40),  # telephone
    # ('02933112', 1.0, 40),  # cabinet
]


def worker(filenames, synset, obj_scale, n_views, save_folder, dataset_folder, blender_root, script, gpu_index, i):
    prev = os.getenv('CUDA_VISIBLE_DEVICES')
    _render_cmd = ('%s -b -P %r -- ' % (blender_root, script)) + '%r'
    _render_cmd += ' --output %r --scale %f --views %d --resolution 1024' % (save_folder, obj_scale, n_views)
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
        for filename in tqdm(
                filenames, desc="GPU: {}; synset: {}".format(gpu_index, synset), position=i, file=sys.stdout
        ):
            render_cmd = _render_cmd % (os.path.join(dataset_folder, synset, filename, 'model.obj'))
            result = subprocess.call(
                shlex.split(render_cmd),
                stdout=open("tmp_{}.out".format(gpu_index), "w"),
                stderr=subprocess.PIPE
            )
            if result:
                raise RuntimeError("Error at file: {}; Errcode: {}; Command: {}".format(filename, result, render_cmd))
    finally:
        if prev is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = prev
        else:
            del os.environ['CUDA_VISIBLE_DEVICES']


def main():
    _available_gpus = list(range(args.num_gpus))
    for synset, obj_scale, n_views in synset_scale_list:
        file_list = sorted(os.listdir(os.path.join(args.dataset_folder, synset)))
        ps = []
        try:
            for i, gpu_index in enumerate(_available_gpus):
                ps.append(
                    multiprocessing.Process(
                        target=worker,
                        args=(file_list[i::len(_available_gpus)], synset, obj_scale, n_views,
                              args.save_folder, args.dataset_folder, args.blender_root, args.script_path,
                              gpu_index, i)
                    )
                )
            for p in ps:
                p.start()
        finally:
            for p in ps:
                p.join()


if __name__ == '__main__':
    main()
