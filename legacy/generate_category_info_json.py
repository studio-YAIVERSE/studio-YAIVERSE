"""
Script Description
    - Get statistic of selected category's obj glb files, in Objaverse or ShapeNet dataset.
Usage
    - $ python generate_category_info_json.py \
        --dataset <'objaverse' or 'shapenet'> \
        --data_root_dir <data_root_dir> \
        --output_json <output_json>
Note
    - this code requires trimesh library to parse .glb file, so install by `pip install trimesh`
Author
    - Minsu Kim
"""

import os
from collections import namedtuple


Config = namedtuple(
    'Config',
    ['dataset', 'data_root_dir', 'output_json', 'categories', 'reader', 'append_minmax', 'directory_map']
)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, choices=['objaverse', 'shapenet'], required=True)
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--output_json', type=str, default='scale_info_{}.json')
    args = parser.parse_args()
    return args


def create_config(args) -> Config:
    try:
        output_json = args.output_json.format(args.dataset)
    except (KeyError, IndexError, ValueError):
        output_json = args.output_json
    assert os.path.isdir(args.data_root_dir)
    if args.dataset == 'objaverse':
        return Config(
            dataset=args.dataset,
            data_root_dir=args.data_root_dir,
            output_json=output_json,
            categories=sorted(os.listdir(args.data_root_dir)),
            reader=read_glb,
            append_minmax=False,
            directory_map={}
        )
    elif args.dataset == 'shapenet':
        import trimesh
        return Config(
            dataset=args.dataset,
            data_root_dir=args.data_root_dir,
            output_json=output_json,
            categories=['car', 'chair', 'loudspeaker', 'telephone', 'cabinet'],  # select keys
            reader=read_obj,
            append_minmax=True,
            directory_map={
                'table': '04379243',
                'car': '02958343',
                'chair': '03001627',
                'airplane': '02691156',
                'sofa': '04256520',
                'watercraft': '04530566',
                'bus': '02924116',
                'loudspeaker': '03691459',
                'telephone': '04401088',
                'cabinet': '02933112',
                'motorbike': '03790512',
            }
        )
    else:
        assert False


def read_glb(dirname):
    import trimesh
    mesh = trimesh.load(os.path.join(dirname, 'model.glb'))
    vtx_x = []
    vtx_y = []
    vtx_z = []
    for geo in mesh.geometry.values():  # has more than one object mesh
        for v in geo.vertices:
            vtx_x.append(float(v[0]))
            vtx_y.append(float(v[1]))
            vtx_z.append(float(v[2]))
    return vtx_x, vtx_y, vtx_z


def read_obj(dirname):
    vtx_x = []
    vtx_y = []
    vtx_z = []
    with open(os.path.join(dirname, 'model.obj'), 'r') as f:
        for line in f.readlines():
            if not line:
                break
            if line[:2] == 'v ':
                coords = line.split(' ')
                vtx_x.append(float(coords[1]))
                vtx_y.append(float(coords[2]))
                vtx_z.append(float(coords[3]))
    return vtx_x, vtx_y, vtx_z


def main(args):

    import json
    import pprint
    import numpy as np
    from tqdm import tqdm

    config = create_config(args)
    data_root_dir = config.data_root_dir   # path for objaverse
    categories = config.categories
    reader = config.reader
    output_json = config.output_json
    append_minmax = config.append_minmax
    directory_map = config.directory_map

    # 'axis' : [store info. of one mesh x N]
    categories_info = {}

    for category in categories:
        category_info = {'x': [], 'y': [], 'z': [], 'size': []}
        base = os.path.join(data_root_dir, directory_map.get(category, category))

        for sub in tqdm(sorted(os.listdir(base))):
            # inspect 3D bbox of one mesh
            vtx_x, vtx_y, vtx_z = reader(os.path.join(base, sub))
            line_x = abs(min(vtx_x) - max(vtx_x))
            line_y = abs(min(vtx_y) - max(vtx_y))
            line_z = abs(min(vtx_z) - max(vtx_z))

            if line_x > 0 and line_y > 0 and line_z > 0:    # filter invalid
                category_info['x'].append(line_x)
                category_info['y'].append(line_y)
                category_info['z'].append(line_z)
                category_info['size'].append(line_x * line_y * line_z)

        for key in ['x', 'y', 'z', 'size']:
            info = [np.mean(category_info[key]), np.std(category_info[key])]
            if append_minmax:
                info.extend([np.max(category_info[key]), np.min(category_info[key])])
            category_info[key] = info

        categories_info[category] = category_info

    # final result
    pprint.pprint(categories_info)

    with open(output_json, 'w') as f:
        json.dump(categories_info, f, indent=4)


if __name__ == '__main__':
    main(parse_args())
