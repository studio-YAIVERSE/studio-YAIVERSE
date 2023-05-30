"""
Script Description
    - Generate Objaverse animal json file.
Usage
    - $ python generate_objaverse_animal_json.py \
        --model_root_dir <model_root_dir> \
        --image_root_dir <image_root_dir> \
        --input_csv <input_csv> \
        --output_json <output_json>
Author
    - Yunsu Park
"""


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_root_dir', type=str, default='Objaverse_model')
    parser.add_argument('--image_root_dir', type=str, default='objaverse/objaverse_result/img')
    parser.add_argument('--input_csv', type=str, default='objaverse_animal_list.csv')
    parser.add_argument('--output_json', type=str, default='objaverse_animal_processed.json')
    return parser.parse_args()


def main(args):
    import os
    import json
    import pprint
    import collections
    import pandas as pd

    assert os.path.isdir(args.model_root_dir) and os.path.isdir(args.image_root_dir)
    args.model_root_dir = os.path.abspath(args.model_root_dir)
    args.image_root_dir = os.path.abspath(args.image_root_dir)

    objaverse = pd.read_csv(args.input_csv, encoding='utf-8')
    objaverse_animal = objaverse.loc[(objaverse['Column2'] == 'o') | (objaverse['Column2'] == 'O')]
    objaverse_animal.reset_index(inplace=True)

    file_data = collections.OrderedDict()
    trg = objaverse_animal["Column1"]
    for i in range(len(trg)):
        file_data[f"{args.model_root_dir}{trg[i]}/model.glb"] = [f"{args.image_root_dir}{trg[i]}", 100]

    pprint.pprint(file_data)
    with open(args.output_json, 'w', encoding='utf-8') as fp:
        json.dump(file_data, fp, ensure_ascii=False, indent="\t")


if __name__ == '__main__':
    main(parse_args())
