"""
Script Description
    - This script is used to find the best checkpoint.pt file for the given input text, in backend.
    - See https://github.com/studio-YAIVERSE/Backend/blob/5a8524585a5f3864790729978e5da1c6afb72062/studio_YAIVERSE/apps/main/pytorch/functions.py#L243
      to see how this script is used in backend.
Usage
    - $ python backend_input_checkpoint_map.py --input_text "car"
    - $ python backend_input_checkpoint_map.py --input_image_path "./car.jpg"
Author
    - Jisoo Kim
"""

from clip_loss import CLIPLoss
import os
import torch
import yaml
import subprocess

DEBUG = False


def match_checkpoint(
        input_text,
        input_image_path,
        embedding_group_pt_path,
        device="cpu"
):
    # load CLIPLoss if first time
    global loss
    if loss is None:
        loss = CLIPLoss(device)

    # load embedding group
    embedding_group = torch.load(embedding_group_pt_path)

    # target features
    if input_image_path is None:
        tar_feature = loss.non_templated_text(input_text.replace('_', ' '))  # shape: (1,512)
    else:
        tar_feature = loss.preprocessing_image(input_image_path)  # shape: (1,512)

    # Select source text (key)
    ex_loss = float('inf')
    fin_key = None
    for key in embedding_group:
        key_loss_versus = loss.compute_loss(embedding_group[key][0], tar_feature)
        if ex_loss > key_loss_versus:
            ex_loss = key_loss_versus
            fin_key = key
    source = fin_key
    print("Selected source key:", source, "| Loss:", ex_loss)

    # Select target text (value)
    ex_loss = float('inf')
    fin_val = None
    for val in embedding_group[fin_key][1]:
        val_loss_versus = loss.compute_loss(embedding_group[fin_key][1][val], tar_feature)
        if fin_key == val and val_loss_versus > 0.1:  # compare with original category
            val_loss_versus = ex_loss
        if ex_loss > val_loss_versus:
            ex_loss = val_loss_versus
            fin_val = val
    target = fin_val
    print("Selected target key:", target, "| Loss:", ex_loss)

    if DEBUG and ex_loss > 0.19 and input_image_path is None:  # TODO
        target = None

    return source, target


def parse_checkpoint_path(source, target, checkpoint_root):
    # TODO: adjust this function to match the checkpoint path
    file_name = source + '_' + target.replace(' ', '') + '_final'  # car_rustycar_final
    return os.path.join(checkpoint_root, file_name, file_name + '.pt')  # ./yaiverse/final/car_rustycar/car_rustycar.pt


def retrain_nada(input_text, source, pt_path):  # TODO
    # Note: this function is not used in the backend, due to runtime issue

    # prepare
    tar_text = input_text.replace('_', ' ')
    tar_name = tar_text.replace(' ', '')  # minsu green car -> minsugreencar
    file_name = source + '_' + tar_name + '_final'  # car_minsugreencar_final
    os.mkdir(os.path.join(pt_path, file_name))

    # load template yaml file and adjust it to save new yaml file
    yaml_input = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
    yaml_input = os.path.join(yaml_input, sorted(fn for fn in os.listdir(yaml_input) if fn.endswith('.yaml'))[0])
    with open(yaml_input, encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['NADA']['target_text'] = tar_text
    cfg['GLOBAL']['outdir'] = os.path.join(pt_path, file_name)
    cfg['GLOBAL']['iter_2nd'] = 0
    yaml_output = os.path.join(pt_path, file_name, file_name + '.yaml')
    with open(yaml_output, 'w') as f:
        yaml.dump(cfg, f)

    # Traun NADA with new yaml file via subprocess
    print("RE-TRAINING WITH YAML : ", yaml_output)
    subprocess.call(f'python train_nada_dist.py --config_path {yaml_output} --suppress', shell=True)

    # update checkpoint.pt
    update = torch.load('checkpoint_group.pt')
    update[source][1][tar_text] = loss.non_templated_text(tar_text).cpu()
    torch.save(update, 'checkpoint_group.pt')

    # final checkpoint path
    return os.path.join(pt_path, file_name, 'checkpoint/best.pt')


loss: "CLIPLoss|None" = None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_group_pt_path', type=str, default='../checkpoint_group.pt')
    parser.add_argument('--input_text', type=str, default=None)
    parser.add_argument('--input_image_path', type=str, default=None)
    parser.add_argument('--checkpoint_root', type=str, default='../best_nada_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    src, trg = match_checkpoint(
        args.input_text,
        args.input_image_path,
        args.embedding_group_pt_path,
        args.device
    )
    if trg is None:
        print(retrain_nada(args.input_text, src, args.checkpoint_root))
    else:
        print(parse_checkpoint_path(src, trg, args.checkpoint_root))
