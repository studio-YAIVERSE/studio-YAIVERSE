"""
Script Description
    - This scripts makes embedding_group file containing embeddings from source text and target text,
      which is used in `backend_input_checkpoint_map.py` to find the best checkpoint for the given input text.
    - Source texts and target texts are found from the name of folder containing checkpoints.
    - The embedding_group file is a dictionary with the following structure:
        >>> car_embedding = rustycar_embedding = hauntedcar_embedding = ...
        >>> embedding_group = {
        ...     'car': [
        ...         [car_embedding],
        ...         {'rusty car': rustycar_embedding, 'haunted car': hauntedcar_embedding}
        ...     ],
        ...     'shoes': [[], {}]
        ... }
    - You can access the embedding of source text and target text by indexing the dictionary:
        >>> import torch
        >>> embedding_group = torch.load('checkpoint_group.pt')
        >>> embedding_group['car'][0] == car_embedding  # True
        >>> embedding_group['car'][1]['rusty car'] == rustycar_embedding  # True
Usage
    - $ python generate_embedding_group_pt.py \
        --checkpoint_root ../best_nada_results \
        --output_path ../checkpoint_group.pt
Author
    - Jisoo Kim
"""


def main(args):
    import os
    import glob
    import yaml
    import torch
    from clip_loss import CLIPLoss
    loss = CLIPLoss(args.device).eval()
    embedding_group = {}
    args.checkpoint_root = os.path.abspath(os.path.expanduser(args.checkpoint_root))
    for folder in os.listdir(args.checkpoint_root):
        with open(
                glob.glob(os.path.join(args.checkpoint_root, folder, '*.yaml'))[0],
                encoding='UTF-8'
        ) as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
        target_text = cfg['NADA']['target_text']
        if cfg['NADA']['source_text'] not in cfg['NADA']['target_text']:
            target_text = cfg['NADA']['target_text'] + ' ' + cfg['NADA']['source_text']
        name = folder.split('_')
        with torch.no_grad():
            if name[0] not in embedding_group.keys():
                embedding_group[name[0]] = []
                # save souce_text embedding
                embedding_group[name[0]].append(loss.templated_mean_text(name[0]).cpu())
                # save target_text embedding
                embedding_group[name[0]].append({})
                embedding_group[name[0]][1][name[0]] = loss.templated_mean_text(name[0]).cpu()
                embedding_group[name[0]][1][target_text] = loss.non_templated_text(target_text).cpu()
            else:
                embedding_group[name[0]][1][target_text] = loss.non_templated_text(target_text).cpu()
    torch.save(embedding_group, args.output_path)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_root', type=str)
    parser.add_argument('--output_path', type=str, default='checkpoint_group.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
