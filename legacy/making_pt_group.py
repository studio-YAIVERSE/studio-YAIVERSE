## making json file for checkpoint groups

import argparse
import os
import yaml
from clip_loss import CLIPLoss
import torch
import glob

def group_maker(args) :
    folder_list = os.listdir(args.file_path) #./yaiverse/final/..
    device = "cuda:2"
    loss = CLIPLoss(device)

    f_list = {}
    for folder in folder_list :
        name = folder.split('_')
        
        with open(glob.glob(os.path.join(args.file_path,folder,'*.yaml'))[0], encoding = 'UTF-8') as f :
                _cfg = yaml.load(f, Loader = yaml.FullLoader)

        target_text = _cfg['NADA']['target_text']
        if _cfg['NADA']['source_text'] not in _cfg['NADA']['target_text'] :
             target_text = _cfg['NADA']['target_text'] + ' ' + _cfg['NADA']['source_text'] #taxi car
             

        #ex. {car:[[car_embedding], {rusty car : rustycar_embedding, haunted car : hauntedcar_embedding}], shoes:[[],{}]}
        # dict[car][0] : car embedding , dict[car][1]['rusty car'] : rusty car embedding
        if name[0] not in f_list.keys():
            f_list[name[0]] = []
            #save souce_text embedding 
            f_list[name[0]]
            f_list[name[0]].append(loss.templated_mean_text(name[0]))
            #save target_text embedding
            f_list[name[0]].append({})
            f_list[name[0]][1][name[0]] = loss.templated_mean_text(name[0])
            f_list[name[0]][1][target_text] = loss.non_templated_text(target_text)
        else:
            f_list[name[0]][1][target_text] = loss.non_templated_text(target_text)

    torch.save(f_list, os.path.join(args.output_path, 'checkpoint_group.pt'))
 


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type = str)
    parser.add_argument('--output_path' , type = str)
    args = parser.parse_args()

    group_maker(args)
