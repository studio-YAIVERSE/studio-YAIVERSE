
from clip_loss import CLIPLoss
import argparse
import os
import torch
import json
import yaml
import shutil
import glob
import subprocess

# ##added ~ ## : js modified

def mapping_checkpoint(args) :

    device = "cuda:2"
    loss = CLIPLoss(device)

    checkpoint_group_path = args.checkpoint_group_dir #.pt
    tar_text = args.input_text
    tar_image = args.input_image
    pt_path = args.pt_folder_dir

    ###############################################################added
    #yaml_input = args.yaml_input_dir
    ###############################################################


    #Load saved source text embeddings
        #ex. {car:[[car_embedding], {rusty car : rustycar_embedding, haunted car : hauntedcar_embedding}], shoes:[[],{}]}
        # dict[car][0] : car embedding , dict[car][1]['rusty car'] : rusty car embedding
    target = torch.load(checkpoint_group_path)

    #remove
    tar_ = tar_text.split('_')
    tar_t = ''
    for tar in tar_ :
        tar_t = tar_t + tar+ ' '
    tar_text = tar_t
    print(tar_text)

    #target features
    if tar_image == None : 
        tar_feature = loss.non_templated_text(tar_text) #(1,512)
    else:
        tar_feature = loss.preprocessing_image(tar_image) #(1,512)


    #Select source text (key)
    ex_loss = torch.tensor(1000000, device = device)
    print(tar_text)
    for key in list(target.keys()) :
        print('---------------------key : ' ,key)
        key_loss_versus = loss.compute_loss(target[key][0], tar_feature)

        if ex_loss > key_loss_versus :
            ex_loss = key_loss_versus
            fin_key = key
    print(fin_key)
    #Select source text (value)
    val = target[fin_key][1] #{rustycar: , haunted car: }
    ex_loss = torch.tensor(1000000, device = device)
    ###############################################################added
    i=0
    for val in list(val.keys()) :
        i+=1
#        print('---------------------key : ' ,key)
        val_loss_versus = loss.compute_loss(target[fin_key][1][val], tar_feature)

        #compare with original category
        if i ==1 :
            print(val_loss_versus)
            if val_loss_versus > torch.tensor(0.1, device = device) :
                val_loss_versus = ex_loss

        ###############################################################
        if ex_loss > val_loss_versus :
            ex_loss = val_loss_versus
            fin_val = val
    ###############################################################added
    print('---------------------fin_val : ' ,fin_val)
    print('---------------------fin_loss : ' ,val_loss_versus)
    
    if ex_loss > torch.tensor(0.19, device = device) :
        
        tar_rename = tar_text.split(' ') # minsu green car
        tar_name = ''

        for tar in tar_rename :
            tar_name += tar #minsugreencar

        file_name = fin_key + '_' + tar_name + '_final' #car_minsugreencar_final
        #yaml_input = glob.glob(os.path.join(pt_path, file_name, '*.yaml')) #default path
        yaml_input = './experiments/shoes_sketch_dist_abl_autok_30.yaml'

        os.mkdir(os.path.join(pt_path, file_name))
        yaml_output = os.path.join(pt_path, file_name, file_name + '.yaml')
        shutil.copy(yaml_input, yaml_output)

        # load template yaml file -> change info -> write
        with open(yaml_output, encoding = 'UTF-8') as f :
            _cfg = yaml.load(f, Loader = yaml.FullLoader)
        _cfg['NADA']['target_text'] = tar_text
        _cfg['GLOBAL']['outdir'] = os.path.join(pt_path, file_name)
        _cfg['GLOBAL']['iter_2nd'] = 0

        ##################remove
        _cfg['GET3D']['n_views'] = 2

        print("OUTPUT PATH : " , yaml_output)
        with open(yaml_output, 'w') as f:
            yaml.dump(_cfg, f)
        
        # DO NADA !

        subprocess.call(f'python train_nada_dist.py --config_path {yaml_output} --suppress', shell=True)

        # RETURN CHECKPOINT
        fin_path = os.path.join(pt_path, file_name, 'checkpoint/best.pt') 

        #update checkpoint.pt
        update = torch.load('checkpoint_group.pt')
        print(tar_rename[-2])
        update[fin_key][1][tar_text] = loss.non_templated_text(tar_text)
        torch.save(update , 'checkpoint_group.pt')
        
    else:
        fin_rename = fin_val.split(' ')
        fin_name = ''
        for fin in fin_rename :
            fin_name += fin
        file_name = fin_key + '_' + fin_name + '_final' #car_rustycar_final
        fin_path = os.path.join(pt_path , file_name, file_name+'.pt' ) #./yaiverse/final/car_rustycar/car_rustycar.pt
    ##
    print(fin_path)
    return fin_path


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_group_dir', type = str, default = './checkpoint_group.pt')
    parser.add_argument('--input_text', type=str, default = None)
    parser.add_argument('--input_image', type = str, default = None)
    parser.add_argument('--pt_folder_dir', type=str, default = './Best_nada_results')
    ###############################################################added
    #parser.add_argument('--yaml_input_dir', type=str, default='./experiments/car_box_dist_abl_autok_20.yaml')
    ###############################################################
    args = parser.parse_args()

    mapping_checkpoint(args)
