import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip
from PIL import Image
import numpy as np
import time
import sys
import os
import random
from tqdm import tqdm

from training.dataset import DebugDataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path',type = str, help = "path for images")
parser.add_argument('--camera_path',type = str, help = "path camera_path")
parser.add_argument('--data_camera_mode',type = str)
parser.add_argument('--split', type = str)
parser.add_argument('--clip_patch', type=int, help = "which patch size will you use for clip-VIT? (32 or 16)")
parser.add_argument('--caption_embedding_path',type = str, help = "path for saving captions")
parser.add_argument('--noun_vocab_path', type = str, help = "path for noun vocabs")
parser.add_argument('--batch_size', type = int, default=24)
args = parser.parse_args()

data_path = args.path
camera_path = args.camera_path
split = args.split
data_camera_mode = args.data_camera_mode

clip_patch = args.clip_patch
assert clip_patch == 16 or clip_patch == 32

caption_embedding_path = args.caption_embedding_path
noun_vocab_path = args.noun_vocab_path
batch_size = args.batch_size

torch.cuda.empty_cache() 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'making captions with CLIP VITB-{clip_patch}!!')

print("loading dataset...")
dataset = DebugDataset(path = data_path, 
                        camera_path= camera_path,
                        resolution = 224,
                        data_camera_mode = data_camera_mode,
                        gen_caption = True,
                        debug = False,
                        split = split,
                        clip_patch = 16,
                        version = 'GET3D',
                        )

print("dataset length: ", len(dataset))

print("loading dataloader...")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size) 
nn_cos_sim = torch.nn.CosineSimilarity(dim=2, eps=1e-08).to(device)

# adjective vocabulary
print("loading adjective vocab...")
with open('adj_vocabs.txt', 'r') as f:
    lines = f.readlines()

adj_vocabs = []
for line in lines:
    adj_vocabs.append(line.strip())
    
adj_vocabs_np = np.array(adj_vocabs)

print("loading noun vocab...")
with open(noun_vocab_path, 'r') as f:
    lines = f.readlines()

noun_vocabs = []
for line in lines:
    noun_vocabs.append(line.strip())
    
noun_vocabs_np = np.array(noun_vocabs)


print("loading CLIP...")
if clip_patch == 32:
    model, preprocess = clip.load("ViT-B/32", device=device)
else:
    model, preprocess = clip.load("ViT-B/16", device=device)

print("making adj, noun features...")
adj_tokens = clip.tokenize(adj_vocabs).to(device)
noun_tokens = clip.tokenize(noun_vocabs).to(device)
with torch.no_grad():
    adj_features = model.encode_text(adj_tokens).unsqueeze(0) #[1,1248,512]
    noun_features = model.encode_text(noun_tokens).unsqueeze(0) #[1,13,512]

k_a = 6
k_n = 3
for epoch, data in enumerate(tqdm(dataloader)):
    fname = data[3][0]
    if "model" in fname:
        fname = fname.replace("/model","")
    class_dir_name = fname.split('/')[-3] # the dir name of class (cars dir name is 02958343) 
    obj_dir_name = fname.split('/')[-2] # the dir name of 3D object

    img_data = data[0].to(device) # (24,3,224,224)
    embedding = model.encode_image(img_data).unsqueeze(1) #(24,1,512)

    adj_img_cos = nn_cos_sim(adj_features,embedding) #(24,1248)
    noun_img_cos = nn_cos_sim(noun_features,embedding) #(24,13)
    
    top_adj_indices = adj_img_cos.topk(k_a, dim=1)[1].detach().cpu().tolist() #(batch_size, k_a) indicies
    top_noun_indices = noun_img_cos.topk(k_n, dim=1)[1].detach().cpu().tolist()#(batch_size, k_n) indicies
    
    adj_list = [[adj_vocabs[ind] for ind in adj_indicies_for_one] for adj_indicies_for_one in top_adj_indices] #(batch_size, k_a) adjectives
    noun_list = [[noun_vocabs[ind] for ind in noun_indicies_for_one] for noun_indicies_for_one in top_noun_indices] #(batch_size, k_n) nouns
    # print(noun_list)
    
    caption_list = []
    for i in range(batch_size):
        captions_for_image = []
        for adj in adj_list[i]:
            for noun in noun_list[i]:
                random_num = random.randint(0,k_a-1)
                if random.randint(0,1):# 1이면 adjtive 한개
                    caption = "a "+ adj + " " + noun
                else: #0 이면 adjective 2개
                    if adj == adj_list[i][random_num] :#같은 adjective면 
                        if random_num == k_a-1: 
                            random_num = -1
                        caption = "a "+ adj + " " + adj_list[i][random_num+1]+ " " + noun
                    else: # 다른 adjective면
                        caption = "a "+ adj + " " + adj_list[i][random_num]+ " " + noun
                captions_for_image.append(caption)
        caption_list.append(captions_for_image)
        
    
    top_caption_indice_list = [] 
    feature_list = []
    #find top caption for each  2D image
    for i in range(batch_size):     
        caption_tokens = clip.tokenize(caption_list[i]).to(device)
        caption_features = model.encode_text(caption_tokens).unsqueeze(0) #(1,K_a*K_n,512)
        caption_img_cos = nn_cos_sim(embedding[i].unsqueeze(1),caption_features) #(1,K_a*K_n) cos similarity

        top_caption_indice = caption_img_cos.topk(1, dim=1)[1].detach().cpu().tolist()[0][0] # indice that has highest cos imilarity
        top_caption_indice_list.append(top_caption_indice) #확인용
        # print(f"top caption for {i}th image is ",caption_list[i][top_caption_indice])

        best_caption_feature = caption_features[0][top_caption_indice] #(512)
        feature_list.append(best_caption_feature)

    best_caption_features = torch.stack(feature_list, dim=0) #(24,512)
    # print(best_caption_features.shape)
    
    #finding best caption for object
    #embedding (24,1,512)
    caption_img_cos = nn_cos_sim(best_caption_features.unsqueeze(0),embedding) #(24,24) (image_embedding, captions)
    # print("cosine sim average (wrt all of rendered images) of the above captions")

    #top 20 captions
    top20_caption_for_object_indicies = caption_img_cos.sum(dim=0).topk(20, dim=0)[1]
    top20_caption_for_object = [caption_list[ind][top_caption_indice_list[ind]] for ind in top20_caption_for_object_indicies]
 

    # best caption
    best_caption_for_object_ind = torch.argmax(caption_img_cos.sum(dim=0))
    best_caption_for_object = caption_list[best_caption_for_object_ind][top_caption_indice_list[best_caption_for_object_ind]]
                            #caption_list[image index that has best caption][get the index for the best caption for the image]
    print("best caption is: ", best_caption_for_object)

    top20_caption_for_object_features = best_caption_features[top20_caption_for_object_indicies].cpu().detach().numpy() # [20,512]
    best_caption_for_object_feature = best_caption_features[best_caption_for_object_ind].cpu().detach().numpy() # [512]


    # 파일 저장 경로
    caption_save_dir = os.path.join(os.path.join(caption_embedding_path,class_dir_name),obj_dir_name)

    os.makedirs(caption_save_dir, exist_ok=True)
    # # top 20 feature 저장
    np.save(os.path.join(caption_save_dir,f'caption_{clip_patch}'),top20_caption_for_object_features)
    # top 
    with open(os.path.join(caption_save_dir,f'caption_{clip_patch}.txt'), 'w') as f:
        for line in top20_caption_for_object:
            f.write(line)
            f.write('\n')
