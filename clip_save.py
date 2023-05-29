"""
To avoid 'Connection Reset by Peer' error by CLIP library, get ckpt using torch.jit
"""

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

import math
import clip  # pip install ftfy regex tqdm \ pip install git+https://github.com/openai/CLIP.git

if __name__ == "__main__":

    device = "cuda:0"
    
    model32, preprocess = clip.load('ViT-B/32', jit=True)
    model16, preprocess = clip.load('ViT-B/16', jit=True)
    modelcnn, _ = clip.load('RN50', jit=True)

    # torch.save(model32, 'clip-vit-b-32.pt')
    torch.jit.save(model32, 'clip-vit-b-32.pt')
    torch.jit.save(model16, 'clip-vit-b-16.pt')
    torch.jit.save(modelcnn, 'clip-cnn.pt')

    model32_re, _ = clip.load('clip-vit-b-32.pt')
    model16_re, _ = clip.load('clip-vit-b-16.pt')


    text = 'rusty car'
    embed32 = model32_re.encode_text(clip.tokenize(text).to(device))
    print(embed32.shape)
    embed16 = model16_re.encode_text(clip.tokenize(text).to(device))
    print(embed16.shape)
    print((embed16 * embed32).sum())
    

    # print(model32.__dict__.keys())
    # print(model16.__dict__)
    # print("SAVE")
    # torch.jit.save(model32, 'clip-vit-b-32.pt' )

    # print("LOAD")
    # model32_re = torch.jit.load('clip-vit-b-32.pt', map_location=device).eval()

    # # print(model32_re)
    # embed = model32_re.encode_text('rusty car')
    # print(embed.shape)
    # """
    # _modules : 'visual' 'token embedding' 'ln_final' 
    #             'positional embedding' 'text projection'
    
    # """