import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image
from lavis.models import load_model_and_preprocess
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import *




def main(args):
    transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
                            torchvision.transforms.Resize(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
                            torchvision.transforms.Lambda(lambda img: to_tensor(img))])
    
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device="cuda")
    model.eval()
    img = Image.open(args.img_path).convert("RGB")
    img = transform(img).cuda().unsqueeze(0)
    
    caption = image_2_cap(model, img)
    print(caption)   
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--model_name", default="blip2_opt", type=str)
    parser.add_argument("--model_type", default="pretrain_opt2.7b", type=str)

    args = parser.parse_args()
    
    main(args)