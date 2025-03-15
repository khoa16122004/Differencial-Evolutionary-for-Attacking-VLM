import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image
from lavis.models import load_model_and_preprocess
from torch.utils.data import ImageCaptionDataset, image_2_cap
from tqdm import tqdm
from utils import *




def main(args):
    seed_everything(22520691)
    transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
                            torchvision.transforms.Resize(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
                            torchvision.transforms.Lambda(lambda img: to_tensor(img))])
    
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device="cuda")
    model.eval()
    img = Image.open(args.img_path).convert("RGB")
    img = transform(img).cuda().unsqueeze(0)
    
    caption = img_2_cap(model, img)

    # data
    data = ImageCaptionDataset(annotations_file=args.annotation_path,
                               image_dir=args.image_dir,
                               target_dir=args.target_dir,
                               target_resolution=args.target_resolution,
                               transform=transform,
                               num_sample=args.num_sample)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--img_path", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--annotation_path", type=str) 
    parser.add_argument("--model_name", default="blip2_opt", type=str)
    parser.add_argument("--model_type", default="pretrain_opt2.7b", type=str)

    args = parser.parse_args()
    
    main(args)