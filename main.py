import argparse

import torch.utils
from lavis.models import load_model_and_preprocess
from utils import *
from tqdm import tqdm
from algorithm import *

def main(args):
    seed_everything(22520691)
    # output
    os.makedirs(args.output_dir, exist_ok=True)
    # model image to text
    model, _, _ = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device="cuda")
    model.eval()

    # clip encode    
    clip_img_model_vitb32, preprocess = clip.load("ViT-B/32", device="cuda")

    # data
    data = ImageCaptionDataset(annotations_file=args.annotation_path,
                               image_dir=args.image_dir,
                               target_dir=args.target_dir,
                               target_resolution=args.target_resolution,
                               transform=transform,
                               num_sample=args.num_sample)
    
    with open(args.output_dir + ".txt", "w") as f:
        for i in tqdm(range(args.num_sample)):
            image_pil, target_image_pil, image, gt_txt, image_path, target_image, tar_txt, target_path = data[i]
            basename = os.path.basename(image_path)
            image = image.cuda()
            image = image.unsqueeze(0)
            
            # c_clean = img_2_cap(model, image)[0]
            # print("c_clean: ", c_clean)
            c_clean = gt_txt
            # print("target text: ", tar_txt)
            text_in = "dog"
            fitness = Fitness(image_pil=image_pil, 
                              image=image, 
                              text_in=text_in, 
                              model=model, 
                              pop_size=args.pop_size, 
                              c_tar="A dog playing with cat", 
                              c_clean=c_clean, 
                              clip_model=clip_img_model_vitb32, 
                              sigma=args.sigma, 
                              transform=transform)
            
            if args.method == "text_in":
                # bounds = [[0, 1], [10, 25], [0, 1], [0, 1], [0, 1], [0, 1]]
                image_adv, best_fitness = DE_text_in_attack(image_pil, args.pop_size,
                                                            fitness, args.F, args.CR,
                                                            args.max_iter, 
                                                            args.location_change_interval, 
                                                            transform, text_in)
            elif args.method == "pertubation":
                image_adv, best_fitness = DE_pertubation_estimation_attack(image, args.pop_size, fitness, args.sigma, args.F, args.CR, args.max_iter, args.alpha)
            
            adv_cap = img_2_cap(model, image_adv)[0]
            print("Adv cap: ", adv_cap)
            print("Best fitness: ", best_fitness)
            
            adv_sim = torch.sum(fitness.c_tar_embedding * fitness.encode_text(adv_cap), dim=1)
            clean_sim = torch.sum(fitness.c_clean_embedding * fitness.encode_text(adv_cap), dim=1)
            print("Adv sim: ", adv_sim)
            print("Clean sim: ", clean_sim)
            
            torchvision.utils.save_image(image_adv, os.path.join(args.output_dir, basename))
            f.write(f"{basename}\t{c_clean}\t{tar_txt}\t{adv_cap}\n")
            break
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--annotation_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--pop_size", type=int)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--F", type=float)
    parser.add_argument("--CR", type=float)
    parser.add_argument("--max_iter", type=int)
    parser.add_argument("--num_sample", type=int, default=1000)
    parser.add_argument("--model_name", default="blip2_opt", type=str)
    parser.add_argument("--model_type", default="pretrain_opt2.7b", type=str)
    parser.add_argument("--method", choices=['perturbation', 'text_in'], type=str)
    parser.add_argument("--location_change_interval", type=int, default=10)
    parser.add_argument("--target_resolution", type=int, default=224)
    args = parser.parse_args()
    main(args)
    

 

