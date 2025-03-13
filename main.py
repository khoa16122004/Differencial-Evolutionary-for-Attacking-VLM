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
                               transform=transform,
                               num_sample=args.num_sample)
    
    with open(args.output_dir + ".txt", "w") as f:
        for i in tqdm(range(args.num_sample)):
            image_pil, image, gt_txt, image_path, target_image, tar_txt, target_path = data[i]
            basename = os.path.basename(image_path)
            image = image.cuda()
            image = image.unsqueeze(0)
            
            c_clean = img_2_cap(model, image)[0]
            # print("c_clean: ", c_clean)
            # print("target text: ", tar_txt)
            fitness = Fitness(image_pil, image, model, args.pop_size, tar_txt, c_clean, clip_img_model_vitb32, args.sigma, args.alpha)
            if args.attack_type == "text_in":
                image_adv, best_fitness = DE_text_in_attack(image, args.pop_size, fitness, args.sigma, args.F, args.CR, args.max_iter, args.alpha)
            elif args.attack_type == "pertubation_estimation":
                image_adv, best_fitness = DE_pertubation_estimation_attack(image, args.pop_size, fitness, args.sigma, args.F, args.CR, args.max_iter, args.alpha)
            
            adv_cap = img_2_cap(model, image_adv)[0]
            # print("Adv cap: ", adv_cap)
            # print("Best fitness: ", best_fitness)
            torchvision.utils.save_image(image_adv, os.path.join(args.output_dir, basename))
            f.write(f"{basename}\t{c_clean}\t{tar_txt}\t{adv_cap}\n")
            
            
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
    parser.add_argument("--method", choice=['perturbation', 'text_in'], type=str)
    args = parser.parse_args()
    main(args)
    

 

