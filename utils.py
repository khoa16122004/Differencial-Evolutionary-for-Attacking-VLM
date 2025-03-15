import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os
import clip
import random
from PIL import Image, ImageDraw, ImageFont

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    


def img_2_cap(model, image):
    seed_everything(22520691)

    image_ = image.clone()
    samples = {"image": image_}
    caption = model.generate(samples, use_nucleus_sampling=True)
    return caption

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    img = img / 255.0
    return img.to(dtype=torch.get_default_dtype())

normalize = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]
)

inverse_normalize = torchvision.transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])
transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
                        torchvision.transforms.Resize(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
                        torchvision.transforms.Lambda(lambda img: to_tensor(img))])


class ImageCaptionDataset(Dataset):
    def __init__(self, annotations_file, image_dir, target_dir, target_resolution=224, transform=transform, num_sample=1000):
        with open(annotations_file, "r") as f:
            lines = [line.strip().split("\t") for line in f.readlines()]
            self.file_names = [line[0] for line in lines][:num_sample]
            self.gt_txts = [line[1] for line in lines][:num_sample]
            self.tar_txts = [line[2] for line in lines][:num_sample]
        
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform
        self.target_resolution = target_resolution

        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.file_names[idx])
        gt_txt = self.gt_txts[idx]
        tar_txt = self.tar_txts[idx]
        target_path = os.path.join(self.target_dir, self.file_names[idx])

        image_pil = Image.open(image_path).convert("RGB").resize((self.target_resolution, self.target_resolution))
        target_image_pil = Image.open(target_path).convert("RGB").resize((self.target_resolution, self.target_resolution))

        # image_processed = vis_processors["eval"](image)
        # target_image_processed = vis_processors["eval"](target_image)
        # text_processed  = txt_processors["eval"](class_text_all[original_tuple[1]])
        image = self.transform(image_pil)
        target_image = self.transform(target_image_pil)
        
        return image_pil, target_image_pil, image, gt_txt, image_path, target_image, tar_txt, target_path


class Fitness:
    def __init__(self, image_pil, image, 
                 text_in, model, 
                 pop_size, c_tar, 
                 c_clean, clip_model, 
                 sigma, alpha, transform):
        self.image = image
        self.c_tar = c_tar
        self.c_clean = c_clean
        self.clip_model = clip_model
        self.model = model
        self.c_tar_embedding = self.encode_text(c_tar)
        self.c_clean_embedding = self.encode_text(c_clean)
        self.sigma = sigma
        self.alpha = alpha
        self.pop_size = pop_size
        self.transform = transform
        self.image_pil = image_pil
        self.text = text_in
        
        
        
    @torch.no_grad()
    def encode_text(self, txt):
        token = clip.tokenize(txt).cuda()
        txt_embedding = self.clip_model.encode_text(token)
        txt_embedding = txt_embedding / txt_embedding.norm(dim=1, keepdim=True)
        
        return txt_embedding
    
    def pertubation_benchmark(self, pop):
        image_advs = self.image + self.alpha * pop.reshape((self.pop_size, self.image.shape[1], self.image.shape[2], self.image.shape[3]))
        image_advs = torch.clamp(image_advs, 0., 1.)
        c_advs = img_2_cap(self.model, image_advs)
        c_adv_embeddings = self.encode_text(c_advs)
        adv_tar_sim = torch.sum(self.c_tar_embedding * c_adv_embeddings, dim=1)
        tar_clean_sim = torch.sum(self.c_clean_embedding * self.c_tar_embedding, dim=1)
        print("tar clean sim: ", tar_clean_sim)
        print("adv tar sim: ", adv_tar_sim)
        fitness_ = adv_tar_sim - tar_clean_sim
        return fitness_, c_advs
    
    def text_in_benchmark(self, pop, position):
        # candidate = [angle, font_size, R, G, B, alpha]
        # bounds = [[0, 360], [10, 25], [0, 1], [0, 1], [0, 1], [0, 1]]
        angles = pop[:, 0] * 360
        font_sizes = pop[:, 1] * 15 + 10
        Rs = pop[:, 2] * 255
        Gs = pop[:, 3] * 255
        Bs = pop[:, 4] * 255
        alphas = pop[:, 5] * 100

        image_advs = putText(self.image_pil, position, self.transform, self.text, angles, font_sizes, Rs, Gs, Bs, alphas)
        c_advs = img_2_cap(self.model, image_advs)
        c_adv_embeddings = self.encode_text(c_advs)

        adv_tar_sim = torch.sum(self.c_tar_embedding * c_adv_embeddings, dim=1)
        adv_clean_sim = torch.sum(self.c_clean_embedding * self.c_tar_embeddings, dim=1)
        
        fitness_ = adv_tar_sim - adv_clean_sim
        return fitness_

def putText(image_pil, position, transform, text, angles, font_sizes, Rs, Gs, Bs, alphas, font_path='arial.ttf'):
    images = []
    for angle, fontsize, R, G, B, alpha in zip(angles, font_sizes, Rs, Gs, Bs, alphas):
        img = image_pil.copy().convert("RGBA")
        txt = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt)
        
        try:
            font = ImageFont.truetype(font_path, int(fontsize.item()))
        except:
            font = ImageFont.load_default()
        
        text_size = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]
        text_height += int(fontsize.item() * 0.3)  # Add padding for descenders
        
        text_img = Image.new("RGBA", (text_width, text_height), (255, 255, 255, 0))
        draw_text = ImageDraw.Draw(text_img)
        draw_text.text((0, 0), text, font=font, fill=(int(R.item()), int(G.item()), int(B.item()), int(alpha.item())))
        
        text_img = text_img.rotate(angle.item(), expand=True)
        pos = (position[0] - text_img.size[0] // 2, position[1] - text_img.size[1] // 2)
        img.paste(text_img, pos, text_img)
        images.append(transform(img))
        # torchvision.utils.save_image(transform(img), f"angle={angle}_fontsize={fontsize}_color={R}_{G}_{B}_{alpha}.png")
    return torch.stack(images, dim=0).cuda()    