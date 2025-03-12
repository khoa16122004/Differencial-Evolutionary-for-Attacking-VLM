import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os
import clip
import random
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def img_2_cap(model, image):
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
    def __init__(self, annotations_file, image_dir, target_dir, transform=transform, num_sample=1000):
        with open(annotations_file, "r") as f:
            lines = [line.strip().split("\t") for line in f.readlines()]
            self.file_names = [line[0] for line in lines][:num_sample]
            self.gt_txts = [line[1] for line in lines][:num_sample]
            self.tar_txts = [line[2] for line in lines][:num_sample]
        
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform

        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.file_names[idx])
        gt_txt = self.gt_txts[idx]
        tar_txt = self.tar_txts[idx]
        target_path = os.path.join(self.target_dir, self.file_names[idx])

        image = Image.open(image_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        # image_processed = vis_processors["eval"](image)
        # target_image_processed = vis_processors["eval"](target_image)
        # text_processed  = txt_processors["eval"](class_text_all[original_tuple[1]])
        if self.transform:
            image = self.transform(image)
            target_image = self.transform(target_image)
        
        return image, gt_txt, image_path, target_image, tar_txt, target_path


class Fitness:
    def __init__(self, image, model, pop_size, c_tar, c_clean, clip_model, sigma, alpha):
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
        
    @torch.no_grad()
    def encode_text(self, txt):
        token = clip.tokenize(txt).cuda()
        txt_embedding = self.clip_model.encode_text(token)
        txt_embedding = txt_embedding / txt_embedding.norm(dim=1, keepdim=True)
        
        return txt_embedding
    
    def benchmark(self, pop):
        image_advs = self.image + self.alpha * pop.reshape((self.pop_size, self.image.shape[1], self.image.shape[2], self.image.shape[3]))
        image_advs = torch.clamp(image_advs, 0., 1.)
        c_advs = img_2_cap(self.model, image_advs)
        c_adv_embeddings = self.encode_text(c_advs)

        adv_tar_sim = torch.sum(self.c_tar_embedding * c_adv_embeddings, dim=1)
        adv_clean_sim = torch.sum(self.c_clean_embedding * c_adv_embeddings, dim=1)
        fitness_ = adv_tar_sim - adv_clean_sim
        return fitness_