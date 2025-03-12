import torch
import clip
import random
import numpy as np
from tqdm import tqdm

def DE_Attack(image, pop_size, fitness, sigma, F, CR, max_iter, alpha):
    
    b, c, w, h = image.shape # 1 x c x w x h
    
    dim = c * w * h
    pop = (torch.rand((pop_size, dim)).cuda() * 2 - 1) * sigma # popsize x dim
    # print("pop shape: ", pop.shape)
    score = fitness.benchmark(pop)
    # print("score shape: ", score.shape)
    for _ in tqdm(range(max_iter)):
        r1, r2, r3 = [], [], []
        for i in range(pop_size):
            r1_, r_2, r_3 = np.random.choice([idx for idx in range(pop_size) if idx != i], size=3, replace=False) # not duplicated sample
            r1.append(r1_)
            r2.append(r_2)
            r3.append(r_3)
        
        x1, x2, x3 = pop[r1], pop[r2], pop[r3]
        v = x1 + F * (x2 - x3)
        v = torch.clamp(v, -sigma, sigma)
        
        j_random = np.random.randint(0, dim, size=pop_size)
        mask = torch.rand((pop_size, dim)).cuda() < CR
        mask[torch.arange(pop_size), j_random] = True
        u = torch.where(mask, v, pop)
        new_score = fitness.benchmark(u)

        improved = new_score > score
        score[improved] = new_score[improved]
        pop[improved] = u[improved]
        if score.max() >= 0:
            break

    
    best_idx = torch.argmax(score)
    best_solution = pop[best_idx]
    best_score = score[best_idx]
    best_adv_image = image + alpha * best_solution.reshape((image.shape[1], image.shape[2], image.shape[3]))
    best_adv_image = torch.clamp(best_adv_image, 0., 1.)
    
    return best_adv_image, best_score