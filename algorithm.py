import torch
import clip
import random
import numpy as np
from tqdm import tqdm

def DE_Attack(image, pop_size, fitness, sigma, F, CR, max_iter):
    
    b, c, w, h = image.shape # 1 x c x w x h
    
    dim = c * w * h
    pop = (torch.random((pop_size, dim)).cuda() * 2 - 1) * sigma # popsize x dim
    score = fitness.benchmark(pop)
    for _ in range(max_iter):
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
        mask = torch.random((pop_size, dim)) < CR
        mask[torch.arrange(pop_size), j_random] = True
        u = torch.where(mask, v, pop)
        
        new_fitness = fitness.benchmark(u)
        improved = new_fitness > fitness
        fitness[improved] = new_fitness[improved]
        pop[improved] = u[improved]
    
    best_idx = torch.argmax(fitness)
    best_solution = pop[best_idx]
    best_fitness = fitness[best_idx]
    best_adv_image = image + best_solution.reshape((pop_size, image.shape[1], image.shape[2], image.shape[3]))
    best_adv_image = torch.claim(best_adv_image, 0., 1.)
    
    return best_adv_image, best_fitness