import torch
import clip
import random
import numpy as np
from tqdm import tqdm
from utils import putText, seed_everything

seed_everything(22520691)

def DE_pertubation_estimation_attack(image, pop_size, fitness, sigma, F, CR, max_iter, alpha):
    
    b, c, w, h = image.shape # 1 x c x w x h
    
    dim = c * w * h
    pop = (torch.rand((pop_size, dim)).cuda() * 2 - 1) * sigma # popsize x dim
    score, c_advs = fitness.pertubation_benchmark(pop)
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
        new_score, c_advs = fitness.pertubation_benchmark(u)

        improved = new_score > score
        score[improved] = new_score[improved]
        pop[improved] = u[improved] 
        pop[improved] = u[improved]

        best_current_index = torch.argmax(score)
        best_current_score = score[best_current_index]
        best_current_c_adv = c_advs[best_current_index]
        
        print("Best score: ", best_current_score)
        print("Best c_adv: ", best_current_c_adv)
        
        
        
        if score.max() >= 0:
            break

    
    best_idx = torch.argmax(score)
    best_solution = pop[best_idx]
    best_score = score[best_idx]
    best_adv_image = image + alpha * best_solution.reshape((image.shape[1], image.shape[2], image.shape[3]))
    best_adv_image = torch.clamp(best_adv_image, 0., 1.)
    
    return best_adv_image, best_score

def DE_text_in_attack(image, pop_size, fitness, F, CR, max_iter, location_change_interval, transform, text_in):
    dim = 6
    
    w, h = image.size
    pop = torch.rand((pop_size, dim)).cuda()
    position = (random.randint(0, int(w * 0.9)), random.randint(0, int(h * 0.9)))
    best_fitness = 0
    best_position = position
    
    score, c_advs = fitness.text_in_benchmark(pop, position)
    for iter_ in tqdm(range(max_iter)):
        if (iter_ + 1) % location_change_interval == 0:
            position = (random.randint(0, int(w * 0.8)), random.randint(0, int(h * 0.8)))
        
        r1, r2, r3 = [], [], []
        for i in range(pop_size):
            r1_, r_2, r_3 = np.random.choice([idx for idx in range(pop_size) if idx != i], size=3, replace=False) # not duplicated sample
            r1.append(r1_)
            r2.append(r_2)
            r3.append(r_3)
        
        x1, x2, x3 = pop[r1], pop[r2], pop[r3]
        v = x1 + F * (x2 - x3)
        v = torch.clamp(v, 0, 1)
        
        j_random = np.random.randint(0, dim, size=pop_size)
        mask = torch.rand((pop_size, dim)).cuda() < CR
        mask[torch.arange(pop_size), j_random] = True
        u = torch.where(mask, v, pop)
        new_score, c_advs = fitness.text_in_benchmark(u, position)
        
        improved = new_score > score
        score[improved] = new_score[improved]
        pop[improved] = u[improved]
        current_best_fitness = score.max()
        print("Fitness: ", current_best_fitness)

        
        if best_fitness < current_best_fitness:
            best_fitness = current_best_fitness
            best_position = position
        # if score.max() >= 0:
        #     break

    
    best_idx = torch.argmax(score)
    best_solution = pop[best_idx]
    print("best_solution: ", best_solution)
    angle, fontsize, R, G, B, alpha = best_solution[0] * 360, best_solution[1] * 15 + 10, best_solution[2] * 255, best_solution[3] * 255, best_solution[4] * 255, best_solution[5] * 100
    best_score = score[best_idx]
    best_adv_image = putText(image, best_position, transform, text_in, [angle], [fontsize], [R], [G], [B], [alpha])
    
    return best_adv_image, best_score