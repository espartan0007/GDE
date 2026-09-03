import numpy as np
from scipy.stats import truncnorm
from scipy.stats import entropy

# ============================================
# DE Operators & Algorithms
# ============================================
def mutation_original(pop, F, bounds):
    pop_size, dim = pop.shape
    idxs = np.random.choice(pop_size, 3, replace=False)
    a, b, c = idxs
    mutant = pop[a] + F*(pop[b] - pop[c])
    return np.clip(mutant, bounds[:,0], bounds[:,1])




def crossover(parent, mutant, CR):
    dim = len(parent)
    cross = np.random.rand(dim) < CR
    cross[np.random.randint(dim)] = True
    return np.where(cross, mutant, parent)

def mutation_smooth(pop, F, bounds, smooth_term, gamma=0.2):
    pop_size, dim = pop.shape
    idxs = np.random.choice(pop_size, 5, replace=False)
    a, b, c ,d,e= idxs
    mutant = (1-gamma)*(pop[a] + F*(pop[b]-pop[c])+F*(pop[d]-pop[e])) + gamma*smooth_term[a]
    return np.clip(mutant, bounds[:,0], bounds[:,1])


def boundConstraint_custom(Xnew, lb, ub):

    out_of_bounds_upper = Xnew > ub
    out_of_bounds_lower = Xnew < lb
    dim = len(lb)

    if np.any(out_of_bounds_upper):
        random_range = ub[out_of_bounds_upper] - lb[out_of_bounds_upper]
        Xnew[out_of_bounds_upper] = lb[out_of_bounds_upper] + np.random.rand(np.sum(out_of_bounds_upper)) * random_range

    if np.any(out_of_bounds_lower):
        random_range = ub[out_of_bounds_lower] - lb[out_of_bounds_lower]
        Xnew[out_of_bounds_lower] = lb[out_of_bounds_lower] + np.random.rand(np.sum(out_of_bounds_lower)) * random_range
        
    return Xnew




def de_run(obj_fun, dim, bounds, pop_size=60, gens=1500, mode="original", schedule=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    F_ORIG = 0.8; F_SMOOTH = 0.3; CR = 0.9; gamma = 0.2; window = 5

    pop = np.random.uniform(bounds[:,0], bounds[:,1], (pop_size, dim))
    fitness = np.array([obj_fun(ind) for ind in pop])
    best_hist = []

    if mode == "original":
        for g in range(gens):
            for i in range(pop_size):
                mutant = mutation_original(pop, F_ORIG, bounds)
                trial = crossover(pop[i], mutant, CR)
                f = obj_fun(trial)
                if f < fitness[i]:
                    pop[i] = trial; fitness[i] = f
            best_hist.append(np.min(fitness))
    elif mode == "hybrid":
        history = [pop.copy()]
        for (phase, phase_gens) in schedule:
            for g in range(phase_gens):
                smooth_term = None
                if phase == "smooth":
                    hist_pop = np.array(history[-window:])
                    if len(hist_pop) > 0:
                         smooth_term = np.mean(hist_pop, axis=0)

                for i in range(pop_size):
                    if phase == "original":
                        mutant = mutation_original(pop, F_ORIG, bounds)
                    else:
                        if smooth_term is None or (np.isnan(smooth_term)).any():
                            mutant = mutation_original(pop, F_ORIG, bounds)
                        else:
                            mutant = mutation_smooth(pop, F_SMOOTH, bounds, smooth_term, gamma)


                    trial = crossover(pop[i], mutant, CR)
                    f = obj_fun(trial)

                    if f < fitness[i]:
                        pop[i] = trial; fitness[i] = f

                best_hist.append(np.min(fitness))
                history.append(pop.copy())

    return np.array(best_hist)



