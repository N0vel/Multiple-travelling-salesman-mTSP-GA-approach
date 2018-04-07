import numpy as np
import random
import math
import matplotlib.pyplot as plt
'''
Genetical path finding
Finds locally best ways from L service centers with [M0, M1, ..., ML] engineers
through atms_number ATMs and back to their service center
'''
def fitness_pop(population, engi_work):
    fitness_result = np.zeros(population_size)
    for i in range(population_size):
        fitness_result[i] = fitness(population[i], engi_work[i])
    return fitness_result

def fitness(sequence, work):
    sum_dist = np.zeros(work.size)
    for j in range(work.size):
        mat_path = np.zeros((dist.shape[0], dist.shape[1]))
        if j != work.size-1:
            path = sequence[np.sum(work[:j]):np.sum(work[:j+1])]
        else:
            path = sequence[np.sum(work[:j]):]
        print(path)
        print('---------------')
        print(sequence)
        print('--- ---  ---')

        if path.size != 0:
            for v in range(path.size):
                if v == 0:
                    mat_path[engineers[j], path[v]] = 1
                else:
                    mat_path[path[v - 1] + service_centers, path[v]] = 1
            mat_path = mat_path * dist
            sum_dist[j] = (np.sum(mat_path) + dist[engineers[j], path[-1]]) / velocity + repair_time * path.size
    return np.max(sum_dist)

def birth_prob(fitness_result):
    birth_prob = np.abs(fitness_result - np.max(fitness_result))
    birth_prob = birth_prob / np.sum(birth_prob)
    return birth_prob

def mutate(creat, engi):
    pnt_1 = random.randint(0, creat.size-1)
    pnt_2 = random.randint(0, creat.size-1)
    if random.random() < mut_1_prob:
        creat[pnt_1], creat[pnt_2] = creat[pnt_2], creat[pnt_1]
    if random.random() < mut_2_prob and pnt_1 != pnt_2:
        if pnt_1 > pnt_2:
            pnt_1, pnt_2 = pnt_2, pnt_1
        creat[pnt_1:pnt_2+1] = np.flip(creat[pnt_1:pnt_2+1], axis=0)
    if random.random() < mut_3_prob:
        engi -= 1
        engi[engi < 0] = 0
        while(np.sum(engi) != atms_number):
            engi[random.randint(0, engi.size-1)] += 1
    return creat, engi

def crossover_mutation(population, birth_prob, engi_work):
    new_population = np.zeros((population_size, atms_number))
    new_engi_work = np.zeros((population_size, engineers.size))
    for i in range(0, population_size, 2):
        pair = np.zeros(2, dtype=int)
        pair[0] = np.random.choice(np.arange(0,population_size,step=1),p=birth_prob)
        while pair[1] != pair[0]:
            pair[1] = np.random.choice(np.arange(0,population_size,step=1),p=birth_prob)
        engi_1 = engi_work[pair[0]]
        engi_2 = engi_work[pair[1]]
        parent_1 = population[pair[0]]
        parent_2 = population[pair[1]]
        creat_1 = np.zeros(parent_1.size)
        creat_2 = np.zeros(parent_2.size)
        cross_point_1 = random.randint(0, len(parent_1) - 1)
        cross_point_2 = random.randint(0, len(parent_2) - 1)
        node_1 = parent_1[cross_point_1:]
        node_2 = parent_2[cross_point_2:]
        w = 0
        for v in range(creat_1.size):
            if parent_2[v] not in node_1:
                creat_1[v] = parent_2[v]
            else:
                creat_1[v] = node_1[w]
                w += 1
        w = 0
        for v in range(len(creat_2)):
            if parent_1[v] not in node_2:
                creat_2[v] = parent_1[v]
            else:
                creat_2[v] = node_2[w]
                w += 1
        # mutations
        creat_1, engi_1_child = mutate(creat_1, engi_1)
        creat_2, engi_2_child = mutate(creat_2, engi_2)
        # children
        child_1 = []
        #################################################################################################################################3
        for v in range(engi_1.size):
            child_1.append(creat_1[np.sum(engi_1_child[:v]):np.sum(engi_1_child[:v+1])])
        child_2 = []
        for v in range(engi_2.size):
            child_2.append(creat_2[np.sum(engi_2_child[:v]):np.sum(engi_2_child[:v+1])])
        together = [(np.array(child_1), engi_1_child), (np.array(child_2), engi_2_child), (population[pair[0]], engi_1),
                    (population[pair[1]], engi_2)]
        fit = np.array([fitness(creature, work) for creature, work in together])
        fit = fit.argsort()
        # if two_opt_search:
        #     new_population.append(two_opt(together[fit[0]]))
        #     new_population.append(two_opt(together[fit[1]]))
        new_population[i] = together[fit[0]]
        new_population[i+1] = together[fit[1]]
    return new_population

def plot_paths(paths):
    plt.clf()
    plt.title('Best path overall')
    for v in range(service_centers):
        plt.scatter(points_locations[v, 0], points_locations[v, 1], c='r')
    for v in range(atms_number):
        plt.scatter(points_locations[v+service_centers, 0], points_locations[v+service_centers, 1], c='b')
    for v in range(len(paths)):
        if len(paths[v]) != 0:
            path_locations = points_locations[service_centers:]
            path_locations = path_locations[np.array(paths[v])]
            path_locations = np.vstack((points_locations[engineers[v]], path_locations))
            path_locations = np.vstack((path_locations, points_locations[engineers[v]]))
            plt.plot(path_locations[:, 0], path_locations[:, 1])
    plt.show()
    plt.pause(0.0001)

# Bank parameters
atms_number = 25         # ATM quantity
service_centers = 3     # service centers quantity
velocity = 100             # 100 / hour
repair_time = 0         # 0.5 hour
max_engi = 3              # maximum number of engineers in one service center
min_engi = 1

# genetic parameters
population_size = 500    # population size (even number!)
generations = 1000       # population's generations
mut_1_prob = 0.4         # prob of replacing together two atms in combined path
mut_2_prob = 0.6      # prob of reversing the sublist in combined path
mut_3_prob = 0.8     # probability of changing the length of paths for engineers
two_opt_search = False  # better convergence, lower speed for large quantity of atms


# seed
np.random.seed(2)
random.seed(1)
plt.ion()
engineers = []

# number_of_engies = np.random.randint(min_engi, max_engi, service_centers)
for i in range(service_centers):
    for j in range(random.randint(min_engi, max_engi)):
        engineers.append(i)
engineers = np.array(engineers)
print('Engineers: {}'.format(engineers))
dist = np.zeros((atms_number+service_centers, atms_number))
points_locations = np.random.randint(0, 100, (service_centers+atms_number)*2).reshape((service_centers+atms_number, 2))
for i in range(dist.shape[0]):
    for j in range(dist.shape[1]):
        dist[i, j] = math.sqrt((points_locations[i, 0] - points_locations[j + service_centers, 0]) ** 2 +
                               (points_locations[i, 1] - points_locations[j + service_centers, 1]) ** 2)
        if j+service_centers == i:
            dist[i, j] = 0
# random population creation
population = np.random.permutation(np.tile(np.arange(0, atms_number, step=1), (population_size, 1)))
for i in range(population.shape[0]):
    np.random.shuffle(population[i,:])
engi_work = np.random.multinomial(atms_number, np.ones(engineers.size)/engineers.size, size=population_size)
fitness_result = fitness_pop(population, engi_work)
best_mean_creature_result = np.mean(fitness_result)
best_creature_result = np.min(fitness_result)
best_selection_prob = birth_prob(fitness_result)
selection_prob = best_selection_prob
# plot_paths(population[np.argmin(fitness_result)])
for i in range(generations):
    new_population, new_engi_work = crossover_mutation(population, selection_prob, engi_work)
    fitness_result = fitness_pop(new_population, engi_work)
    mean_creature_result = np.mean(fitness_result)
    if np.min(fitness_result) < best_creature_result:
        # plot_paths(population[np.argmin(fitness_result)])           ######
        best_creature_result = np.min(fitness_result)
    if mean_creature_result < best_mean_creature_result:
        best_mean_creature_result = mean_creature_result
        best_selection_prob = birth_prob(fitness_result)
        selection_prob = best_selection_prob
        population = new_population[:]
        engi_work = new_engi_work[:]
    print('Mean population time: {0} Best time: {1}'.format(best_mean_creature_result, best_creature_result))
plt.ioff()
plt.show()