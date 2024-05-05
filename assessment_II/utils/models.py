from utils.common import create_matrix_csv, calc_distance, distance_to_probability
from collections import defaultdict
import numpy as np
from random import choices, randint, random, choice, seed
seed(1111)

class BaseTSP:
    def __init__(self, heuristic=None):
        self.heuristic = calc_distance if heuristic is None else heuristic

    def set_matrix(self, f):
        """
        Set the distance matrix.

        Args:
        - f: Either a filename (string) or a matrix (numpy array).
        """
        self.m = create_matrix_csv(f) if isinstance(f, str) else f
        self.n = len(self.m) - 2

    def get_threshold(self, m, n_population=None):
        """Calculate thresholds based on distance matrix for population initialization"""
        dist_non_zeros = m[m > 0]
        nbin = int(len(m)*0.7)
        counts, values = np.histogram(dist_non_zeros, bins=nbin)
        j_max = np.argmax(counts)
        j_low, j_up = 0, j_max+(nbin-j_max)//3
        if n_population is None: n_population = j_up - j_low
        thresholds = [threshold for threshold in np.linspace(values[j_low], values[j_up], n_population)]
        return thresholds

    def local_search(self, path, start_node=4):
        """
        Perform local search on the given path using 2-opt algorithm.

        Parameters:
        - path: list
        List representing the path.
        - start_node: int
            Index of the starting node for local search.

        Returns:
        Optimized path after local search.
        """
        stable, best = False, self.heuristic(self.m, path)
        while not stable:
            stable = True
            for i in range(start_node, self.n):
                for j in range(i + 1, self.n+1):
                    candidate = path[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    sub = candidate[i-1:j+2]
                    if not np.all(self.m[sub[:-1], sub[1:]] != 0):
                        continue
                    length_candidate = self.heuristic(self.m, candidate)
                    if best > length_candidate:
                        path, best = candidate, length_candidate
                        stable = False
        return path

    def next_with_threshold(self, **kwargs):
        """Choose the next node based on threshold."""
        cur, threshold, nexts, path  = kwargs['cur'], kwargs['threshold'], kwargs['nexts'], kwargs['path']
        costs = np.abs(self.m[cur, nexts] - threshold)
        j_min = np.argmin(costs)
        next = nexts[j_min]
        path.append(next)
        return next, path

    def next_with_prob_threshold(self, **kwargs):
        """Choose the next node based on threshold."""
        cur, threshold, nexts, path  = kwargs['cur'], kwargs['threshold'], kwargs['nexts'], kwargs['path']
        costs = np.abs(self.m[cur, nexts] - threshold)
        idxs = np.argsort(costs)[:3]
        distances = costs[idxs]
        probs = distance_to_probability(distances)
        j_min = choices(idxs, weights=probs)[0]
        next = nexts[j_min]
        path.append(next)
        return next, path


    def next_with_heuristic(self, **kwargs):
        """Select the next node based on cost."""
        nexts, path = kwargs['nexts'], kwargs['path']
        costs = [self.heuristic(self.m, path + [next]) for next in nexts]
        j_min = np.argmin(costs)
        next = nexts[j_min]
        path.append(next)
        return next, path

    def search_path(self, fnext, k, init_path=[0], **kwargs):
        """
        Search for a path through the TSP using a specified function.

        Parameters:
        - fnext: function
            Function to use for selecting the next node.
        - k: int
            Number of nodes in the path.
        - init_path: list, optional
        Initial path. Default is [0].

        Returns:
        Path through the TSP.
        """
        cur, path = init_path[-1], init_path
        forbidden = defaultdict(list)
        count = 0
        while len(path) < k:
            _k = tuple(path) # Convert path to a tuple for easy indexing
            nexts = np.argwhere(self.m[cur, :] != 0).flatten()
            nexts = [next for next in nexts
                    if (next not in path and next != self.n+1 and next not in forbidden[_k])
                    or (len(path) == self.n+1 and next == self.n+1)]

            # If there are no valid neighbors
            if len(nexts) <= 0:
                cur = path[-2] # Move back to the previous node
                forbidden[_k[:-1]].append(path[-1])
                path = path[:-1]
                count += 1
                if count > 1000: return path # Prevent infinite loops
                continue
            kwargs['nexts']=nexts
            kwargs['path']=path
            kwargs['cur']=cur
            cur, path = fnext(**kwargs)
        return path

    def fit(self):
        pass

class MyTSP(BaseTSP):
    def __init__(self, heuristic, k_percent = 0.2):
        """
        Parameters:
        - n_init_paths (int): number of initial paths.
        - (float) the percentage of initial vertices traversed
        """
        super().__init__(heuristic)
        self.k_percent = k_percent

    def initialize(self):
        """
        Initialize the heads for the search.

        Returns:
        - List of heads (list): List of initial path heads.
        """
        self.k_head = int(self.n*self.k_percent) # the number of initial vertices traversed
        thresholds = self.get_threshold(self.m)
        heads = [self.search_path(fnext=self.next_with_prob_threshold, k=self.k_head,
                                    init_path=[0], threshold=t) for t in thresholds]
        heads = list(set(tuple(head) for head in heads))
        return heads

    def fit(self):
        heads = self.initialize()
        result = []
        for head in heads:
            path = self.search_path(fnext=self.next_with_heuristic, k=self.n+2, init_path=list(head))
            if path[-1] != self.n+1: continue
            new_path = self.local_search(path, start_node=self.k_head)
            result.append([self.heuristic(self.m, new_path), new_path])
        result = min(result)
        return result


class EvolutionaryTSP(BaseTSP):
    def __init__(self, heuristic,
                    n_population=100,
                    n_epoch=100,
                    pathnament_size=4,
                    mutation_rate=0.5,
                    crossover_rate=0.9):
        super().__init__(heuristic)
        self.n_population = n_population
        self.n_epoch = n_epoch
        self.pathnament_size = pathnament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def initialize(self):
        """
        Initialize the population using thresholds.

        Returns:
        - population (list): List of individuals in the population.
        """
        thresholds = self.get_threshold(self.m, self.n_population)
        population = []
        for threshold in thresholds:
            path = self.search_path(fnext=self.next_with_threshold, k=self.n+2,
                                    init_path=[0], threshold=threshold)
            if path[-1] != self.n+1: continue
            population.append([self.heuristic(self.m, path), path])
        if len(population) < self.n_population:
            population = population + choices(population, k=self.n_population-len(population))
        return population

    def cross_over(self, parent1, parent2, k):
        """
        Perform crossover operation between two parents.

        Args:
        - parent1 (list): First parent.
        - parent2 (list): Second parent.
        - k (int): Crossover point.

        Returns:
        - child (list): Child produced by crossover operation.
        """
        child = parent1[:k]
        parent2 = parent2.copy()
        while len(child) < len(parent1)-1:
            nexts = np.where(self.m[child[-1]]!=0)[0]
            nexts = [next for next in nexts if next not in child and next < len(parent1)-1]
            if len(nexts) == 0:
                return parent1
            for j in parent2:
                if j in nexts:
                    child.append(j)
                    parent2.remove(j)
                    break
        return child + [len(parent1)-1]

    def mutation(self, child):
        """
        Perform mutation operation on the child.

        Args:
        - child (list): Child to be mutated.

        Returns:
        - mutated_child (list): Mutated child.
        """
        repeat = {}
        n = len(child)
        while True:
            k1 = randint(1, n - 3)
            k2 = randint(k1 + 1, n-2)
            while (k1, k2) in repeat:
                k1 = randint(1, n - 3)
                k2 = randint(k1 + 1, n-2)
            candidate = child[:]
            candidate[k1], candidate[k2] = candidate[k2], candidate[k1]
            sub = candidate[k1-1:k2+2]
            repeat[(k1, k2)] = True
            if not np.all(self.m[sub[:-1], sub[1:]] != 0):
                continue
            return candidate
        return child

    def fit(self):
        assert self.m is not None, "Matrix not set"
        population = self.initialize()
        for epoch in range(self.n_epoch):
            # selecting two of the best options we have (elitism)
            p1 = min(population)
            population.remove(p1)
            p2 = min(population)
            population.remove(p2)
            new_population = [p1, p2]

            for i in range(len(population) // 2):
                # CROSSOVER
                random_number = random()
                if random_number < self.crossover_rate:
                    parent1 = sorted(choices(population, k=self.pathnament_size))[0][1]
                    parent2 = sorted(choices(population, k=self.pathnament_size))[0][1]
                    k = randint(1, self.n)
                    child1 = self.cross_over(parent1, parent2, k)
                    child2 = self.cross_over(parent2, parent1, k)
                # If crossover not happen
                else:
                    child1 = choice(population)[1]
                    child2 = choice(population)[1]

                # MUTATION
                if random() < self.mutation_rate:
                    child1 = self.mutation(child1)
                    child2 = self.mutation(child1)

                new_population.append([self.heuristic(self.m, child1), child1])
                new_population.append([self.heuristic(self.m, child2), child2])

            population = new_population
            if epoch % 10 == 0:
                print(f"epoch {epoch}:", sorted(population)[0][0])

        return min(population)


class AntColonyTSP(BaseTSP):
    def __init__(self, heuristic, n_ant=50, n_epoch=100, alpha=1.0, beta=1.0,
                        rho=0.5, del_tau=1.0, k=2, is_local_search=True):
        """
        Initialize the Ant Colony Optimization for TSP.

        Parameters:
            n_ant (int): Number of ants.
            n_epoch (int): Number of epochs.
            alpha (float): Alpha parameter for pheromone influence.
            beta (float): Beta parameter for heuristic information influence.
            rho (float): Evaporation rate for pheromones.
            del_tau (float): Pheromone increment.
            k (int): Number of nodes to be selected initially.
            is_local_search (bool): Whether to apply local search after each iteration.
        """
        super().__init__(heuristic)
        self.k = k
        self.n_ant = n_ant
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.del_tau = del_tau
        self.is_local_search = is_local_search

    def initialize(self):
        """
        Initialize ants and pheromones.

        Returns:
            tuple: Tuple containing heads (starting nodes for each ant) and pheromones matrix.
        """
        thresholds = self.get_threshold(self.m, self.n_ant)
        heads = [self.search_path(fnext=self.next_with_threshold, k=self.k,
                                    init_path=[0], threshold=t) for t in thresholds]
        pheromones = np.zeros((self.n+2, self.n+2))
        for head in heads:
            for x, y in zip(head[:-1], head[1:]):
                pheromones[x,y] += self.del_tau
        return heads, pheromones

    def calc_prob(self, a, b, alpha, beta):
        """
        Calculate probability based on pheromones and heuristic information.

        Parameters:
            a (numpy.ndarray): Pheromones information.
            b (numpy.ndarray): Heuristic information.
            alpha (float): Alpha parameter for pheromone influence.
            beta (float): Beta parameter for heuristic information influence.

        Returns:
            numpy.ndarray: Calculated probabilities.
        """
        return (a**alpha + b**beta) / (a.sum()**alpha + b.sum()**beta)

    def next_with_pheromones(self, **kwargs):
        """
        Choose the next node based on pheromones.

        Parameters:
            **kwargs: Arbitrary keyword arguments containing 'cur', 'path', 'nexts', and 'pheromones'.

        Returns:
            tuple: Tuple containing the next node and updated path.
        """
        cur, path, nexts, pheromones = kwargs["cur"], kwargs["path"], kwargs["nexts"], kwargs["pheromones"]
        costs = np.array([1/self.heuristic(self.m, path + [next]) for next in nexts])
        next_prob = self.calc_prob(pheromones[cur, nexts], costs, self.alpha, self.beta)
        j_min = np.argmax(next_prob)
        next = nexts[j_min]
        path.append(next)
        pheromones[cur, next] += self.del_tau
        cur = next
        return next, path

    def fit(self):
        assert self.m is not None, "Matrix not set"

        heads, pheromones = self.initialize()
        elitist_epochs = []
        for epoch in range(self.n_epoch):
            elitist_ants = []
            for init_path in heads:
                nexts = np.argwhere(self.m[init_path[-1], :] != 0).flatten()
                next = choice([next for next in nexts if next not in init_path])
                init_path = init_path + [next]
                path = self.search_path(fnext=self.next_with_pheromones, k=self.n+2,
                                init_path=init_path, pheromones=pheromones)
                if path[-1] != self.n+1: continue
                elitist_ants.append([self.heuristic(self.m, path), path])

            # Update elite pheromones
            elitist = min(elitist_ants)
            if self.is_local_search:
                elitist=self.local_search(path=elitist[1], start_node=self.k+1)
                elitist = [self.heuristic(self.m, elitist), elitist]
            path = elitist[1]
            for i in range(self.n+1):
                cur, next = path[i], path[i + 1]
                pheromones[cur][next] += self.del_tau
            pheromones *= (1 - self.rho)
            elitist_epochs.append(elitist)


            if epoch % 10 == 0:
                print(f"epoch {epoch}:", elitist[0])

        return min(elitist_epochs)
