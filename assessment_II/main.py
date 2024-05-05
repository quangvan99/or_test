import numpy as np
from utils.models import BaseTSP, MyTSP, EvolutionaryTSP, AntColonyTSP
from utils.common import calc_custom, calc_distance, create_weight_matrix, create_matrix_csv
import time

# (for PART 1)
def find_a_feasible_path(file):
    """Find_a_feasible_path  (for PART 1)"""
    tsp = BaseTSP()
    m = create_matrix_csv(file)
    m = create_weight_matrix(m)
    tsp.set_matrix(m)
    path = tsp.search_path(fnext=tsp.next_with_threshold, k=len(m), init_tour=[0], threshold=0)
    return path

# (for solve_atsp)
def solve_atsp(file, method, f_heuristic = None):
    tsp = method(f_heuristic)
    tsp.set_matrix(file)
    best = tsp.fit()
    with open(file.replace('input', 'output').replace('csv', 'txt'), "w") as f:
        f.write("           Path : " + "-".join([str(i) for i in best[-1]]) + '\n')
        f.write("Total distances : " + f"{calc_distance(tsp.m, best[1]):.1f}")
    return best

if __name__ == "__main__":

    # (for PART 1)
    # path = find_a_feasible_path("input/I20.csv")
    # print("Path:", path)

    # (for solve atsp)
    methods = [MyTSP, EvolutionaryTSP, AntColonyTSP]
    otp_for = [calc_distance, calc_custom]
    method, f_heuristic = methods[0], otp_for[1]
    print("---", method.__name__, "---")
    for i in [20, 30, 40, 50, 60]:
        file=f"input/I{i}.csv"
        t1 = time.time()
        bests = [solve_atsp(file,method,f_heuristic) for _ in range(5)]
        t2 = time.time()
        print(f"Time of {file}:", (t2-t1)/5)
        print(f"Best of {file}:", min(bests))
