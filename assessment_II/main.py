import numpy as np
from utils.models import MyTSP, EvolutionaryTSP, AntColonyTSP
from utils.common import calc_custom, calc_distance

def main(file, method = MyTSP, f_heuristic = None):
    print("---", method.__name__, "---")
    tsp = method(f_heuristic)
    tsp.set_matrix(file)
    best = tsp.fit()
    with open(file.replace('input', 'output').replace('csv', 'txt'), "w") as f:
        f.write("           Path : " + "-".join([str(i) for i in best[-1]]) + '\n')
        f.write("Total distances : " + f"{calc_distance(tsp.m, best[1]):.1f}")
    print("Solved:", file)
    print("SOLUTION=", best)
    print("------")

if __name__ == "__main__":
    methods = [MyTSP, EvolutionaryTSP, AntColonyTSP]
    otp_for = [calc_distance, calc_custom]
    main(file="input/I20.csv", method=methods[0], f_heuristic=otp_for[1])
