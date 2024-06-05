import pandas as pd
import time
import json
import os
from tqdm import tqdm
import sys
from MaterialHandlingSchedulingProblemSolvers import MHSP_Benchmark, CP_Solver, IP_Solver, DE_Solver
import pickle
from itertools import product

OA = [[1,1,1],
      [1,2,2],
      [1,3,3],
      [2,1,2],
      [2,2,3],
      [2,3,1],
      [3,1,3],
      [3,2,1],
      [3,3,2]]

def get_next_benchmark(FOLDER):
    NETWORKs = os.listdir(FOLDER)
    NETWORKs = sorted([s for s in NETWORKs if s.startswith("Network")])
    print(NETWORKs)
    for networks in NETWORKs:
        for net in sorted(os.listdir(os.path.join(FOLDER,networks))):
            if net.startswith("network"):
            # if net == "network-11":
                NETWORK = net
                JOB_FILE = [s for s in os.listdir(os.path.join(FOLDER,networks,NETWORK)) if s.startswith("Flexible")][0]
                FILENAME = os.path.join(FOLDER,networks,NETWORK,JOB_FILE)
                print(FILENAME)
                with open(FILENAME) as f:
                    data = json.loads(f.read())
                with open(os.path.join(FOLDER,networks,NETWORK,"Network config.json")) as f:
                    data.update(json.loads(f.read()))
                yield data, networks, NETWORK

def get_next_benchmark_tmp(FOLDER):
    NETWORKS = ["Network N5S10PD20C1","Network N8S24PD40C2", "Network N12S48PD50C3"]
    network = ["network-1","network-1", "network-1"]
    for networks, NETWORK in zip(NETWORKS, network):
        JOB_FILE = [s for s in os.listdir(os.path.join(FOLDER,networks,NETWORK)) if s.startswith("Flexible")][0]
        FILENAME = os.path.join(FOLDER,networks,NETWORK,JOB_FILE)
        print(FILENAME)
        with open(FILENAME) as f:
            data = json.loads(f.read())
        with open(os.path.join(FOLDER,networks,NETWORK,"Network config.json")) as f:
            data.update(json.loads(f.read()))
        yield data, networks, NETWORK
            
def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def solve(path, Benchmark, fixed, inf):
    if fixed:
        Benchmark.GenerateOperations(min_optional_path = 1, max_optional_path = 1, factor=1)
    else:
        Benchmark.GenerateOperations(min_optional_path = 3, max_optional_path = 100, factor=5)
    
    Benchmark.InfCapacity = inf
    text = "fixed" if fixed else "flexible"
    text += " inf" if inf else " finite"
    print(get_time(), text)
    cp_best = 0
    
    try:
        # if not os.path.exists(os.path.join(path,f"CP {text}.pkl")):
            CP = CP_Solver(Benchmark)
            msol = CP.Solve()
            print("CP", msol.get_objective_value(), msol.get_solve_time())
            with open(os.path.join(path,f"CP {text}.pkl"), "bw") as f:
                pickle.dump(msol, f)
            if msol.get_solve_time() < 60:
                cp_best = round(msol.get_objective_value())
                # print(msol.print_solution())
    except:
        pass
    
    try:
        # if not os.path.exists(os.path.join(path,f"IP {text} sol.json")):
            IP = IP_Solver(Benchmark)
            msol, mdl = IP.Solve()
            print("IP", mdl.objective_value, mdl.solve_details.time)
            s = msol.export_as_json_string()
            s = json.loads(s)
            s.update({"time":mdl.solve_details.time})
            s = json.dumps(s)
            if mdl.solve_details.time < 60 and cp_best != 0 and cp_best != round(mdl.objective_value):
                print("Error!!!!!!!!!!!!!!!!!!!!!!!!")
                with open(os.path.join(path,f"IP {text} sol error.json"), "w") as f:
                    f.write(s)
            else:
                with open(os.path.join(path,f"IP {text} sol.json"), "w") as f:
                    f.write(s)
                
            # print(mdl.print_solution())
                # raise
    except:
        pass
    try:
        DE = DE_Solver(Benchmark)
        best, msol = DE.Solve(5, 1000000, 10)
        print("DE Best", best)
        if round(best) < cp_best:
            print("Error!!!!!!!!!!!!!!!!!!!!!!!!")
            # sys.exit()
            with open(os.path.join(path,f"DE {text} error.pkl"), "bw") as f:
                pickle.dump(msol, f)
        else:
            with open(os.path.join(path,f"DE {text}.pkl"), "bw") as f:
                pickle.dump(msol, f)
            
    except:
        pass
    
    # if not fixed:
    #     Benchmark.GenerateOperations(min_optional_path = 10, max_optional_path = 30, factor=3)
    #     Benchmark.InfCapacity = inf
    #     text = "fixed" if fixed else "flexible"
    #     text += " inf" if inf else " finite"
    
    #     print(get_time(), text, "extended")
        
    #     try:
    #         DE = DE_Solver(Benchmark)
    #         best, msol = DE.Solve(10, 100000, 10)
    #         print("DE Best", best)
    #         with open(os.path.join(path,f"DE {text} extended.pkl"), "bw") as f:
    #             pickle.dump(msol, f)
    #     except:
    #         pass
    
def test_param(Benchmark, fixed, pop_size, Mu, Cr):
    # Benchmark.GenerateOperations(min_optional_path = 3, max_optional_path = 100, factor=5)    
    # Benchmark.InfCapacity = False

    DE = DE_Solver(Benchmark)
    if fixed:
        best, msol = DE.Solve(5, 1000000, 2, pop_size, Mu, Cr)
    else:
        best, msol = DE.Solve(5, 1000000, 20, pop_size, Mu, Cr)
    print("DE Best", best, [r[:2] for r in msol])
    return msol
   
            
if __name__ == "__main__":
    FOLDER_network = "Thesis Benchmark (Reachability not Guaranteed)"
    FOLDER_results = "Thesis Benchmark (Reachability not Guaranteed) Results"
    os.makedirs(os.path.join(FOLDER_results, "Taguchi"), exist_ok=True)
    # for data, networks, NETWORK in get_next_benchmark_tmp(FOLDER_network):
    pop_sizes = [8, 64, 128]
    Mus = [0.001, 0.01, 0.1]
    Crs = [0.05, 0.1, 0.2]
    fixeds = [True, False]
    infs = [True, False]
    
    results = []
    for data, networks, NETWORK in get_next_benchmark_tmp(FOLDER_network):
        # data, networks, NETWORK = next(iter(get_next_benchmark_tmp(FOLDER_network)))
        print(networks, NETWORK)
        Benchmark = MHSP_Benchmark(data, max_num_jobs = 8)
        for fixed, inf in product(fixeds, infs):
            if fixed:
                Benchmark.GenerateOperations(min_optional_path = 1, max_optional_path = 1, factor=1)
            else:
                Benchmark.GenerateOperations(min_optional_path = 3, max_optional_path = 100, factor=5)
            
            Benchmark.InfCapacity = inf
            text = "fixed" if fixed else "flexible"
            text += " inf" if inf else " finite"
            print(get_time(), text)
            # cp_best = 0
            result = [networks, NETWORK, fixed, inf]
            for design in OA:
                print(design, pop_sizes[design[0]-1], Mus[design[1]-1], Crs[design[2]-1])
                # print()
                msol = test_param(Benchmark, fixed, pop_sizes[design[0]-1], Mus[design[1]-1], Crs[design[2]-1])
                result += (design, (pop_sizes[design[0]-1], Mus[design[1]-1], Crs[design[2]-1]), msol),
            results.append(result)
            # print(results)
    with open(os.path.join(FOLDER_results,"Taguchi", "Taguchi results.pkl"), "bw") as f:
        pickle.dump(results, f)
    # test_param(Benchmark)  
    
    
    # for data, networks, NETWORK in get_next_benchmark(FOLDER_network):
    #     Benchmark = MHSP_Benchmark(data, max_num_jobs = 8)
    #     # print([len(f) for f in Benchmark.job_all_free_run_time])
    #     path = os.path.join(FOLDER_results,networks,NETWORK)
    #     os.makedirs(path, exist_ok=True)
    #     with open(os.path.join(path, "Benchmark.pkl"), "bw") as f:
    #         pickle.dump(Benchmark, f)
    #     print(path)       
    #     solve(path, Benchmark, fixed=True, inf=True)
    #     solve(path, Benchmark, fixed=True, inf=False)
    #     solve(path, Benchmark, fixed=False, inf=True)
    #     solve(path, Benchmark, fixed=False, inf=False)    
