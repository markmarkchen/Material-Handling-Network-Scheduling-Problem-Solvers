import pandas as pd
import time
import json
import os
from tqdm import tqdm
import sys
from MaterialHandlingSchedulingProblemSolvers import MHSP_Benchmark, CP_Solver, IP_Solver, DE_Solver
import pickle

def get_next_benchmark(FOLDER):
    NETWORKs = ["Network N30S100PD100C3"]
    # NETWORKs = os.listdir(FOLDER)
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
    NETWORKS = ["Network N5S10PD6C1","Network N8S20PD6C2"]
    network = ["network-14","network-28"]
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

def solve(path, Benchmark, fixed=False, inf=False, ext=False):
    if fixed:
        Benchmark.GenerateOperations(min_optional_path = 1, max_optional_path = 1, factor=1)
    else:
        Benchmark.GenerateOperations(min_optional_path = 3, max_optional_path = 30, factor=2)
    
    Benchmark.InfCapacity = inf
    text = "fixed" if fixed else "flexible"
    text += " inf" if inf else " finite"
    text += " pth-ext" if ext else ""
    print(get_time(), text)
    if not fixed:
        print("Number of paths:", Benchmark.job_num_opt)
    cp_best = 0
    
    # try:
    #     # if not os.path.exists(os.path.join(path,f"CP {text}.pkl")):
    #         CP = CP_Solver(Benchmark)
    #         msol = CP.Solve(3600, True)
    #         print("CP", round(msol.get_objective_value()), msol.get_solve_time())
    #         with open(os.path.join(path,f"CP {text}.pkl"), "bw") as f:
    #             pickle.dump(msol, f)
    #         if msol.get_solve_time() < 3600:
    #             cp_best = round(msol.get_objective_value())
    #             # print(msol.print_solution())
    # except:
    #     pass
    
    # try:
    #     # if not os.path.exists(os.path.join(path,f"IP {text} sol.json")):
    #         IP = IP_Solver(Benchmark)
    #         msol, mdl = IP.Solve()
    #         print("IP", round(mdl.objective_value), mdl.solve_details.time)
    #         s = msol.export_as_json_string()
    #         s = json.loads(s)
    #         s.update({"time":mdl.solve_details.time})
    #         s = json.dumps(s)
    #         if mdl.solve_details.time < 60 and cp_best != 0 and cp_best != round(mdl.objective_value):
    #             print("Error!!!!!!!!!!!!!!!!!!!!!!!!")
    #             with open(os.path.join(path,f"IP {text} sol error.json"), "w") as f:
    #                 f.write(s)
    #         else:
    #             with open(os.path.join(path,f"IP {text} sol.json"), "w") as f:
    #                 f.write(s)
                
    #         # print(mdl.print_solution())
    #             # raise
    # except:
    #     pass
    try:
        DE = DE_Solver(Benchmark)
        if fixed:
            best, msol = DE.Solve(5, 1000000, 2, 8, 0.1, 0.1)
        else:
            best, msol = DE.Solve(5, 10000000, 720, 8, 0.1, 0.1)
        print("DE Best", round(best))
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

            
if __name__ == "__main__":
    ext = True
    text = " pth-ext" if ext else ""
    FOLDER_network = "Thesis Benchmark (Reachability not Guaranteed)"
    FOLDER_results = "Thesis Benchmark (Reachability not Guaranteed) Results"
    # for data, networks, NETWORK in get_next_benchmark_tmp(FOLDER_network):
    for data, networks, NETWORK in get_next_benchmark(FOLDER_network):
        Benchmark = MHSP_Benchmark(data, max_num_jobs = 100, ext=ext)
        # print([len(f) for f in Benchmark.job_all_free_run_time])
        path = os.path.join(FOLDER_results,networks,NETWORK)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"Benchmark{text}.pkl"), "bw") as f:
            pickle.dump(Benchmark, f)
        print(path)       
        # solve(path, Benchmark, fixed=True, inf=True, ext=ext)
        # solve(path, Benchmark, fixed=True, inf=False, ext=ext)
        # solve(path, Benchmark, fixed=False, inf=True, ext=ext)
        solve(path, Benchmark, fixed=False, inf=False, ext=ext)    
