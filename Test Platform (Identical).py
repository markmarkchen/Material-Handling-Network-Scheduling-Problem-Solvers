import pandas as pd
import time
import json
import os
from tqdm import tqdm
import sys
from MaterialHandlingNetworkSchedulingProblemSolvers import MHSP_Benchmark, CP_Solver, IP_Solver, DE_Solver
import pickle
import csv

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
        
def get_next_benchmark_special(FOLDER):
    NETWORKS = ["Network N12S48PD50C3"]
    network = ["network-1"]
    for networks, NETWORK in zip(NETWORKS, network):
        JOB_FILE = [s for s in os.listdir(os.path.join(FOLDER,networks,NETWORK)) if s.startswith("Special")][0]
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
        Benchmark.GenerateOperations(min_optional_path = 3, max_optional_path = 1, factor=2)
    else:
        Benchmark.GenerateOperations(min_optional_path = 3, max_optional_path = 30, factor=2)
    
    Benchmark.InfCapacity = inf
    text = "fixed" if fixed else "flexible"
    text += " inf" if inf else " finite"
    print(get_time(), text)
    
    cp_best = 0
    
    try:
            CP = CP_Solver(Benchmark)
            msol = CP.Solve()
        # if not os.path.exists(os.path.join(path,f"CP {text}.pkl")):
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
            s = msol.export_as_json_string()
            s = json.loads(s)
            s.update({"time":mdl.solve_details.time})
            s = json.dumps(s)
            with open(os.path.join(path,f"IP {text} sol.json"), "w") as f:
                f.write(s)
            print("IP", mdl.objective_value, mdl.solve_details.time)
            # print(mdl.print_solution())
            if mdl.solve_details.time < 60 and cp_best != 0 and cp_best != round(mdl.objective_value):
                print("Error!!!!!!!!!!!!!!!!!!!!!!!!")
                # raise
    except:
        pass
    DE = DE_Solver(Benchmark)
    best, msol = DE.Solve(5, 1000000, 10)
    try:
        print("DE Best", best)
        if round(best) < cp_best:
            print("Error!!!!!!!!!!!!!!!!!!!!!!!!")
            # sys.exit()
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

def solve_special(path, Benchmark, net_size, fixed, inf, random_one = False, id = None, params=None):
    if fixed:
        if not random_one:
            Benchmark.GenerateOperations(min_optional_path = 3, max_optional_path = 1, factor=2)
        else:
            Benchmark.GenerateOperations_special(min_optional_path = 3, max_optional_path = 30, factor=2, random_one = True)
    else:
        Benchmark.GenerateOperations(min_optional_path = 3, max_optional_path = 30, factor=2)
    
    Benchmark.InfCapacity = inf
    text = "fixed" if fixed else "flexible"
    text += " random one" if random_one else ""
    text += " inf" if inf else " finite"
    print(get_time(), text)
    
    text_params = "fixed" if fixed else "flexible"
    text_params += " inf" if inf else " finite"
    cp_best = 0
    
    try:
        # if not os.path.exists(os.path.join(path,f"Special CP {text}.pkl")):
            CP = CP_Solver(Benchmark)
            msol = CP.Solve()
            print("CP", round(msol.get_objective_value()), msol.get_solve_time())
            if random_one:
                with open(os.path.join(path,f"Special CP {text}-{id}.pkl"), "bw") as f:
                    pickle.dump(msol, f)
            else:
                with open(os.path.join(path,f"Special CP {text}.pkl"), "bw") as f:
                    pickle.dump(msol, f)
            if msol.get_solve_time() < 60:
                cp_best = round(msol.get_objective_value())
                # print(msol.print_solution())
    except:
        pass
    
    try:
        # if not os.path.exists(os.path.join(path,f"Special IP {text} sol.json")):
            IP = IP_Solver(Benchmark)
            msol, mdl = IP.Solve()
            s = msol.export_as_json_string()
            s = json.loads(s)
            s.update({"time":mdl.solve_details.time})
            s = json.dumps(s)
            if random_one:
                with open(os.path.join(path,f"Special IP {text}-{id} sol.json"), "w") as f:
                    f.write(s)
            else:
                with open(os.path.join(path,f"Special IP {text} sol.json"), "w") as f:
                    f.write(s)
            print("IP", round(mdl.objective_value), mdl.solve_details.time)
            # print(mdl.print_solution())
            if mdl.solve_details.time < 60 and cp_best != 0 and cp_best != round(mdl.objective_value):
                print("Error!!!!!!!!!!!!!!!!!!!!!!!!")
                # raise
    except:
        pass
    
    try:
        # if not os.path.exists(os.path.join(path,f"Special DE {text}.pkl")):
            DE = DE_Solver(Benchmark)
            if fixed:
                if params is not None:
                    print("DE params:", params[net_size+" "+text_params])
                    best, msol = DE.Solve(5, 1000000, 2, *params[net_size+" "+text_params])
                else:
                    best, msol = DE.Solve(5, 1000000, 2)
            else:
                if params is not None:
                    print("DE params:", params[net_size+" "+text_params])
                    best, msol = DE.Solve(5, 1000000, 20, *params[net_size+" "+text_params])
                else:
                    best, msol = DE.Solve(5, 1000000, 20)
            # best, msol = DE.Solve(5, 1000000, 10)
            print("DE Best", round(best))
            if round(best) < cp_best:
                print("Error!!!!!!!!!!!!!!!!!!!!!!!!")
                # sys.exit()
            if random_one:
                with open(os.path.join(path,f"Special DE {text}-{id}.pkl"), "bw") as f:
                    pickle.dump(msol, f)
            else:    
                with open(os.path.join(path,f"Special DE {text}.pkl"), "bw") as f:
                    pickle.dump(msol, f)
    except:
        pass
    
def problem_type_parser(s):
    s1,s2 = s.split("/")
    text = "fixed" if s1=="1" else "flexible"
    text += " inf" if s2 != "n" else " finite"
    return text
            
if __name__ == "__main__":
    params = None
    if os.path.exists("Taguchi_results.csv"):
        with open("Taguchi_results.csv") as f:
            reader = csv.reader(f)
            next(reader)
            params = {row[0]+" "+problem_type_parser(row[1]):(int(row[2]), float(row[3]), float(row[4])) for row in reader}
            
    FOLDER_network = "Thesis Benchmark (Reachability not Guaranteed)"
    FOLDER_results = "Thesis Benchmark (Reachability not Guaranteed) Results (Special Case)"
    for data, networks, NETWORK in get_next_benchmark_special(FOLDER_network):
    # for data, networks, NETWORK in get_next_benchmark_tmp(FOLDER_network):
    # for data, networks, NETWORK in get_next_benchmark(FOLDER_network):
        Benchmark = MHSP_Benchmark(data, max_num_jobs = 5)
        path = os.path.join(FOLDER_results,networks,NETWORK)
        os.makedirs(path, exist_ok=True)
        # with open("Benchmark.pkl", "bw") as f:
        #     pickle.dump(Benchmark, f)
        # print(path)        
        solve_special(path, Benchmark, "Large", fixed=True, inf=False, params=params)
        for i in range(10):
            solve_special(path, Benchmark, "Large", fixed=True, inf=False, random_one = True, id = i+1, params=params)
        solve_special(path, Benchmark, "Large", fixed=False, inf=False, params=params)
        
        
