# %%
import networkx as nx
import numpy as np
from numpy.random import randint
from tqdm import tqdm
import json
from numba import njit, prange
from itertools import combinations
from tqdm.auto import tqdm 
import yaml
import argparse
from collections import defaultdict
import os
import pprint
import shutil
from scipy.spatial.distance import cdist


# %%

# %%
def topk(input, k, axis=None, ascending=False):
    input = input.copy()
    if not ascending:
        input *= -1
    ind = np.argsort(input, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind, axis=axis) 
    return ind, val

def create_from_to(num_pd, max_from_to):
    xy = np.random.rand(num_pd, 2)
    dist = cdist(xy, xy, 'euclidean')
    dist = dist/np.sqrt(2)*max_from_to
    return np.ceil(dist).astype(np.int32)
    
    # np.random.rand(num_pd, 2)
    # w = randint(0, max_from_to)
    # h = max_from_to - w

    # x = randint(0,w+1, num_pd)
    # y = randint(0,h+1, num_pd)

    # # all-pair distance
    # from_to = np.zeros((num_pd, num_pd), dtype=np.int_)
    # for i in range(num_pd):
    #     for j in range(num_pd):
    #         from_to[i,j] = abs(x[i]-x[j])+abs(y[i]-y[j])
    #         from_to[j,i] = from_to[i,j]
    # return from_to
    
@njit(fastmath=True)
def check_tri(a,b,c):
    if a > b+c:
        return False
    elif a < abs(b-c):
        return False
    return True

@njit(fastmath=True)
def check_from_to(from_to, max_from_to):
    dim = from_to.shape[0]
    if (from_to <= max_from_to).sum() != dim*dim:
        return False
    for i in range(dim):
        for j in range(i+1,dim):
            for k in range(j+1,dim):
                if not check_tri(from_to[i,j], from_to[j,k], from_to[k,i]):
                    return False
                
    return True

def not_in_path(path, node):
    for n,p in path:
        if n == node:
            return False
    return True

def find_all_paths(node_outPD, PD2nextPD, all_paths, path, target_node, target_pd):
    current_node, current_pd = path[-1]
    if current_node == target_node:
        # assert current_pd != target_pd
        all_paths += (path + [(target_node, target_pd)]),
        return
    for pd in node_outPD[current_node]:
        if len(path) == 1 and pd == current_pd:
            find_all_paths(node_outPD, PD2nextPD, all_paths, path+[PD2nextPD[(current_node, pd)]], target_node, target_pd)
        elif not_in_path(path, PD2nextPD[(current_node, pd)][0]):
        # if (current_node, pd) not in path and PD2nextPD[(current_node, pd)] not in path:
            find_all_paths(node_outPD, PD2nextPD, all_paths, path+[(current_node, pd), PD2nextPD[(current_node, pd)]], target_node, target_pd)

def Find_All_Paths(initial_node, initial_pd, target_node, target_pd, node_outPD, PD2nextPD):
    all_paths = []
    find_all_paths(node_outPD, PD2nextPD, all_paths, [(initial_node, initial_pd)], target_node, target_pd)
    return all_paths

    # %%
def path_to_time(path, node_from_to, site_pt):
    time = 0
    for i in range(1, len(path)):
        n1,pd1 = path[i-1]
        n2,pd2 = path[i]
        if n1 == n2:
            time += node_from_to[n1][pd1,pd2]
        else:
            time += site_pt[(n1,pd1,n2,pd2)][0]
    return time
# %%

def create_a_graph(config):
    num_nodes = config["num_nodes"]
    num_pd = config["num_pd"]
    num_trans_site = config["num_trans_site"]
    max_trans_capacity = config["max_trans_capacity"]
    max_trans_pt = config["max_trans_pt"]
    
    
    trans_count = num_trans_site
    num_node_pd = randint(1, num_pd+1, num_nodes)
    num_node_pd_sum = num_node_pd.sum()
    if num_node_pd_sum < num_trans_site*2+2:
        num_node_pd[np.random.choice(num_nodes, 2, replace=False)] += 1
        for i in range(num_trans_site*2-num_node_pd_sum):
            num_node_pd[randint(0, num_nodes)] +=1
            # num_node_pd_sum += 1

    num_node_pd.sort()
    num_node_pd = num_node_pd[::-1]

    assert num_node_pd.sum() >= num_trans_site*2+2
    remain_pd = num_node_pd.copy()


    G = nx.MultiDiGraph()
    G.add_node(0)

    for i in range(num_nodes-1):
        G.add_node(i+1)
        
        node = np.random.randint(i+1)
        while G.degree[node]+1 > num_node_pd[node]:
            node = np.random.randint(i+1)
        # node = np.argmax(remain_pd[:i+1])
        assert G.degree[node]+1 <= num_node_pd[node]

        pt = int(randint(1, max_trans_pt+1))
        cap = int(randint(1, max_trans_capacity+1))
        n1, n2 = np.random.choice([int(node), i+1], 2, replace=False)
        G.add_edge(int(n1), int(n2), pt=pt, capacity=cap)
        remain_pd[node] -= 1
        remain_pd[i+1] -= 1
        # assert remain_pd[node] >= 1
        # assert remain_pd[i+1] >= 1
        
        # node = np.argmax(remain_pd[:i+1])
        # assert G.degree[node]+1 <= num_node_pd[node]

        # pt = int(randint(max_trans_pt+1))
        # cap = int(randint(1, max_trans_capacity+1))
        # G.add_edge(i+1, int(node), pt=pt, capacity=cap)
        # remain_pd[node] -= 1
        trans_count -= 1
        
    # remain_pd = num_node_pd - np.array([G.degree[i] for i in range(num_nodes)])
    assert remain_pd.sum() >= trans_count*2+2
    trial = 0
    for i in range(trans_count):
        ind = np.random.choice(num_nodes, 2, replace=False)
        fnode, tnode = ind
        trial = 0
        while G.degree[fnode]+1 > num_node_pd[fnode] or G.degree[tnode]+1 > num_node_pd[tnode]:
            ind = np.random.choice(num_nodes, 2, replace=False)
            fnode, tnode = ind
            if trial > 10000:
                break
        if trial > 10000:
            break
        # ind, val = topk(remain_pd,2)
        # if val[0] < 1 or val[1] < 1:
        #     print(trans_count-i,remain_pd)
        assert G.degree[fnode]+1 <= num_node_pd[fnode] and G.degree[tnode]+1 <= num_node_pd[tnode]
        remain_pd[fnode] -= 1
        remain_pd[tnode] -= 1
        pt = int(randint(1, max_trans_pt+1))
        cap = int(randint(1, max_trans_capacity+1))
        if np.random.rand() < 0.5:
            G.add_edge(int(fnode), int(tnode), pt=pt, capacity=cap)
        else:
            G.add_edge(int(tnode), int(fnode), pt=pt, capacity=cap)
    return G, num_node_pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="benchmark config.yml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    num_networks = config["num_networks"]
    num_nodes = config["num_nodes"]
    num_pd = config["num_pd"]
    num_trans_site = config["num_trans_site"]
    num_jobs = config["num_jobs"]
    max_node_vehicle = config["max_node_vehicle"]
    max_from_to = config["max_from_to"]
    
    max_trans_capacity = config["max_trans_capacity"]
    max_trans_pt = config["max_trans_pt"]
    
    max_start_after = config["max_start_after"]
    flexible = config["flexible"]
    
    node_init_pos = config["node_init_pos"]

   
    assert num_trans_site >= (num_nodes-1)
    assert num_nodes*num_pd >= 2*num_trans_site+2
    
    FOLDER_network =f"Thesis Benchmark (Reachability not Guaranteed)/Network N{num_nodes}S{num_trans_site}PD{num_pd}C{max_trans_capacity}"
    os.makedirs(FOLDER_network, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(FOLDER_network,os.path.basename(args.config)))
    
    for net in tqdm(range(num_networks), position=0):
        G, num_node_pd = create_a_graph(config)
    
        assert np.alltrue([G.degree[i] <= num_node_pd[i] for i in range(num_nodes)])

        # create from-to tables
        node_from_to =[[]]*num_nodes
        for i in range(num_nodes):
            node_from_to[i] = create_from_to(num_node_pd[i], int(np.sqrt(num_node_pd[i])*max_from_to))
            assert check_from_to(node_from_to[i], int(np.sqrt(num_node_pd[i])*max_from_to))
        
        trans_sites = []
        for node in range(num_nodes):
            trans_sites += np.random.choice(num_node_pd[node], G.degree[node], replace=False).tolist(),

        nodes2trans = defaultdict(list)

        # create sites
        site_pd = []
        sites = []
        for edge in G.edges(data=True):
            from_pd = trans_sites[edge[0]].pop()
            to_pd = trans_sites[edge[1]].pop()
            edge[2]["from_pd"] = int(from_pd)
            edge[2]["to_pd"] = int(to_pd)
            edge_data = G.get_edge_data(edge[0],edge[1])[0]
            sites += (edge[0], from_pd, edge[1],to_pd, edge_data["pt"], edge_data["capacity"]),
            site_pd += [(edge[0], from_pd), (edge[1],to_pd)]
            nodes2trans[(edge[0], edge[1])] += (from_pd,to_pd),
        
        node_outPD = defaultdict(list)
        PD2nextPD = {}
        for site in sites:
            node_outPD[site[0]] += site[1],
            PD2nextPD[(site[0],site[1])] = (site[2],site[3])
        site_pt = {s[:4]:s[4:] for s in sites}
    
    
        # save network
        network_id = [int(s.split("-")[1]) for s in os.listdir(FOLDER_network) if s.startswith("network")]
        if len(network_id) > 0:
            network_id = max(network_id)+1
        else:
            network_id = 1
        os.makedirs(f"{FOLDER_network}/network-{network_id}", exist_ok=True)
        FOLDER = f"{FOLDER_network}/network-{network_id}"
        
        G.graph["node"] = {"shape": "circle", "fontsize":"25"}
        G.graph["edge"] = {"arrowsize": "0.5", "splines": "true"}
        G.graph["graph"] = {"scale": "1", "splines": "spline","rankdir": "LR","ratio": "0.5", "overlap":"false"}
        A = nx.nx_agraph.to_agraph(G)  # convert to a graphviz graph

        max_degree = (max([G.degree[n] for n in range(num_nodes)])-2)
        for node in G.nodes():
            A.get_node(node).attr['label'] = f"{node+1}"
            A.get_node(node).attr['width'] = 0.75 + 1.5*(G.degree[node]-2)/max_degree
        
        for edge in A.edges():
        # for e in G.edges(data=True):
            # edge = A.get_edge(e[0], e[1])
            edge.attr['label'] = f"{edge.attr['capacity']}"
            # edge.attr["weight"] = 1/(e[2]["pt"]+1)
            edge.attr['penwidth'] = (10*(max_trans_pt-int(edge.attr['pt']))/max_trans_pt)+0.5
            # edge.attr['penwidth'] = e[2]['capacity']
            edge.attr['color'] = "grey"
            edge.attr["taillabel"] = f"{int(edge.attr['from_pd'])+1}"
            edge.attr["headlabel"] = f"{int(edge.attr['to_pd'])+1}"
        A.draw(os.path.join(FOLDER,"Graph.pdf"), prog="dot")
        A.draw(os.path.join(FOLDER,"Graph.dot"), prog="dot")
        text = json.dumps(nx.node_link_data(G), indent=2)[:-2]+',\n'
        num_node_veh = randint(1,max_node_vehicle+1, num_nodes).tolist()


        data = {"num_nodes": num_nodes,
                # "num_pd": num_pd, 
                "num_trans_site": num_trans_site, 
                # "num_jobs": num_jobs,
                # "max_node_vehicle": max_node_vehicle,
                # "max_trans_capacity": max_trans_capacity,
                "flexible": flexible,
                "node_init_pos": node_init_pos,
                }

        text += json.dumps(data, indent=2)[1:-2]+',\n'
        text += f'  "num_node_veh":\n'+ "\n".join([f"    {line}" for line in pprint.pformat(num_node_veh).split("\n")]) + ",\n"
        text += f'  "num_pds_in_node":\n'+ "\n".join([f"    {line}" for line in pprint.pformat(num_node_pd.tolist()).split("\n")]) + ",\n"
        text += f'  "node_from_to":\n'+ "\n".join([f"    {line}" for line in pprint.pformat([n.tolist() for n in node_from_to]).split("\n")]) + ",\n"
        text += f'  "trans_sites":\n'+  "\n".join([f"    {line}" for line in pprint.pformat([[s for s in site] for site in sites]).split("\n")])+ ",\n"
        # text += f'  "input_nodes":\n'+ "\n".join([f"    {line}" for line in pprint.pformat(input_nodes).split("\n")]) + ",\n"
        # text += f'  "output_nodes":\n'+ "\n".join([f"    {line}" for line in pprint.pformat(output_nodes).split("\n")]) + ",\n"
        text = text[:-2] + "\n}"
        with open(f"{FOLDER}/Network config.json", "w") as f:
                f.write(text)
        with open(f"{FOLDER}/Network config.json") as f:
            G_ = nx.node_link_graph(json.loads(f.read()))
            assert nx.utils.graphs_equal(G_, G)
            # print(f"Graph {network_id} saved: {'Success!' if  else 'Failed!'}")

        
        if max_start_after == 0:
            start_after = np.zeros(num_jobs, dtype=np.int32).tolist()
        else:
            start_after = randint(0, max_start_after, num_jobs).tolist()

        # text = '{\n  \"num_jobs\": ' + f"{num_jobs},\n  " +  '"jobs\": {\n'
        text = '{\n  \"num_jobs\": ' + f"{num_jobs},\n"
        text += f'  "start_after":\n'+  "\n".join([f"    {line}" for line in pprint.pformat(start_after).split("\n")])+ ",\n"
        text += '  "jobs\": \n'
        data["jobs"] = {}
        Requests = []
        for j in tqdm(range(num_jobs), position=1):
            Find = False
            while not Find:
                n1,n2 = np.random.choice(num_nodes, 2, replace=True)
                pd1 = np.random.choice(num_node_pd[n1])
                pd2 = np.random.choice(num_node_pd[n2])
                if n1 == n2 and pd1 == pd2:
                    continue
                if (n1, pd1) in site_pd and G.degree[n1] != 1:
                    continue
                if (n2, pd2) in site_pd and G.degree[n2] != 1:
                    continue
                all_paths = Find_All_Paths(n1, pd1, n2, pd2, node_outPD, PD2nextPD)
                if len(all_paths) < 1:
                    continue
                Requests += [n1,pd1,n2,pd2],
                Find = True
            
        if flexible:
            text += "\n".join([f"      {line}" for line in pprint.pformat(Requests).split("\n")])+ "\n}"
            pass
        else:
            for request in Requests:
                all_paths = []
                find_all_paths(node_outPD, PD2nextPD, all_paths, [(request[0],request[1])], request[2], request[3])
                path_time = np.array([path_to_time(p, node_from_to, site_pt) for p in all_paths])
                fastest_path_id = np.argmin(path_time)
                ops = []
                path = all_paths[fastest_path_id]
                
                for i in range(0, len(path), 2):
                    ops += (path[i][0], path[i][1], path[i+1][1]),
                
                if len(path) % 2 == 1:
                    ops += (path[-1][0], path[-1][1], path[-1][1]),
                
                ops = [ops]
                text += f'    "job {0}":\n' + "\n".join([f"        {line}" for line in pprint.pformat(ops, width=20).split("\n")])
                text += "  }\n}"
                
        with open(f"{FOLDER}/{'Flexible ' if flexible else ''}J{num_jobs}.json", "w") as f:
                f.write(text)
    