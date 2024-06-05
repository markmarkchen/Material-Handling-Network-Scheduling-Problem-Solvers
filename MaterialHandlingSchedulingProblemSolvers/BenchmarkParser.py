from collections import namedtuple, defaultdict
import numpy as np
from tqdm import tqdm

operation = namedtuple("operation", ["Job", "Option", "Order", "Node", "From", "To", "pt"])
site_operation = namedtuple("site_operation", ["Job", "Option", "Order", "Inbound_Node", "Inbound_pd", "Outbound_Node", "Outbound_pd", "pt"])
trans_site = namedtuple("trans_site", ["pt","capacity"])

def not_in_path(path, node):
    for n,p in path:
        if n == node:
            return False
    return True

def find_all_paths(node_outPD, PD2nextPD, all_paths, path, target_node, target_pd):
    current_node, current_pd = path[-1]
    if current_node == target_node:
        assert current_pd != target_pd
        all_paths += (path + [(target_node, target_pd)]),
        return
    for pd in node_outPD[current_node]:
        if len(path) == 1 and pd == current_pd:
            find_all_paths(node_outPD, PD2nextPD, all_paths, path+[PD2nextPD[(current_node, pd)]], target_node, target_pd)
        if not_in_path(path, PD2nextPD[(current_node, pd)][0]):
        # if (current_node, pd) not in path and PD2nextPD[(current_node, pd)] not in path:
            find_all_paths(node_outPD, PD2nextPD, all_paths, path+[(current_node, pd), PD2nextPD[(current_node, pd)]], target_node, target_pd)

def Find_All_Paths(initial_node, initial_pd, target_node, target_pd, node_outPD, PD2nextPD):
    all_paths = []
    find_all_paths(node_outPD, PD2nextPD, all_paths, [(initial_node, initial_pd)], target_node, target_pd)
    return all_paths

def path_to_time(path, node_from_to, site_pt):
    time = 0
    for i in range(1, len(path)):
        n1,pd1 = path[i-1]
        n2,pd2 = path[i]
        if n1 == n2:
            time += node_from_to[n1][pd1][pd2]
        else:
            time += site_pt[(n1,pd1,n2,pd2)][0]
    return time

class MHSP_Benchmark:
    def __init__(self, data, max_num_jobs = 15):
        self.num_nodes = data["num_nodes"]
        self.node_from_to = data["node_from_to"]
        self.node_num_veh = data['num_node_veh']
        self.num_trans = data["num_trans_site"]
        self.num_jobs = data["num_jobs"]
        self.num_jobs = min(self.num_jobs, max_num_jobs)
        
        
        self.requests = data["jobs"][:self.num_jobs]
        self.trans_sites = data["trans_sites"]
        self.start_after = np.array([0]*self.num_jobs)
        self.InfCapacity = False
        
        self.node_init_pos = data["node_init_pos"]

        self.trans_site_info = {}
        for site in self.trans_sites:
            n1,s1,n2,s2,pt,cap = site
            self.trans_site_info[(n1,s1,n2,s2)] = trans_site(max(pt,1),cap)

        self.node_outPD = defaultdict(list)
        self.PD2nextPD = {}
        self.site_pt = {}
        for site in self.trans_sites:
            self.node_outPD[site[0]] += site[1],
            self.PD2nextPD[(site[0],site[1])] = (site[2],site[3])

        for s in self.trans_sites:
            self.site_pt[(s[0],s[1],s[2],s[3])] = (max(s[4],1),s[5])
            
        self.job_all_paths = []
        self.job_all_free_run_time = []
        for j, request in tqdm(enumerate(data["jobs"][:self.num_jobs]), total=self.num_jobs):
            n1,pd1,n2,pd2 = request
            
            all_paths = Find_All_Paths(n1, pd1, n2, pd2, self.node_outPD, self.PD2nextPD)
            time = np.array([path_to_time(path, self.node_from_to, self.site_pt) for path in all_paths])
            ids_ascending = np.argsort(time)
            
            self.job_all_paths += [all_paths[id] for id in ids_ascending],
            self.job_all_free_run_time += time[ids_ascending],
                    

    def GenerateOperations(self, min_optional_path = 3, max_optional_path = 100, factor = 2):
        num_jobs = self.num_jobs
        # all operations
        self.operations = []

        # all operations in a job
        self.jobs = []

        # number of options in a job
        self.job_num_opt = []

        # (number of operations) in an option of a job
        self.num_ops = []

        # all free-run time
        # all_free_run_time = []
        # free-run time
        self.free_run_time = []
        
        
        for j, request in enumerate(self.requests):
            n1,pd1,n2,pd2 = request
            
            min_time = self.job_all_free_run_time[j][0]
            for i in range(len(self.job_all_free_run_time[j])):
                if self.job_all_free_run_time[j][i] > factor*min_time:
                    break
            # i = len(time)
            id = min(min(max(i, min_optional_path), max_optional_path), len(self.job_all_free_run_time[j]))
            self.free_run_time += self.job_all_free_run_time[j][:id],
            self.job_num_opt += id,
            self.jobs += [],
            self.num_ops += [],
            
            options = []
            for i in range(id):
                path = self.job_all_paths[j][i]
            # for path in all_paths[ids]:
                ops = []
                if path[0][0] != path[1][0]:
                # if path[0][1] in node_outPD[path[0][0]]:
                    path = [path[0]]+path
                for i in range(0, len(path)-1, 2):
                    ops += (path[i][0], path[i][1], path[i+1][1]),
                if len(path) % 2 == 1:
                    ops += (path[-1][0], path[-1][1], path[-1][1]),
                options += ops,

            for o, option in enumerate(options):
                ops = []
                self.num_ops[-1] += len(option)*2-1,
                for op_i, op in enumerate(option):
                    n, f, t = op
                    # assert n < num_nodes and f < num_pd and t < num_pd
                    ops += operation(j, o, op_i*2, n, f, t, self.node_from_to[n][f][t]),
                        
                for i in range(1,len(option)):
                    n1,_,t1 = option[i-1]
                    n2,f2,_ = option[i]
                    assert (n1,t1,n2,f2) in self.trans_site_info
                    ops += site_operation(j, o, 2*i-1, n1, t1, n2, f2, self.trans_site_info[(n1,t1,n2,f2)].pt),
                
                self.jobs[-1] += ops,
                self.operations.extend(ops)
    # jobs += ops,

        self.jobs = [[sorted(ops, key=lambda op:(op.Order)) for ops in opt] for opt in self.jobs]
        self.end_before = np.array([s+f.max()*20 for f,s in zip(self.free_run_time, self.start_after)])


        self.node_ops = [[] for _ in range(self.num_nodes)]
        self.site_ops = defaultdict(list)
        
        for i, job in enumerate(self.jobs):
            for ops in job:
                for op in ops:
                    if isinstance(op, operation):
                        self.node_ops[op.Node] += op,
                    if isinstance(op, site_operation):
                        self.site_ops[(op.Inbound_Node, op.Inbound_pd, op.Outbound_Node, op.Outbound_pd)] += op,

        self.node_ops = [sorted(ops, key=lambda p:(p.Job)) for ops in self.node_ops]
        
    def GenerateOperations_special(self, min_optional_path = 3, max_optional_path = 100, factor = 2, random_one = False):
        num_jobs = self.num_jobs
        # all operations
        self.operations = []

        # all operations in a job
        self.jobs = []

        # number of options in a job
        self.job_num_opt = []

        # (number of operations) in an option of a job
        self.num_ops = []

        # all free-run time
        # all_free_run_time = []
        # free-run time
        self.free_run_time = []
        
        
        for j, request in enumerate(self.requests):
            n1,pd1,n2,pd2 = request
            
            min_time = self.job_all_free_run_time[j][0]
            for i in range(len(self.job_all_free_run_time[j])):
                if self.job_all_free_run_time[j][i] > factor*min_time:
                    break
            # i = len(time)
            id = min(min(max(i, min_optional_path), max_optional_path), len(self.job_all_free_run_time[j]))
            if random_one:
                id = np.random.randint(0,id,1)
                self.free_run_time += self.job_all_free_run_time[j][id[0]],
            else:
                self.free_run_time += self.job_all_free_run_time[j][:id],
            self.job_num_opt += len(id),
            self.jobs += [],
            self.num_ops += [],
            
            options = []
            if not random_one:
                id = range(id)
            
            for i in id:
                path = self.job_all_paths[j][i]
            # for path in all_paths[ids]:
                ops = []
                if path[0][0] != path[1][0]:
                # if path[0][1] in node_outPD[path[0][0]]:
                    path = [path[0]]+path
                for i in range(0, len(path)-1, 2):
                    ops += (path[i][0], path[i][1], path[i+1][1]),
                if len(path) % 2 == 1:
                    ops += (path[-1][0], path[-1][1], path[-1][1]),
                options += ops,

            for o, option in enumerate(options):
                ops = []
                self.num_ops[-1] += len(option)*2-1,
                for op_i, op in enumerate(option):
                    n, f, t = op
                    # assert n < num_nodes and f < num_pd and t < num_pd
                    ops += operation(j, o, op_i*2, n, f, t, self.node_from_to[n][f][t]),
                        
                for i in range(1,len(option)):
                    n1,_,t1 = option[i-1]
                    n2,f2,_ = option[i]
                    assert (n1,t1,n2,f2) in self.trans_site_info
                    ops += site_operation(j, o, 2*i-1, n1, t1, n2, f2, self.trans_site_info[(n1,t1,n2,f2)].pt),
                
                self.jobs[-1] += ops,
                self.operations.extend(ops)
    # jobs += ops,

        self.jobs = [[sorted(ops, key=lambda op:(op.Order)) for ops in opt] for opt in self.jobs]
        self.end_before = np.array([s+f.max()*20 for f,s in zip(self.free_run_time, self.start_after)])


        self.node_ops = [[] for _ in range(self.num_nodes)]
        self.site_ops = defaultdict(list)
        
        for i, job in enumerate(self.jobs):
            for ops in job:
                for op in ops:
                    if isinstance(op, operation):
                        self.node_ops[op.Node] += op,
                    if isinstance(op, site_operation):
                        self.site_ops[(op.Inbound_Node, op.Inbound_pd, op.Outbound_Node, op.Outbound_pd)] += op,

        self.node_ops = [sorted(ops, key=lambda p:(p.Job)) for ops in self.node_ops]