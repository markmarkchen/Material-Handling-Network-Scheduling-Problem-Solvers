from .BenchmarkParser import operation, site_operation, MHSP_Benchmark
import numpy as np
from itertools import combinations
from docplex.mp.model import Model
from docplex.mp.error_handler import DefaultErrorHandler, InfoLevel
import warnings

warnings.simplefilter("ignore")

class IP_Solver:
    def __init__(self, Beanchmark: MHSP_Benchmark) -> None:
        self.Benchmark = Beanchmark
        
    def Solve(self):
        model = Model("Flexible MHSP")
        num_jobs = self.Benchmark.num_jobs
        node_from_to = self.Benchmark.node_from_to
        trans_site_info = self.Benchmark.trans_site_info
        num_ops = self.Benchmark.num_ops
        num_nodes = self.Benchmark.num_nodes
        node_ops = self.Benchmark.node_ops
        site_ops = self.Benchmark.site_ops
        job_num_opt = self.Benchmark.job_num_opt
        jobs = self.Benchmark.jobs
        start_after = self.Benchmark.start_after
        end_before = self.Benchmark.end_before
        V = int(1e6)
        
        job_opt_op_id = [(k,l,t) for k in range(num_jobs) for l in range(len(num_ops[k])) for t in range(num_ops[k][l])]
        job_opt_id = [(k,l) for k in range(num_jobs) for l in range(len(num_ops[k]))]
        all_process_time = [[np.array([op.pt for op in opt]) for opt in jobs[j]] for j in range(num_jobs)]
        x_lb = lambda f: start_after[f[0]]+all_process_time[f[0]][f[1]][:f[2]].sum()
        x_ub = lambda f: end_before[f[0]]-(all_process_time[f[0]][f[1]][f[2]:].sum())
        
        # x = model.integer_var_dict(job_opt_op_id, lb=x_lb, name="x")
        x = model.integer_var_dict(job_opt_op_id, lb=x_lb, ub=x_ub, name="x")
        # print(x)
        z = model.binary_var_dict(job_opt_id, name="z")
        c_max = model.integer_var(0, name="makespan")

        # Definition of Cmax
        for k in range(num_jobs):
            for l in range(job_num_opt[k]):
                model.add_constraint(x[(k,l,num_ops[k][l]-1)] + all_process_time[k][l][-1] <= c_max + (1-z[(k,l)])*V)
                model.add_constraint(x[(k,l,num_ops[k][l]-1)] + all_process_time[k][l][-1] <= end_before[k] + (1-z[(k,l)])*V)

        # Objective function
        model.minimize(c_max)
        
        # Precedence constraints
        for k in range(num_jobs):
            for l in range(job_num_opt[k]):
                for t in range(num_ops[k][l]-1):
                    model.add_indicator(z[(k,l)], x[(k,l,t+1)] >= x[(k,l,t)]+all_process_time[k][l][t])
        
        # Select one option for each job
        for j in range(num_jobs):
            model.add_constraint(model.sum([z[(j,o)] for o in range(job_num_opt[j])]) == 1)
        
        node_d_i_j = [(d,i,j) for d in range(num_nodes) for i in range(len(node_ops[d])) for j in range(len(node_ops[d]))]
        w = model.binary_var_dict(node_d_i_j, name="w")
        empty_car = []
        for d in range(num_nodes):
            empty_car += [],
            for i in range(len(node_ops[d])):
                empty_car[-1] += [],
                for j in range(len(node_ops[d])):
                    empty_car[-1][-1] += node_from_to[d][node_ops[d][i].To][node_ops[d][j].From],

        for d in range(num_nodes):
            for i in range(len(node_ops[d])):
                j1,op1,o1 = node_ops[d][i].Job,node_ops[d][i].Option,node_ops[d][i].Order
                model.add_indicator(z[(j1,op1)], x[(j1,op1,o1)] >= node_from_to[d][self.Benchmark.node_init_pos][node_ops[d][i].From])
                for j in range(len(node_ops[d])):
                    j2,op2,o2 = node_ops[d][j].Job,node_ops[d][j].Option,node_ops[d][j].Order
                    
                    if i == j:
                        model.add_constraint(w[(d,i,j)] == 0)
                        continue
                    # last operation
                    if o1 == num_ops[j1][op1]-1:
                        model.add_indicator(w[(d,i,j)], x[(j2,op2,o2)] >= x[(j1,op1,o1)]+all_process_time[j1][op1][o1]+empty_car[d][i][j])
                        # model.add_if_then(z[(j1,op1)] + z[(j2,op2)] + w[(d,i,j)] == 3, x[(j2,op2,o2)] >= x[(j1,op1,o1)]+all_process_time[(j1,op1,o1)]+empty_car[d][i][j])
                    else:
                        model.add_indicator(w[(d,i,j)], x[(j2,op2,o2)] >= (x[(j1,op1,o1+1)]+empty_car[d][i][j]))
                        # model.add_if_then(z[(j1,op1)] + z[(j2,op2)] + w[(d,i,j)] == 3, x[(j2,op2,o2)] >= x[(j1,op1,o1+1)]+empty_car[d][i][j])
                    model.add_constraint(w[(d,i,j)] <= z[(j1,op1)])
                    model.add_constraint(w[(d,i,j)] <= z[(j2,op2)])

        # %%
        for d in range(num_nodes):
            for i in range(len(node_ops[d])):
                op = node_ops[d][i]
                model.add_constraint(model.sum([w[(d,i,j)] for j in range(len(node_ops[d]))]) <= z[(op.Job,op.Option)])
                model.add_constraint(model.sum([w[(d,j,i)] for j in range(len(node_ops[d]))]) <= z[(op.Job,op.Option)])
            model.add_if_then((model.sum([z[(op.Job,op.Option)] for op in node_ops[d]])) >= 1, model.sum(w[(d,i,j)] for i in range(len(node_ops[d])) for j in range(len(node_ops[d])) if i != j) == (model.sum([z[(op.Job,op.Option)] for op in node_ops[d]])-1))

        if not self.Benchmark.InfCapacity:
            site_id_ops = [site_ops[site] for d,site in enumerate(site_ops.keys())]
            site_id_cap = [trans_site_info[site][1] for d,site in enumerate(site_ops.keys())]
            site_d_i_j = [(d,i,j) for d in range(len(site_id_ops)) for i in range(len(site_id_ops[d])) for j in range(len(site_id_ops[d]))]
            site_d_i = [(d,i) for d in range(len(site_id_ops)) for i in range(len(site_id_ops[d]))]

            # %%
            # start time of operation i in site d
            s = model.integer_var_dict(site_d_i, name="s")
            # end time of operation i in site d
            e = model.integer_var_dict(site_d_i, name="e")

            # %%
            for d in range(len(site_id_ops)):
                for i in range(len(site_id_ops[d])):
                    model.add_constraint(s[(d,i)] == x[(site_id_ops[d][i].Job, site_id_ops[d][i].Option, site_id_ops[d][i].Order)])
                    model.add_constraint(e[(d,i)] == (x[(site_id_ops[d][i].Job, site_id_ops[d][i].Option, site_id_ops[d][i].Order+1)]-1))

            # %%
            phi = model.binary_var_dict(site_d_i_j, name="phi")

            # %%
            A = model.binary_var_dict(site_d_i_j, name="A")
            B = model.binary_var_dict(site_d_i_j, name="B")
            C = model.binary_var_dict(site_d_i_j, name="C")
            D = model.binary_var_dict(site_d_i_j, name="D")

            # %%
            for d in range(len(site_id_ops)):
                for i in range(len(site_id_ops[d])):
                    for j in range(len(site_id_ops[d])):
                        if i == j:
                            model.add_constraint(phi[(d,i,j)] == 0)
                            continue
                        
                        j1,op1,o1 = site_id_ops[d][i].Job,site_id_ops[d][i].Option,site_id_ops[d][i].Order
                        j2,op2,o2 = site_id_ops[d][j].Job,site_id_ops[d][j].Option,site_id_ops[d][j].Order

                        model.add_equivalence(A[(d,i,j)], s[(d,j)]>= s[(d,i)])
                        model.add_equivalence(B[(d,i,j)], e[(d,i)]>= s[(d,j)])
                        model.add_equivalence(C[(d,i,j)], e[(d,j)]>= s[(d,i)])
                        model.add_equivalence(D[(d,i,j)], e[(d,i)]>= e[(d,j)])
            
                        model.add_constraint(z[(j1,op1)] >= phi[(d,i,j)])
                        model.add_constraint(z[(j2,op2)] >= phi[(d,i,j)])
                        
                        model.add_constraint(B[(d,i,j)]+C[(d,i,j)]-phi[(d,i,j)] >= 0)
                        model.add_constraint(A[(d,i,j)]+C[(d,i,j)]+D[(d,j,i)]-phi[(d,i,j)] >= 0)
                        model.add_constraint(B[(d,i,j)]+D[(d,i,j)]+A[(d,j,i)]-phi[(d,i,j)] >= 0)
                        model.add_constraint(A[(d,i,j)]+A[(d,j,i)]+D[(d,i,j)]+D[(d,j,i)]-phi[(d,i,j)] >= 0)
                        
                        model.add_constraint(phi[(d,i,j)]-A[(d,i,j)]-B[(d,i,j)]-z[(j1,op1)]-z[(j2,op2)]+3 >= 0)
                        model.add_constraint(phi[(d,i,j)]-B[(d,i,j)]-D[(d,j,i)]-z[(j1,op1)]-z[(j2,op2)]+3 >= 0)
                        model.add_constraint(phi[(d,i,j)]-C[(d,i,j)]-D[(d,i,j)]-z[(j1,op1)]-z[(j2,op2)]+3 >= 0)
                        model.add_constraint(phi[(d,i,j)]-C[(d,i,j)]-A[(d,j,i)]-z[(j1,op1)]-z[(j2,op2)]+3 >= 0)
                        
                        model.add_constraint(phi[(d,i,j)] == phi[(d,j,i)])
 
            for d in range(len(site_id_ops)):
                cap = site_id_cap[d]
                if cap == 1:
                    for i in range(len(site_id_ops[d])):
                        for j in range(len(site_id_ops[d])):
                            if i == j:
                                continue
                            model.add_constraint(phi[(d,i,j)] == 0)
                else:
                    for comb in combinations(range(len(site_id_ops[d])), cap+1):
                        model.add_constraint(model.sum([phi[(d,comb[i], comb[j])] for i in range(cap) for j in range(i+1, cap+1)]) <= (cap)*(cap+1)//2-1)        
        
        model.set_time_limit(60)
        # model.error_handler = DefaultErrorHandler(InfoLevel.ERROR)
        model.error_handler.InfoLevel = InfoLevel.ERROR
        model.context.cplex_parameters.threads = 8
        msol = model.solve()
        
        # Job verification (only one option is selected)
        selected_opt = []
        for k in range(num_jobs):
            count = 0
            for l in range(job_num_opt[k]):
                if z[(k,l)].solution_value == 1:
                    selected_opt += l,
                    # print(l, x[(k,l,num_ops[k][l]-1)].solution_value+all_process_time[k][l][-1]-x[(k,l,0)].solution_value)
                    count += 1
            assert count == 1

        # Option verification
        for k in range(num_jobs):
            l = selected_opt[k]
            x_sol = np.array([x[(k,l,t)].solution_value for t in range(num_ops[k][l])], dtype=np.int64)
            assert model.objective_value >= x_sol[-1]+all_process_time[k][l][-1]
            # print(x_sol)
            # assert np.allclose(x_sol[1:] >= x_sol[:-1]+all_process_time[k][l][:-1], atol=1e-5)
            assert z[(k,l)].solution_value == 1
            assert np.all(np.round(x_sol[1:]) >= np.round(x_sol[:-1]+all_process_time[k][l][:-1]))

        # node verification
        for d in range(num_nodes):
            time_slots = np.zeros(round(model.objective_value))
            num_op = len(node_ops[d])
            w_sol = np.zeros((num_op, num_op))
            for i in range(num_op):
                for j in range(num_op):
                    w_sol[i,j] = w[(d,i,j)].solution_value
            # assert np.all(w_sol.sum(axis=0) <= 1)
            # print(w_sol)
            assert np.all(np.round(w_sol.sum(1)) <= 1)
            assert np.all(np.round(w_sol.sum(0)) <= 1)
            
            if sum([z[(op.Job,op.Option)].solution_value for op in node_ops[d]]) >= 1:
                # print(sum([z[(op.Job,op.Option)].solution_value for op in node_ops[d]]),w_sol)
                assert round(w_sol.sum()) == round(sum([z[(op.Job,op.Option)].solution_value for op in node_ops[d]])-1)
            for i in range(num_op):
                j1,op1,o1 = node_ops[d][i].Job,node_ops[d][i].Option,node_ops[d][i].Order
                if z[(j1,op1)].solution_value == 1:
                    start = x[(j1,op1,o1)].solution_value
                    assert round(start) >= node_from_to[d][self.Benchmark.node_init_pos][node_ops[d][i].From]
                    if o1 == num_ops[j1][op1]-1:
                        end = start+all_process_time[j1][op1][o1]
                    else:
                        end = x[(j1,op1,o1+1)].solution_value
                    time_slots[round(start):round(end)] += 1
            # print(time_slots)
            assert np.all(time_slots <= 1)
                
            for i in range(num_op):
                for j in range(num_op):
                    if w[(d,i,j)].solution_value == 1:
                        assert i != j
                        j1,op1,o1 = node_ops[d][i].Job,node_ops[d][i].Option,node_ops[d][i].Order
                        j2,op2,o2 = node_ops[d][j].Job,node_ops[d][j].Option,node_ops[d][j].Order
                        assert z[(j1,op1)].solution_value == 1
                        assert z[(j2,op2)].solution_value == 1
                        if o1 == num_ops[j1][op1]-1:
                            assert round(x[(j2,op2,o2)].solution_value) >= round(x[(j1,op1,o1)].solution_value+all_process_time[j1][op1][o1]+empty_car[d][i][j])
                            # print(x[(j2,op2,o2)].solution_value, x[(j1,op1,o1)].solution_value+all_process_time[j1][op1][o1]+empty_car[d][i][j],all_process_time[j1][op1][o1],empty_car[d][i][j])
                            # assert x[(j2,op2,o2)].solution_value >= x[(j1,op1,o1)].solution_value+all_process_time[j1][op1][o1]+empty_car[d][i][j]
                        else:
                            # print(x[(j2,op2,o2)].solution_value, x[(j1,op1,o1+1)].solution_value+empty_car[d][i][j])
                            assert round(x[(j2,op2,o2)].solution_value) >= round(x[(j1,op1,o1+1)].solution_value+empty_car[d][i][j])
 

        # %%
        # site verification
        if not self.Benchmark.InfCapacity:
            for s, ops in enumerate(site_id_ops):
                time_slots = np.zeros(round(model.objective_value))
                for i in range(len(ops)):
                    j1,op1,o1 = site_id_ops[s][i].Job,site_id_ops[s][i].Option,site_id_ops[s][i].Order
                    if z[(j1,op1)].solution_value == 1:
                        time_slots[round(x[(j1,op1,o1)].solution_value):round(x[(j1,op1,o1+1)].solution_value)] += 1
                assert np.all(time_slots <= site_id_cap[s])

            # %%
            # site verification (phi)
            for s, ops in enumerate(site_id_ops):
                num_ops = len(ops)
                for i in range(num_ops):
                    for j in range(num_ops):
                        assert phi[(s,i,j)].solution_value == phi[(s,j,i)].solution_value
                        j1,op1,o1 = site_id_ops[s][i].Job,site_id_ops[s][i].Option,site_id_ops[s][i].Order
                        j2,op2,o2 = site_id_ops[s][j].Job,site_id_ops[s][j].Option,site_id_ops[s][j].Order
                        start1 = x[(j1,op1,o1)].solution_value
                        end1 = x[(j1,op1,o1+1)].solution_value
                        start2 = x[(j2,op2,o2)].solution_value
                        end2 = x[(j2,op2,o2+1)].solution_value
                        
                        if z[(j1,op1)].solution_value == 1 and z[(j2,op2)].solution_value == 1 and phi[(s,i,j)].solution_value == 1:
                            if start1 >= start2:
                                assert round(end2) >= round(start1)
                            else:
                                assert round(end1) >= round(start2)
                                
        return msol, model