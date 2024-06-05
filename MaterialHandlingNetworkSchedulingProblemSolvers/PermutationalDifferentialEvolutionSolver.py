from .BenchmarkParser import operation, site_operation, MHSP_Benchmark
from .MetaheuristicAlgorithmLibrary import DE_parallel
import numpy as np
import heapq
import numpy as np
import warnings
from numba import njit, types, prange
from numba.typed import Dict, List

warnings.filterwarnings("ignore")

# %%
# event: job 0, node 1, site 2

@njit(fastmath=True, cache=True)
def create_engine():
    engine = List()
    engine.append((0,0,0))
    engine.pop()
    # engine.pop()
    return engine

@njit(fastmath=True, cache=True)
def add_event(engine, event):
    heapq.heappush(engine, event)

@njit(fastmath=True, cache=True)
def get_event(engine):
    return heapq.heappop(engine)

@njit(fastmath=True, cache=True)
def FiniteCap_MHSP_Simulate(decode_ids, init_pos, start_after, end_before, opID2op, jobs_nb, trans_site_info_nb, node_from_to_nb):
    job_exec_ops = List()
    node_exec_ops = List()
    num_jobs = len(jobs_nb)
    num_nodes = len(node_from_to_nb)
    job_add_op_id = np.zeros(num_jobs, dtype=np.int64)
    total_exec_ops = List()

    for j in range(num_jobs):
        j_nb = List()
        j_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        job_exec_ops.append(j_nb)
        j_nb.pop()
        
    for n in range(num_nodes):
        n_nb = List()
        n_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        node_exec_ops.append(n_nb)
        n_nb.pop()

    total_exec_ops.append(operation(-1,-1,-1,-1,-1,-1,-1))
    total_exec_ops.pop()
    job_selected_path = np.full(num_jobs, -1, dtype=np.int64)
    
    for id in decode_ids:
        # id = int(id)
        if job_selected_path[opID2op[id].Job] == -1:
            job_selected_path[opID2op[id].Job] = opID2op[id].Option
            for op in jobs_nb[opID2op[id].Job][opID2op[id].Option]:
                job_exec_ops[opID2op[id].Job].append(op)
        if job_selected_path[opID2op[id].Job] == opID2op[id].Option:
            op = jobs_nb[opID2op[id].Job][opID2op[id].Option][job_add_op_id[opID2op[id].Job]]
            # if isinstance(op, operation):
            # job_exec_ops[op.Job].append(op)
            node_exec_ops[op.Node].append(op)
            total_exec_ops.append(op)
            job_add_op_id[op.Job] += 1
            assert opID2op[id].Job == op.Job
    
    job_ops_start_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_ops_end_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_available_time = np.zeros(num_jobs, dtype=np.int64)
    node_available_time = np.zeros(num_nodes, dtype=np.int64)
    site_buffer_level = {k:0 for k in trans_site_info_nb.keys()}
    
    engine = create_engine()
    for j in range(num_jobs):
        job_available_time[j] = start_after[j]
    for d in range(num_nodes):
        if len(node_exec_ops[d]) > 0:
            node_available_time[d] = node_from_to_nb[d][init_pos][node_exec_ops[d][0].From]
            add_event(engine, (node_from_to_nb[d][init_pos][node_exec_ops[d][0].From], d, 1))
        # first_op = job_exec_ops[j][0]
        # if node_exec_ops[first_op.Node][0] == first_op:
        #     # print(first_op)
        #     add_event(engine, (job_available_time[j], first_op.Node, 1))
            
    while len(engine)>0:
        time, node, event_type = get_event(engine)
        if len(node_exec_ops[node]) == 0:
            continue
        if time < node_available_time[node]:
                continue
        op = node_exec_ops[node][0]
        if job_exec_ops[op.Job][0] != op:
            continue
        if time < job_available_time[op.Job]:
            continue
        if event_type == 0 and job_ops_start_time[op.Job][op.Order//2] == -1:
            continue
        # completion
        if event_type == 0:
            complete = True
            # if the operation is last in a job
            if op == job_exec_ops[op.Job][-1]:
                job_ops_end_time[op.Job][-1] = time
                job_exec_ops[op.Job].pop(0)
                # node_available_time[node] = time
                node_exec_ops[node].pop(0)        
            else:
                next_node_op = job_exec_ops[op.Job][1]
                next_site = (op.Node, op.To, next_node_op.Node, next_node_op.From)
                # next_site = (next_site.Inbound_Node, next_site.Inbound_pd, next_site.Outbound_Node, next_site.Outbound_pd)
                # if the next site is available
                if site_buffer_level[next_site] + 1 <= trans_site_info_nb[next_site][1]:
                    # print("job", op.Job, "operation", op.Order//2, "completed at", time, "at node", node, "to site", next_site)
                    site_buffer_level[next_site] += 1
                    job_available_time[op.Job] = time+max(trans_site_info_nb[next_site][0],1)
                    job_ops_end_time[op.Job][op.Order//2] = time
                    # node_available_time[node] = time
                    job_exec_ops[op.Job].pop(0)
                    
                    node_exec_ops[node].pop(0)
                    # print(job_exec_ops[op.Job])
                    add_event(engine, (job_available_time[op.Job], job_exec_ops[op.Job][0].Node, 1))
                else:
                    # print("job", op.Job, "operation", op.Order//2, "at", time, "at node", node, "cannot enter  site", next_site, "due to buffer limit", trans_site_info[next_site][1], site_buffer_level[next_site])
                    complete = False
                    # node_available_time[node] = inf
            
            if complete:
                # add_event(engine, (time, jobs[op.Job][op.Option][op.Order-2].Node, 0))
                if len(node_exec_ops[node]) > 0:
                    op_next = node_exec_ops[node][0]
                    node_available_time[node] = time+node_from_to_nb[node][op.To][op_next.From]
                    add_event(engine, (node_available_time[node], node, 1))
        # arrival
        else:
            if job_available_time[op.Job] <= time:
                # print("job", op.Job, "operation", op.Order//2, "started at", time, "at node", node)
                # print(node_exec_ops[node])
                # print(job_exec_ops[op.Job])
                node_available_time[node] = time+op.pt
                job_available_time[op.Job] = time+op.pt
                # print(job_ops_start_time[op.Job])
                job_ops_start_time[op.Job][op.Order//2] = time
                # print(job_ops_start_time[op.Job])
                add_event(engine, (node_available_time[node], op.Node, 0))
                if op.Order > 0:
                    prev_node_op = jobs_nb[op.Job][op.Option][op.Order//2-1]
                    prev_site = (prev_node_op.Node, prev_node_op.To, op.Node, op.From)
                    # prev_site = (prev_site_op.Inbound_Node, prev_site_op.Inbound_pd, prev_site_op.Outbound_Node, prev_site_op.Outbound_pd)
                    site_buffer_level[prev_site] -= 1
                    if site_buffer_level[prev_site] == trans_site_info_nb[prev_site][1] - 1:
                    # print(jobs[op.Job][op.Option][op.Order-2].Node)
                        add_event(engine, (time, prev_node_op.Node, 0))
            else:
                add_event(engine, (job_available_time[op.Job], op.Node, 1))     

    return job_ops_end_time

@njit(fastmath=True, cache=True)
def FiniteCap_MHSP_Simulate_explicit(decode_ids, init_pos, start_after, end_before, opID2op, jobs_nb, trans_site_info_nb, node_from_to_nb):
    job_exec_ops = List()
    node_exec_ops = List()
    num_jobs = len(jobs_nb)
    num_nodes = len(node_from_to_nb)
    job_add_op_id = np.zeros(num_jobs, dtype=np.int64)
    total_exec_ops = List()

    for j in range(num_jobs):
        j_nb = List()
        j_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        job_exec_ops.append(j_nb)
        j_nb.pop()
        
    for n in range(num_nodes):
        n_nb = List()
        n_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        node_exec_ops.append(n_nb)
        n_nb.pop()

    total_exec_ops.append(operation(-1,-1,-1,-1,-1,-1,-1))
    total_exec_ops.pop()
    job_selected_path = np.full(num_jobs, -1, dtype=np.int64)
    
    for id in decode_ids:
        # id = int(id)
        if job_selected_path[opID2op[id].Job] == -1:
            job_selected_path[opID2op[id].Job] = opID2op[id].Option
            for op in jobs_nb[opID2op[id].Job][opID2op[id].Option]:
                job_exec_ops[opID2op[id].Job].append(op)
        if job_selected_path[opID2op[id].Job] == opID2op[id].Option:
            op = jobs_nb[opID2op[id].Job][opID2op[id].Option][job_add_op_id[opID2op[id].Job]]
            # if isinstance(op, operation):
            # job_exec_ops[op.Job].append(op)
            node_exec_ops[op.Node].append(op)
            total_exec_ops.append(op)
            job_add_op_id[op.Job] += 1
            assert opID2op[id].Job == op.Job
    
    job_ops_start_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_ops_end_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_available_time = np.zeros(num_jobs, dtype=np.int64)
    node_available_time = np.zeros(num_nodes, dtype=np.int64)
    site_buffer_level = {k:0 for k in trans_site_info_nb.keys()}
    
    engine = create_engine()
    for j in range(num_jobs):
        job_available_time[j] = start_after[j]
    for d in range(num_nodes):
        if len(node_exec_ops[d]) > 0:
            node_available_time[d] = node_from_to_nb[d][init_pos][node_exec_ops[d][0].From]
            add_event(engine, (node_from_to_nb[d][init_pos][node_exec_ops[d][0].From], d, 1))
    # for j in range(num_jobs):
    #     job_available_time[j] = start_after[j]
    #     first_op = job_exec_ops[j][0]
    #     if node_exec_ops[first_op.Node][0] == first_op:
    #         # print(first_op)
    #         add_event(engine, (job_available_time[j], first_op.Node, 1))
            
    while len(engine)>0:
        time, node, event_type = get_event(engine)
        if len(node_exec_ops[node]) == 0:
            continue
        if time < node_available_time[node]:
                continue
        op = node_exec_ops[node][0]
        if job_exec_ops[op.Job][0] != op:
            continue
        if time < job_available_time[op.Job]:
            continue
        if event_type == 0 and job_ops_start_time[op.Job][op.Order//2] == -1:
            continue
        # completion
        if event_type == 0:
            complete = True
            # if the operation is last in a job
            if op == job_exec_ops[op.Job][-1]:
                job_ops_end_time[op.Job][-1] = time
                job_exec_ops[op.Job].pop(0)
                # node_available_time[node] = time
                node_exec_ops[node].pop(0)        
            else:
                next_node_op = job_exec_ops[op.Job][1]
                next_site = (op.Node, op.To, next_node_op.Node, next_node_op.From)
                # next_site = (next_site.Inbound_Node, next_site.Inbound_pd, next_site.Outbound_Node, next_site.Outbound_pd)
                # if the next site is available
                if site_buffer_level[next_site] + 1 <= trans_site_info_nb[next_site][1]:
                    # print("job", op.Job, "operation", op.Order//2, "completed at", time, "at node", node, "to site", next_site)
                    site_buffer_level[next_site] += 1
                    job_available_time[op.Job] = time+max(trans_site_info_nb[next_site][0],1)
                    job_ops_end_time[op.Job][op.Order//2] = time
                    # node_available_time[node] = time
                    job_exec_ops[op.Job].pop(0)
                    
                    node_exec_ops[node].pop(0)
                    # print(job_exec_ops[op.Job])
                    add_event(engine, (job_available_time[op.Job], job_exec_ops[op.Job][0].Node, 1))
                else:
                    # print("job", op.Job, "operation", op.Order//2, "at", time, "at node", node, "cannot enter  site", next_site, "due to buffer limit", trans_site_info[next_site][1], site_buffer_level[next_site])
                    complete = False
                    # node_available_time[node] = inf
            
            if complete:
                # add_event(engine, (time, jobs[op.Job][op.Option][op.Order-2].Node, 0))
                if len(node_exec_ops[node]) > 0:
                    op_next = node_exec_ops[node][0]
                    node_available_time[node] = time+node_from_to_nb[node][op.To][op_next.From]
                    add_event(engine, (node_available_time[node], node, 1))
        # arrival
        else:
            if job_available_time[op.Job] <= time:
                # print("job", op.Job, "operation", op.Order//2, "started at", time, "at node", node)
                # print(node_exec_ops[node])
                # print(job_exec_ops[op.Job])
                node_available_time[node] = time+op.pt
                job_available_time[op.Job] = time+op.pt
                # print(job_ops_start_time[op.Job])
                job_ops_start_time[op.Job][op.Order//2] = time
                # print(job_ops_start_time[op.Job])
                add_event(engine, (node_available_time[node], op.Node, 0))
                if op.Order > 0:
                    prev_node_op = jobs_nb[op.Job][op.Option][op.Order//2-1]
                    prev_site = (prev_node_op.Node, prev_node_op.To, op.Node, op.From)
                    # prev_site = (prev_site_op.Inbound_Node, prev_site_op.Inbound_pd, prev_site_op.Outbound_Node, prev_site_op.Outbound_pd)
                    site_buffer_level[prev_site] -= 1
                    if site_buffer_level[prev_site] == trans_site_info_nb[prev_site][1] - 1:
                    # print(jobs[op.Job][op.Option][op.Order-2].Node)
                        add_event(engine, (time, prev_node_op.Node, 0))
            else:
                add_event(engine, (job_available_time[op.Job], op.Node, 1))     

    return job_selected_path, job_ops_start_time, job_ops_end_time

@njit(fastmath=True, cache=True)
def FiniteCap_MHSP_Simulate_debug(decode_ids, init_pos, start_after, end_before, opID2op, jobs_nb, trans_site_info_nb, node_from_to_nb):
    job_exec_ops = List()
    node_exec_ops = List()
    num_jobs = len(jobs_nb)
    num_nodes = len(node_from_to_nb)
    job_add_op_id = np.zeros(num_jobs, dtype=np.int64)
    total_exec_ops = List()

    for j in range(num_jobs):
        j_nb = List()
        j_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        job_exec_ops.append(j_nb)
        j_nb.pop()
        
    for n in range(num_nodes):
        n_nb = List()
        n_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        node_exec_ops.append(n_nb)
        n_nb.pop()

    total_exec_ops.append(operation(-1,-1,-1,-1,-1,-1,-1))
    total_exec_ops.pop()
    job_selected_path = np.full(num_jobs, -1, dtype=np.int64)
    
    for id in decode_ids:
        # id = int(id)
        if job_selected_path[opID2op[id].Job] == -1:
            job_selected_path[opID2op[id].Job] = opID2op[id].Option
        if job_selected_path[opID2op[id].Job] == opID2op[id].Option:
            op = jobs_nb[opID2op[id].Job][opID2op[id].Option][job_add_op_id[opID2op[id].Job]]
            # if isinstance(op, operation):
            job_exec_ops[op.Job].append(op)
            node_exec_ops[op.Node].append(op)
            total_exec_ops.append(op)
            job_add_op_id[op.Job] += 1
            assert opID2op[id].Job == op.Job
    
    job_ops_start_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_ops_end_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_available_time = np.zeros(num_jobs, dtype=np.int64)
    node_available_time = np.zeros(num_nodes, dtype=np.int64)
    site_buffer_level = {k:0 for k in trans_site_info_nb.keys()}
    site2ID = {k:i for i,k in enumerate(trans_site_info_nb.keys())}
    
    engine = create_engine()
    for j in range(num_jobs):
        job_available_time[j] = start_after[j]
    for d in range(num_nodes):
        if len(node_exec_ops[d]) > 0:
            node_available_time[d] = node_from_to_nb[d][init_pos][node_exec_ops[d][0].From]
            add_event(engine, (node_from_to_nb[d][init_pos][node_exec_ops[d][0].From], d, 1))
    # for j in range(num_jobs):
    #     job_available_time[j] = start_after[j]
    #     first_op = job_exec_ops[j][0]
    #     if node_exec_ops[first_op.Node][0] == first_op:
    #         # print(first_op)
    #         add_event(engine, (job_available_time[j], first_op.Node, 1))
            
    while len(engine)>0:
        time, node, event_type = get_event(engine)
        if len(node_exec_ops[node]) == 0:
            continue
        if time < node_available_time[node]:
                continue
        op = node_exec_ops[node][0]
        if job_exec_ops[op.Job][0] != op:
            continue
        if time < job_available_time[op.Job]:
            continue
        if event_type == 0 and job_ops_start_time[op.Job][op.Order//2] == -1:
            continue
        # completion
        if event_type == 0:
            complete = True
            # if the operation is last in a job
            if op == job_exec_ops[op.Job][-1]:
                job_ops_end_time[op.Job][-1] = time
                job_exec_ops[op.Job].pop(0)
                # node_available_time[node] = time
                node_exec_ops[node].pop(0)        
            else:
                next_node_op = job_exec_ops[op.Job][1]
                # next_node_op = jobs_nb[op.Job][op.Option][op.Order//2+1]
                next_site = (op.Node, op.To, next_node_op.Node, next_node_op.From)
                # next_site = (next_site.Inbound_Node, next_site.Inbound_pd, next_site.Outbound_Node, next_site.Outbound_pd)
                # if the next site is available
                if site_buffer_level[next_site] + 1 <= trans_site_info_nb[next_site][1]:
                    # print("job", op.Job, "operation", op.Order//2, "completed at", time, "at node", node, "to site", next_site)
                    site_buffer_level[next_site] += 1
                    job_available_time[op.Job] = time+max(trans_site_info_nb[next_site][0],1)
                    job_ops_end_time[op.Job][op.Order//2] = time
                    # node_available_time[node] = time
                    job_exec_ops[op.Job].pop(0)
                    
                    node_exec_ops[node].pop(0)
                    # print(job_exec_ops[op.Job])
                    add_event(engine, (job_available_time[op.Job], job_exec_ops[op.Job][0].Node, 1))
                else:
                    # print("job", op.Job, "operation", op.Order//2, "at", time, "at node", node, "cannot enter  site", next_site, "due to buffer limit", trans_site_info[next_site][1], site_buffer_level[next_site])
                    complete = False
                    # node_available_time[node] = inf
            
            if complete:
                # add_event(engine, (time, jobs[op.Job][op.Option][op.Order-2].Node, 0))
                if len(node_exec_ops[node]) > 0:
                    op_next = node_exec_ops[node][0]
                    node_available_time[node] = time+node_from_to_nb[node][op.To][op_next.From]
                    add_event(engine, (node_available_time[node], node, 1))
        # arrival
        else:
            if job_available_time[op.Job] <= time:
                # print("job", op.Job, "operation", op.Order//2, "started at", time, "at node", node)
                # print(node_exec_ops[node])
                # print(job_exec_ops[op.Job])
                node_available_time[node] = time+op.pt
                # print(job_ops_start_time[op.Job])
                job_ops_start_time[op.Job][op.Order//2] = time
                # print(job_ops_start_time[op.Job])
                add_event(engine, (node_available_time[node], op.Node, 0))
                if op.Order > 0:
                    prev_node_op = jobs_nb[op.Job][op.Option][op.Order//2-1]
                    prev_site = (prev_node_op.Node, prev_node_op.To, op.Node, op.From)
                    # prev_site = (prev_site_op.Inbound_Node, prev_site_op.Inbound_pd, prev_site_op.Outbound_Node, prev_site_op.Outbound_pd)
                    site_buffer_level[prev_site] -= 1
                    if site_buffer_level[prev_site] == trans_site_info_nb[prev_site][1] - 1:
                    # print(jobs[op.Job][op.Option][op.Order-2].Node)
                        add_event(engine, (time, prev_node_op.Node, 0))
            else:
                add_event(engine, (job_available_time[op.Job], op.Node, 1))     

    obj  = max([j.max() for j in job_ops_end_time])
    num_sites = len(trans_site_info_nb.keys())
    time_slots = [np.zeros(round(obj))]*num_sites
    # Option verification
    for k in range(num_jobs):
        l = job_selected_path[k]
        for i in range(len(jobs_nb[k][l])):
            assert job_ops_end_time[k][i] >= job_ops_start_time[k][i]+jobs_nb[k][l][i].pt
        for i in range(1,len(jobs_nb[k][l])):
            op1 = jobs_nb[k][l][i-1]
            op2 = jobs_nb[k][l][i]
            if trans_site_info_nb[(op1.Node, op1.To, op2.Node, op2.From)][0] > job_ops_start_time[k][i]-job_ops_end_time[k][i-1]:
                print(trans_site_info_nb[(op1.Node, op1.To, op2.Node, op2.From)][0], job_ops_start_time[k][i]-job_ops_end_time[k][i-1])
                print(job_ops_start_time[k])
                print(job_ops_end_time[k])
                raise
            time_slots[site2ID[(op1.Node, op1.To, op2.Node, op2.From)]][round(job_ops_end_time[k][i-1]):round(job_ops_start_time[k][i])] += 1
    
    for site in trans_site_info_nb.keys():
        s = site2ID[site]
        assert np.all(time_slots[s] <= trans_site_info_nb[site][1])

    job_add_op_id = np.zeros(num_jobs, dtype=np.int64)
    for id in decode_ids:
        if job_selected_path[opID2op[id].Job] == opID2op[id].Option:
            op = jobs_nb[opID2op[id].Job][opID2op[id].Option][job_add_op_id[opID2op[id].Job]]
            # if isinstance(op, operation):
            # job_exec_ops[op.Job].append(op)
            node_exec_ops[op.Node].append(op)
            # total_exec_ops.append(op)
            job_add_op_id[op.Job] += 1
            # assert opID2op[id].Job == op.Job
    
    # node verification
    for d in range(num_nodes):
        op1 = node_exec_ops[d][i]
        assert job_ops_start_time[op1.Job][op1.Order//2] >= node_from_to_nb[d][init_pos][op1.From]

    for d in range(num_nodes):
        add_event(engine, (node_from_to_nb[d][init_pos][node_exec_ops[d][0].From], d, 1))
        num_op = len(node_exec_ops[d])
        for i in range(1, num_op):
            op1 = node_exec_ops[d][i-1]
            op2 = node_exec_ops[d][i]
            
            assert job_ops_end_time[op1.Job][op1.Order//2] + node_from_to_nb[d][op1.To][op2.From]<= job_ops_start_time[op2.Job][op2.Order//2]
    
    # site verification
    return job_ops_end_time

@njit(fastmath=True, cache=True, debug=True)
def InfCap_MHSP_Simulate(decode_ids, init_pos, start_after, end_before, opID2op, jobs_nb, trans_site_info_nb, node_from_to_nb):
    job_exec_ops = List()
    node_exec_ops = List()
    num_jobs = len(jobs_nb)
    num_nodes = len(node_from_to_nb)
    job_add_op_id = np.zeros(num_jobs, dtype=np.int64)
    total_exec_ops = List()

    for j in range(num_jobs):
        j_nb = List()
        j_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        job_exec_ops.append(j_nb)
        j_nb.pop()
        
    for n in range(num_nodes):
        n_nb = List()
        n_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        node_exec_ops.append(n_nb)
        n_nb.pop()

    total_exec_ops.append(operation(-1,-1,-1,-1,-1,-1,-1))
    total_exec_ops.pop()
    job_selected_path = np.full(num_jobs, -1, dtype=np.int64)

    for id in decode_ids:
        # id = int(id)
        if job_selected_path[opID2op[id].Job] == -1:
            job_selected_path[opID2op[id].Job] = opID2op[id].Option
            for op in jobs_nb[opID2op[id].Job][opID2op[id].Option]:
                job_exec_ops[opID2op[id].Job].append(op)
        if job_selected_path[opID2op[id].Job] == opID2op[id].Option:
            op = jobs_nb[opID2op[id].Job][opID2op[id].Option][job_add_op_id[opID2op[id].Job]]
            # if isinstance(op, operation):
            # job_exec_ops[op.Job].append(op)
            node_exec_ops[op.Node].append(op)
            total_exec_ops.append(op)
            job_add_op_id[op.Job] += 1

    job_ops_start_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_ops_end_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_available_time = start_after.copy()
    job_current_id = np.zeros(num_jobs, dtype=np.int64)
    node_available_time = np.zeros(num_nodes, dtype=np.int64)
    
    for d in range(num_nodes):
        if len(node_exec_ops[d]) > 0:
            node_available_time[d] = node_from_to_nb[d][init_pos][node_exec_ops[d][0].From]
    
    for o,op in enumerate(total_exec_ops):
        if op != job_exec_ops[op.Job][0]:
            print(op, job_exec_ops[op.Job])
        assert op == job_exec_ops[op.Job][0]
        assert op == node_exec_ops[op.Node][0]
        
        id = job_current_id[op.Job]
        job_ops_start_time[op.Job][id] = max(node_available_time[op.Node], job_available_time[op.Job])
        job_ops_end_time[op.Job][id] = job_ops_start_time[op.Job][id]+op.pt
        
        job_exec_ops[op.Job].pop(0)
        if len(job_exec_ops[op.Job]) > 0:
            next_node_op = job_exec_ops[op.Job][0]
            next_site = (op.Node, op.To, next_node_op.Node, next_node_op.From)
            job_available_time[op.Job] = job_ops_end_time[op.Job][id]+max(trans_site_info_nb[next_site][0],1)
            job_current_id[op.Job] += 1
        
        node_exec_ops[op.Node].pop(0)
        if len(node_exec_ops[op.Node]) > 0 :
            node_available_time[op.Node] = job_ops_end_time[op.Job][id]+node_from_to_nb[op.Node][op.To][node_exec_ops[op.Node][0].From]
    
    return job_ops_end_time

@njit(fastmath=True, cache=True, debug=True)
def InfCap_MHSP_Simulate_explicit(decode_ids, init_pos, start_after, end_before, opID2op, jobs_nb, trans_site_info_nb, node_from_to_nb):
    job_exec_ops = List()
    node_exec_ops = List()
    num_jobs = len(jobs_nb)
    num_nodes = len(node_from_to_nb)
    job_add_op_id = np.zeros(num_jobs, dtype=np.int64)
    total_exec_ops = List()

    for j in range(num_jobs):
        j_nb = List()
        j_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        job_exec_ops.append(j_nb)
        j_nb.pop()
        
    for n in range(num_nodes):
        n_nb = List()
        n_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        node_exec_ops.append(n_nb)
        n_nb.pop()

    total_exec_ops.append(operation(-1,-1,-1,-1,-1,-1,-1))
    total_exec_ops.pop()
    job_selected_path = np.full(num_jobs, -1, dtype=np.int64)

    for id in decode_ids:
        # id = int(id)
        if job_selected_path[opID2op[id].Job] == -1:
            job_selected_path[opID2op[id].Job] = opID2op[id].Option
            for op in jobs_nb[opID2op[id].Job][opID2op[id].Option]:
                job_exec_ops[opID2op[id].Job].append(op)
        if job_selected_path[opID2op[id].Job] == opID2op[id].Option:
            op = jobs_nb[opID2op[id].Job][opID2op[id].Option][job_add_op_id[opID2op[id].Job]]
            # if isinstance(op, operation):
            # job_exec_ops[op.Job].append(op)
            node_exec_ops[op.Node].append(op)
            total_exec_ops.append(op)
            job_add_op_id[op.Job] += 1

    job_ops_start_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_ops_end_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_available_time = start_after.copy()
    job_current_id = np.zeros(num_jobs, dtype=np.int64)
    node_available_time = np.zeros(num_nodes, dtype=np.int64)
    
    for d in range(num_nodes):
        if len(node_exec_ops[d]) > 0:
            node_available_time[d] = node_from_to_nb[d][init_pos][node_exec_ops[d][0].From]
    
    for o,op in enumerate(total_exec_ops):
        if op != job_exec_ops[op.Job][0]:
            print(op, job_exec_ops[op.Job])
        assert op == job_exec_ops[op.Job][0]
        assert op == node_exec_ops[op.Node][0]
        
        id = job_current_id[op.Job]
        job_ops_start_time[op.Job][id] = max(node_available_time[op.Node], job_available_time[op.Job])
        job_ops_end_time[op.Job][id] = job_ops_start_time[op.Job][id]+op.pt
        
        job_exec_ops[op.Job].pop(0)
        if len(job_exec_ops[op.Job]) > 0:
            next_node_op = job_exec_ops[op.Job][0]
            next_site = (op.Node, op.To, next_node_op.Node, next_node_op.From)
            job_available_time[op.Job] = job_ops_end_time[op.Job][id]+max(trans_site_info_nb[next_site][0],1)
            job_current_id[op.Job] += 1
        
        node_exec_ops[op.Node].pop(0)
        if len(node_exec_ops[op.Node]) > 0 :
            node_available_time[op.Node] = job_ops_end_time[op.Job][id]+node_from_to_nb[op.Node][op.To][node_exec_ops[op.Node][0].From]
    
    return job_selected_path, job_ops_start_time, job_ops_end_time

@njit(fastmath=True, cache=True)
def InfCap_MHSP_Simulate_debug(decode_ids, init_pos, start_after, end_before, opID2op, jobs_nb, trans_site_info_nb, node_from_to_nb):
    job_exec_ops = List()
    node_exec_ops = List()
    num_jobs = len(jobs_nb)
    num_nodes = len(node_from_to_nb)
    job_add_op_id = np.zeros(num_jobs, dtype=np.int64)
    total_exec_ops = List()

    for j in range(num_jobs):
        j_nb = List()
        j_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        job_exec_ops.append(j_nb)
        j_nb.pop()
        
    for n in range(num_nodes):
        n_nb = List()
        n_nb.append(operation(-1,-1,-1,-1,-1,-1,-1))
        node_exec_ops.append(n_nb)
        n_nb.pop()

    total_exec_ops.append(operation(-1,-1,-1,-1,-1,-1,-1))
    total_exec_ops.pop()
    job_selected_path = np.full(num_jobs, -1, dtype=np.int64)

    for id in decode_ids:
        # id = int(id)
        if job_selected_path[opID2op[id].Job] == -1:
            job_selected_path[opID2op[id].Job] = opID2op[id].Option
            for op in jobs_nb[opID2op[id].Job][opID2op[id].Option]:
                job_exec_ops[opID2op[id].Job].append(op)
        if job_selected_path[opID2op[id].Job] == opID2op[id].Option:
            op = jobs_nb[opID2op[id].Job][opID2op[id].Option][job_add_op_id[opID2op[id].Job]]
            # if isinstance(op, operation):
            # job_exec_ops[op.Job].append(op)
            node_exec_ops[op.Node].append(op)
            total_exec_ops.append(op)
            job_add_op_id[op.Job] += 1

    job_ops_start_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_ops_end_time = [np.full(len(jobs_nb[j][job_selected_path[j]]), -1, dtype=np.int64) for j in range(num_jobs)]
    job_available_time = start_after.copy()
    job_current_id = np.zeros(num_jobs, dtype=np.int64)
    node_available_time = np.zeros(num_nodes, dtype=np.int64)
    
    for d in range(num_nodes):
        if len(node_exec_ops[d]) > 0:
            node_available_time[d] = node_from_to_nb[d][init_pos][node_exec_ops[d][0].From]
    
    for o,op in enumerate(total_exec_ops):
        if op != job_exec_ops[op.Job][0]:
            print(op, job_exec_ops[op.Job])
        assert op == job_exec_ops[op.Job][0]
        assert op == node_exec_ops[op.Node][0]
        
        id = job_current_id[op.Job]
        job_ops_start_time[op.Job][id] = max(node_available_time[op.Node], job_available_time[op.Job])
        job_ops_end_time[op.Job][id] = job_ops_start_time[op.Job][id]+op.pt
        
        job_exec_ops[op.Job].pop(0)
        if len(job_exec_ops[op.Job]) > 0:
            next_node_op = job_exec_ops[op.Job][0]
            next_site = (op.Node, op.To, next_node_op.Node, next_node_op.From)
            job_available_time[op.Job] = job_ops_end_time[op.Job][id]+max(trans_site_info_nb[next_site][0],1)
            job_current_id[op.Job] += 1
        
        node_exec_ops[op.Node].pop(0)
        if len(node_exec_ops[op.Node]) > 0 :
            node_available_time[op.Node] = job_ops_end_time[op.Job][id]+node_from_to_nb[op.Node][op.To][node_exec_ops[op.Node][0].From]
    
    # Option verification
    for k in range(num_jobs):
        l = job_selected_path[k]
        for i in range(len(jobs_nb[k][l])):
            assert job_ops_end_time[k][i] >= job_ops_start_time[k][i]+jobs_nb[k][l][i].pt
        for i in range(1,len(jobs_nb[k][l])):
            op1 = jobs_nb[k][l][i-1]
            op2 = jobs_nb[k][l][i]
            if trans_site_info_nb[(op1.Node, op1.To, op2.Node, op2.From)][0] > job_ops_start_time[k][i]-job_ops_end_time[k][i-1]:
                print(trans_site_info_nb[(op1.Node, op1.To, op2.Node, op2.From)][0], job_ops_start_time[k][i]-job_ops_end_time[k][i-1])
                print(job_ops_start_time[k])
                print(job_ops_end_time[k])
                raise

    job_add_op_id = np.zeros(num_jobs, dtype=np.int64)
    for id in decode_ids:
        if job_selected_path[opID2op[id].Job] == opID2op[id].Option:
            op = jobs_nb[opID2op[id].Job][opID2op[id].Option][job_add_op_id[opID2op[id].Job]]
            # if isinstance(op, operation):
            # job_exec_ops[op.Job].append(op)
            node_exec_ops[op.Node].append(op)
            # total_exec_ops.append(op)
            job_add_op_id[op.Job] += 1
            # assert opID2op[id].Job == op.Job
    
    # node verification
    for d in range(num_nodes):
        op1 = node_exec_ops[d][i]
        assert job_ops_start_time[op1.Job][op1.Order//2] >= node_from_to_nb[d][init_pos][op1.From]
    for d in range(num_nodes):   
        num_op = len(node_exec_ops[d])
        for i in range(1, num_op):
            op1 = node_exec_ops[d][i-1]
            op2 = node_exec_ops[d][i]
            
            assert job_ops_end_time[op1.Job][op1.Order//2] + node_from_to_nb[d][op1.To][op2.From]<= job_ops_start_time[op2.Job][op2.Order//2]
    
    # site verification
    return job_ops_end_time

@njit(fastmath=True, cache=True)
def obj_function_cap(encoding, *args):
    init_pos, start_after, end_before, opID2op, jobs_nb, trans_site_info_nb, node_from_to_nb = args
    num_jobs = len(start_after)

    N = 1
    result = np.zeros(N)
    for n in range(N):
        # if N > 1:
        #     tmp_encoding = encoding + np.random.randn(encoding.shape[0])*1e-2
        # else:
        #     tmp_encoding = encoding
        decode_ids = np.full(encoding.shape[0],-1, dtype=np.int64)
        for i in range(encoding.shape[0]):
            decode_ids[encoding[i]] = i
        # if np.unique(decode_ids).shape[0] != decode_ids.shape[0]:
        #     print("error")
        # if decode_ids.min() != 0:
        #     return 1e6
        # assert np.unique(encoding).shape[0] == encoding.shape[0]
        # decode_ids = tmp_encoding.argsort()[::-1]
        job_ops_end_time = FiniteCap_MHSP_Simulate(decode_ids, *args)
        completion_time = [job_ops_end_time[j][-1] for j in range(len(job_ops_end_time))]
        obj_value = 0
        extra = 0
        for j in range(num_jobs):
            if completion_time[j] == -1:
                obj_value += (job_ops_end_time[j] == -1).sum()/len(job_ops_end_time[j])*1e6+1e6
            elif completion_time[j] > end_before[j]:
                extra += (completion_time[j]-end_before[j])**2*1e3
                obj_value = max(obj_value, completion_time[j])
            else:
                obj_value = max(obj_value, completion_time[j])
        result[n] = obj_value+extra
    return result.mean()

@njit(fastmath=True, cache=True)
def obj_function_cap_debug(encoding, *args):
    init_pos, start_after, end_before, opID2op, jobs_nb, trans_site_info_nb, node_from_to_nb = args
    num_jobs = len(start_after)

    N = 1
    result = np.zeros(N)
    for n in range(N):
        # if N > 1:
        #     tmp_encoding = encoding + np.random.randn(encoding.shape[0])*1e-2
        # else:
        #     tmp_encoding = encoding
        decode_ids = np.full(encoding.shape[0],-1, dtype=np.int64)
        for i in range(encoding.shape[0]):
            decode_ids[encoding[i]] = i
        if decode_ids.min() != 0:
            return 1e6
        # if np.unique(decode_ids).shape[0] != decode_ids.shape[0]:
        #     print("error")
        # assert decode_ids.min() == 0
        # assert np.unique(encoding).shape[0] == encoding.shape[0]
        # decode_ids = tmp_encoding.argsort()[::-1]
        job_ops_end_time = FiniteCap_MHSP_Simulate_debug(decode_ids, *args)
        completion_time = [job_ops_end_time[j][-1] for j in range(len(job_ops_end_time))]
        obj_value = 0
        extra = 0
        for j in range(num_jobs):
            if  np.any(job_ops_end_time[j] == -1):
                obj_value += (job_ops_end_time[j] == -1).sum()/len(job_ops_end_time[j])*1e6+1e6
            elif completion_time[j] > end_before[j]:
                extra += (completion_time[j]-end_before[j])**2*1e3
                obj_value = max(obj_value, completion_time[j])
            else:
                obj_value = max(obj_value, completion_time[j])
        result[n] = obj_value+extra
    return result.mean()


@njit(fastmath=True, cache=True, debug=True)
def obj_function_infcap(encoding, *args):
    init_pos, start_after, end_before, opID2op, jobs_nb, trans_site_info_nb, node_from_to_nb = args
    num_jobs = len(start_after)

    N = 1
    result = np.zeros(N)
    for n in range(N):
        decode_ids = np.full(encoding.shape[0],-1, dtype=np.int64)
        for i in range(encoding.shape[0]):
            decode_ids[encoding[i]] = i
        # assert decode_ids.min() == 0
        # if decode_ids.min() != 0:
        #     return 1e6
        job_ops_end_time = InfCap_MHSP_Simulate(decode_ids, *args)
        completion_time = [job_ops_end_time[j][-1] for j in range(num_jobs)]
        obj_value = 0
        extra = 0
        for j in range(num_jobs):
            if np.any(job_ops_end_time[j] == -1):
                obj_value += (job_ops_end_time[j] == -1).sum()/len(job_ops_end_time[j])*1e6+1e6
            elif completion_time[j] > end_before[j]:
                extra += (completion_time[j]-end_before[j])**2*1e3
                obj_value = max(obj_value, completion_time[j])
            else:
                obj_value = max(obj_value, completion_time[j])
        result[n] = obj_value+extra
    return result.mean()

@njit(fastmath=True, cache=True, debug=True)
def obj_function_infcap_debug(encoding, *args):
    init_pos, start_after, end_before, opID2op, jobs_nb, trans_site_info_nb, node_from_to_nb = args
    num_jobs = len(start_after)

    N = 1
    result = np.zeros(N)
    for n in range(N):
        decode_ids = np.full(encoding.shape[0],-1, dtype=np.int64)
        for i in range(encoding.shape[0]):
            decode_ids[encoding[i]] = i
        # assert decode_ids.min() == 0
        # if decode_ids.min() != 0:
        #     return 1e6
        job_ops_end_time = InfCap_MHSP_Simulate_debug(decode_ids, *args)
        completion_time = [job_ops_end_time[j][-1] for j in range(num_jobs)]
        obj_value = 0
        extra = 0
        for j in range(num_jobs):
            if np.any(job_ops_end_time[j] == -1):
                obj_value += (job_ops_end_time[j] == -1).sum()/len(job_ops_end_time[j])*1e6+1e6
            elif completion_time[j] > end_before[j]:
                extra += (completion_time[j]-end_before[j])**2
                obj_value = max(obj_value, completion_time[j])
            else:
                obj_value = max(obj_value, completion_time[j])
        result[n] = obj_value+extra
    return result.mean()

# %%
class DE_Solver:
    def __init__(self, Beanchmark: MHSP_Benchmark) -> None:
        self.Benchmark = Beanchmark
        operations = self.Benchmark.operations
        jobs = self.Benchmark.jobs
        trans_site_info = self.Benchmark.trans_site_info
        node_from_to = self.Benchmark.node_from_to
        start_after = self.Benchmark.start_after
        end_before = self.Benchmark.end_before
        num_nodes = self.Benchmark.num_nodes
        num_jobs = self.Benchmark.num_jobs
        
        
        self.total_num_ops = len([op for op in operations if isinstance(op, operation)])
        self.op2ID = {op:i for i,op in enumerate(operations) if isinstance(op, operation)}
        self.constant_opID2op = List()
        for op in operations:
            if isinstance(op, operation):
                self.constant_opID2op.append(op)

        self.constant_jobs_nb = List()
        for job in jobs:
            job_nb = List()
            for opt in job:
                opt_nb = List()
                for op in opt:
                    if isinstance(op, operation):
                        opt_nb.append(op)
                job_nb.append(opt_nb)
            self.constant_jobs_nb.append(job_nb)

        self.constant_trans_site_info_nb = Dict.empty(
            key_type=types.UniTuple(types.int64, 4),
            value_type=types.UniTuple(types.int64, 2),
        )
        for k,v in trans_site_info.items():
            self.constant_trans_site_info_nb[k] = (v[0], v[1])

        self.start_after = np.array(start_after, dtype=np.int64)
        self.end_before = np.array(end_before, dtype=np.int64)

        self.constant_node_from_to_nb = List()
        for n in range(num_nodes):
            self.constant_node_from_to_nb.append(np.array(node_from_to[n], dtype=np.int64))
            
    def Solve(self, N=10, iter=100000, timeout=30, pop_size=8, Mu=0.001, Cr=0.05):
        best = np.inf
        result = []
        # pop_size = 8
        if self.Benchmark.InfCapacity:
            DE = DE_parallel(NumberOfIteration = iter, ObjectiveFunction = obj_function_infcap, variant="DE/rand/1/bin", F=Mu, Cr=Cr, ShowError=False, Timeout=timeout, ShowResult=False, Logging=False, FastLogging=False, LogInterval=1, AgeLimit=np.inf, obj_func_args=[self.Benchmark.node_init_pos, self.start_after, self.end_before,  self.constant_opID2op, self.constant_jobs_nb, self.constant_trans_site_info_nb, self.constant_node_from_to_nb])
            # DE = DE_parallel(NumberOfIteration = iter, ObjectiveFunction = obj_function_infcap, variant="DE/rand/1/bin", F=Mu, Cr=Cr, ShowError=False, Timeout=timeout, ShowResult=False, Logging=False, FastLogging=False, LogInterval=100, AgeLimit=np.inf, obj_func_args=[self.Benchmark.node_init_pos, self.start_after, self.end_before,  self.constant_opID2op, self.constant_jobs_nb, self.constant_trans_site_info_nb, self.constant_node_from_to_nb])
        else:
            DE = DE_parallel(NumberOfIteration = iter, ObjectiveFunction = obj_function_cap, variant="DE/rand/1/bin", F=Mu, Cr=Cr, ShowError=False, Timeout=timeout, ShowResult=False, Logging=False, FastLogging=False, LogInterval=1, AgeLimit=np.inf, obj_func_args=[self.Benchmark.node_init_pos, self.start_after, self.end_before,  self.constant_opID2op, self.constant_jobs_nb, self.constant_trans_site_info_nb, self.constant_node_from_to_nb])
            # DE = DE_parallel(NumberOfIteration = iter, ObjectiveFunction = obj_function_cap, variant="DE/rand/1/bin", F=Mu, Cr=Cr, ShowError=False, Timeout=timeout, ShowResult=False, Logging=False, FastLogging=False, LogInterval=100, AgeLimit=np.inf, obj_func_args=[self.Benchmark.node_init_pos, self.start_after, self.end_before,  self.constant_opID2op, self.constant_jobs_nb, self.constant_trans_site_info_nb, self.constant_node_from_to_nb])
        DE.VariableBounds = np.array([[0,self.total_num_ops]]*self.total_num_ops)
        
        for n in range(N):
            DE.Reset()
            DE.Population = np.stack([np.random.permutation(self.total_num_ops) for _ in range(pop_size)], axis=0)
            DE.Run()
            result += (DE.ElapseTime, DE.BestObj, [[l for l in log] for log in DE.Log], DE.ObjectiveValues.copy(), DE.Population.copy()),
            # if DE.BestObj < 209:
            #     obj_function_infcap_debug(DE.BestSol, self.Benchmark.node_init_pos, self.start_after, self.end_before,  self.constant_opID2op, self.constant_jobs_nb, self.constant_trans_site_info_nb, self.constant_node_from_to_nb)
            if DE.BestObj < best:
                best = DE.BestObj
            print(f"DE {n+1} Best:", DE.BestObj, "Time:", DE.ElapseTime, "Iter:", DE._iteration)
                
        return best, result
    
    def Decode(self, solution):
        decode_ids = np.full(solution.shape[0],-1, dtype=np.int64)
        for i in range(solution.shape[0]):
            decode_ids[solution[i]] = i
        assert decode_ids.min() == 0
        if self.Benchmark.InfCapacity:
            return InfCap_MHSP_Simulate_explicit(decode_ids, self.Benchmark.node_init_pos, self.start_after, self.end_before,  self.constant_opID2op, self.constant_jobs_nb, self.constant_trans_site_info_nb, self.constant_node_from_to_nb)
        else:
            return FiniteCap_MHSP_Simulate_explicit(decode_ids, self.Benchmark.node_init_pos, self.start_after, self.end_before,  self.constant_opID2op, self.constant_jobs_nb, self.constant_trans_site_info_nb, self.constant_node_from_to_nb)
            