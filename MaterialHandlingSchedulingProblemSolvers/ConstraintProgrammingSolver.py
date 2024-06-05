from docplex.cp.model import *
from docplex.cp.config import context
from .BenchmarkParser import operation, site_operation, MHSP_Benchmark
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class Job:
    id = 0
    def __init__(self, Benchmark: MHSP_Benchmark, start_after=0, end_before=INTERVAL_MAX) -> None:
        self.id = self.__class__.id
        self.__class__.id += 1
        self.Benchmark = Benchmark
        
        self.options = []
        self.start_after = start_after
        self.end_before = end_before
        self.overall_intv = interval_var(name=f"J{self.id} Overall", start=(self.start_after, INTERVAL_MAX), end=(0,self.end_before), size=(0,self.end_before-self.start_after), optional=False)
        self.option_intv = []
        self.operation_intv = []
        self.option_seq = []
        
    def add_option(self, option):
        op_id = len(self.option_intv)
        self.options.append(option)
        self.option_intv += interval_var(name=f"J{self.id} Option {op_id}", start=(self.start_after, INTERVAL_MAX), end=(0, self.end_before), optional=True),
        
        ops = []
        pts = np.array([op.pt for op in option])
        min_total_pt = pts.sum()
        
        # add operations
        for i, op in enumerate(option):
            if isinstance(op, operation):
                ops += interval_var(name=f"J{self.id} Option {op_id} Operation {i} (Node {op.Node} From {op.From} To {op.To})", start=(self.start_after+pts[:i].sum(), self.end_before-pts[i:].sum()), end=(self.start_after+pts[:i+1].sum(), self.end_before), size=(op.pt, max(self.end_before-self.start_after-min_total_pt+op.pt, op.pt)), optional=True),
            if isinstance(op, site_operation):
                ops += interval_var(name=f"J{self.id} Option {op_id} Operation {i} (From {op.Inbound_Node} {op.Inbound_pd} To {op.Outbound_Node} {op.Outbound_pd})", start=(self.start_after+pts[:i].sum(), self.end_before-pts[i:].sum()), end=(self.start_after+pts[:i+1].sum(), self.end_before), size=(op.pt, max(self.end_before-self.start_after-min_total_pt+op.pt, op.pt)), optional=True),
        # if min_total_pt <= self.overall_intv.size[0]:
        #     self.overall_intv.size = (min_total_pt, self.overall_intv.size[1])
        self.operation_intv += ops,
        self.option_seq += sequence_var(ops, name=f"J{self.id} Option {op_id} Sequence"),
        return ops
    
    def get_constraints(self):
        if len(self.option_intv) == 0:
            return []
        consts = []
        
        # only one option exists
        consts += alternative(self.overall_intv, self.option_intv),
        
        # if the option is present, all operations are present
        for i, operations in enumerate(self.operation_intv):
            consts += span(self.option_intv[i], operations),
            for j in range(1, len(operations)-1):
                consts += (presence_of(operations[j]) == presence_of(self.option_intv[i])),
                consts += (presence_of(operations[j]) == presence_of(operations[j-1])),
                consts += (presence_of(operations[j]) == presence_of(operations[j+1])),

        # no overlap within option
        for seq in self.option_seq:
            consts += no_overlap(seq),
            
        # start-after
        consts += (start_of(self.overall_intv) >= self.start_after),
        
        # end-before
        consts += (end_of(self.overall_intv) <= self.end_before),
        
        # execute one-by-one
        for option in self.operation_intv:
            for i in range(1,len(option)):
                consts += end_at_start(option[i-1], option[i]),
                
        # right process time
        for i, option in enumerate(self.options):
            for j, op in enumerate(option):
                consts += if_then(presence_of(self.operation_intv[i][j]), length_of(self.operation_intv[i][j]) >= op.pt),
                 
        return consts
    
    def verify(self, msol):
        if len(self.option_intv) == 0:
            return
        
        obj = msol.get_objective_value()
        
        # overall
        sol = msol.get_var_solution(self.overall_intv.get_name())
        assert sol.is_present()
        
        overall_intv_sol = sol.get_value()
        
        # only one option exists
        opt = 0
        opt_id = -1
        for i, intv in enumerate(self.option_intv):
            opt_present = msol.get_var_solution(intv.get_name()).is_present()
            if opt_present:
                opt += 1
                opt_id = i
            
            # if the option is present, all operations are present
            s = sum([msol.get_var_solution(op.get_name()).is_present() for op in self.operation_intv[i]])
            assert (s == 0 and not opt_present) or (s == len(self.operation_intv[i]) and opt_present)
                
        assert opt == 1
        option_intv_sol = msol.get_var_solution(self.option_intv[opt_id].get_name())
        assert option_intv_sol.start == overall_intv_sol.start
        assert option_intv_sol.end == overall_intv_sol.end
        
        #start-after
        assert overall_intv_sol.start >= self.start_after
        first_intv_sol = msol.get_var_solution(self.operation_intv[opt_id][0].get_name())
        assert first_intv_sol.end <= obj
        assert first_intv_sol.start >= self.start_after
        assert first_intv_sol.start == overall_intv_sol.start
        
        # end-before
        last_intv_sol = msol.get_var_solution(self.operation_intv[opt_id][-1].get_name())
        assert last_intv_sol.end <= obj
        assert last_intv_sol.end <= self.end_before
        assert last_intv_sol.end == overall_intv_sol.end
        
        operation_intvs_sol = [msol.get_var_solution(op.get_name()).get_value() for op in self.operation_intv[opt_id]]
        
        # execute one-by-one
        for i in range(1,len(operation_intvs_sol)):
            assert operation_intvs_sol[i-1].end == operation_intvs_sol[i].start
        
        # end-start >= pt
        for i, op in enumerate(self.options[opt_id]):
            assert operation_intvs_sol[i].end - operation_intvs_sol[i].start >= op.pt
    
    def __repr__(self):
        text = f"Job {self.id}:\n"
        text += f"\tOverall: {self.overall_intv}\n\n\tOptions:\n"
        for i, option in enumerate(self.option_intv):
            text += f"\t{option}\n"
            text += "\t\t"
            for op in self.operation_intv[i]:
                text += f"{op}, "
            text += "\n\n"
        return text
    
    def optiaml_path(self, msol):
        for i, intv in enumerate(self.option_intv):
            opt_present = msol.get_var_solution(intv.get_name()).is_present()
            if opt_present:
                return i
    def optimal_option(self, msol):
        for i, intv in enumerate(self.option_intv):
            opt_present = msol.get_var_solution(intv.get_name()).is_present()
            if opt_present:
                print(self.option_intv[i])
    def optimal_process_time(self, msol):
        return msol.get_var_solution(self.overall_intv.get_name()).get_value().size

class Node:
    id = 0
    def __init__(self, Benchmark: MHSP_Benchmark) -> None:
        self.id = self.__class__.id
        self.__class__.id += 1
        self.Benchmark = Benchmark
        
        self.seq = None
        self.seq_start_end = None
        
        self.operations = []
        self.intvs = []
        self.intvs_start = []
        self.intvs_end = []
        
        self.node_init_intv = interval_var(name=f"Node {self.id} Init", start=(0,0), end=(0,0), size=(0,0), optional=False)
        self.init_pos = self.Benchmark.node_init_pos
        
        self.intv_start_types = []
        self.intv_end_types = []
        self.name2type = {f"Node {self.id} Init": self.init_pos}
        
    def add_intv(self, intv, operation):
        self.intvs += intv,
        self.operations += operation,
        
        self.intvs_start += interval_var(name=f"{intv.name} start", optional=True, size=0),
        self.intvs_end += interval_var(name=f"{intv.name} end", optional=True, size=0),
        
        self.intv_start_types += operation.From,
        self.intv_end_types += operation.To,
        
        self.name2type[self.intvs_start[-1].get_name()] = operation.From
        self.name2type[self.intvs_end[-1].get_name()] = operation.To
        
    def get_constraints(self):
        consts = []
        
        self.seq = sequence_var(self.intvs, name=f"Node {self.id} seq")
        # no overlap
        consts += no_overlap(self.seq),
        
        self.seq_start_end = sequence_var([self.node_init_intv]+self.intvs_start+self.intvs_end, [self.init_pos]+self.intv_start_types+self.intv_end_types ,name=f"Node {self.id} seq (start & end)")
        # no overlap (moving time)
        consts += no_overlap(self.seq_start_end, self.Benchmark.node_from_to[self.id], is_direct=True),
        consts += first(self.seq_start_end, self.node_init_intv),
        
        for i in range(len(self.intvs)):
            consts += span(self.intvs[i], [self.intvs_start[i], self.intvs_end[i]]),
            consts += end_before_start(self.intvs_start[i], self.intvs_end[i]),
            consts += presence_of(self.intvs_start[i]) == presence_of(self.intvs_end[i]),
                
        return consts
    
    def verify(self, msol):
        if len(self.intvs) == 0:
            return
        
        # check presence
        for i in range(len(self.intvs)):
            intv_sol = msol.get_var_solution(self.intvs[i].get_name())
            intv_start_sol = msol.get_var_solution(self.intvs_start[i].get_name())
            intv_end_sol = msol.get_var_solution(self.intvs_end[i].get_name())
            
            assert (intv_sol.is_present() == intv_start_sol.is_present()) and (intv_sol.is_present() == intv_end_sol.is_present())
            if intv_sol.is_present():
                # align with intv
                assert intv_sol.get_value().start == intv_start_sol.get_value().start
                assert intv_sol.get_value().end == intv_end_sol.get_value().end
                # 0 size
                assert intv_start_sol.get_value().start == intv_start_sol.get_value().end
                assert intv_end_sol.get_value().start == intv_end_sol.get_value().end
        
        seq_sol = msol.get_var_solution(self.seq.get_name())
        seq_intvs = seq_sol.get_value()
        seq_start_end_sol = msol.get_var_solution(self.seq_start_end.get_name())
        seq_start_end_intvs = seq_start_end_sol.get_value()
        
        # check no-overlap
        for i in range(1, len(seq_intvs)):
            assert seq_intvs[i-1].end <= seq_intvs[i].start
            
        if len(seq_start_end_intvs) > 1:
            ft = self.Benchmark.node_from_to[self.id][self.init_pos][self.name2type[seq_start_end_intvs[1].get_name()]]
            assert  seq_start_end_intvs[1].start >= ft
        
        # check no-overlap (moving time)
        for i in range(1, len(seq_start_end_intvs)):
            first_intv = seq_start_end_intvs[i-1].get_value()
            first_type = self.name2type[seq_start_end_intvs[i-1].get_name()]
            second_intv = seq_start_end_intvs[i].get_value()
            second_tpye = self.name2type[seq_start_end_intvs[i].get_name()]
          
            assert first_intv.end <= second_intv.start
            assert (second_intv.start - first_intv.end) >= self.Benchmark.node_from_to[self.id][first_type][second_tpye]

            
    def __repr__(self) -> str:
        text = f"Node {self.id}:\n"
        text += f"\tIntvervals:\n"
        for i, intv in enumerate(self.intvs):
            text += f"\t{intv}, "
            text += f"\t{self.intvs_start[i]}, "
            text += f"\t{self.intvs_end[i]}\n"

        return text

# %%
class Site:
    id = 0
    def __init__(self, Benchmark: MHSP_Benchmark, capacity) -> None:
        self.id = self.__class__.id
        self.__class__.id += 1
        self.Beanchmark = Benchmark
        
        self.capacity = capacity
        self.InfCapacity = self.Beanchmark.InfCapacity
         
        self.intvs = []
        self.opertaions = []
    
    def add_intv(self, intv, operation):
        self.intvs += intv,
        self.opertaions += operation,
        
    def get_constraints(self):
        consts = []
        
        if not self.InfCapacity:
            cumfun = 0
            for intv in self.intvs:
                cumfun += pulse(intv, 1)    
            
            for intv in self.intvs:
                consts += always_in(cumfun, intv, 0, self.capacity),
        return consts
    
    def verify(self, msol):
        obj = msol.get_objective_value()
        if not self.InfCapacity:
            time_slots = np.zeros(obj)
            
            for intv in self.intvs:
                sol = msol.get_var_solution(intv.get_name())
                if sol.is_present():
                    intv_sol = sol.get_value()
                    time_slots[intv_sol.start:intv_sol.end] += 1
            
            assert np.all(time_slots <= self.capacity)
        
    def __repr__(self) -> str:
        text = f"Site {self.id}:\n"
        text += f"\tIntvervals:\n\t"
        text += ", ".join([str(intv) for intv in self.intvs])
        return text
        

# %%
class CP_Solver:
    def __init__(self, Beanchmark: MHSP_Benchmark) -> None:
        self.Benchmark = Beanchmark
        Job.id = 0
        Node.id = 0
        Site.id = 0

        self.JOBs = [Job(self.Benchmark, s, e) for s,e in zip(self.Benchmark.start_after, self.Benchmark.end_before)]
        self.NODEs = [Node(self.Benchmark) for _ in range(self.Benchmark.num_nodes)]
        self.SITES = {key: Site(self.Benchmark, value.capacity) for key, value in self.Benchmark.trans_site_info.items()}

        num_jobs = self.Benchmark.num_jobs
# %%
        for j in range(num_jobs):
            for i in range(len(self.Benchmark.jobs[j])):
                operations = self.Benchmark.jobs[j][i]
                intvs = self.JOBs[j].add_option(self.Benchmark.jobs[j][i])
                
                for i in range(len(operations)):
                    # node operation
                    if isinstance(operations[i], operation):
                        self.NODEs[operations[i].Node].add_intv(intvs[i], operations[i])
                    # site operation
                    if isinstance(operations[i], site_operation):
                        self.SITES[(operations[i].Inbound_Node, operations[i].Inbound_pd, operations[i].Outbound_Node, operations[i].Outbound_pd)].add_intv(intvs[i], operations[i])

    def Solve(self):
        context.params.DefaultInferenceLevel = "Extended"
        context.params.SequenceInferenceLevel = "Extended"
        context.params.NoOverlapInferenceLevel = "Extended"
        context.params.CumulFunctionInferenceLevel = "Extended"
        context.params.PrecedenceInferenceLevel = "Extended"
        context.params.IntervalSequenceInferenceLevel = "Extended"
        context.params.ConflictRefinerOnVariables = "On"
        model = CpoModel(name="TN")

# %%
        # model.add(JOBs[0].get_constraints())
        for JOB in self.JOBs:
            model.add(JOB.get_constraints())

        for NODE in self.NODEs:
            model.add(NODE.get_constraints())
            
        for SITE in self.SITES.values():
            model.add(SITE.get_constraints())
# %%
# minimize makespan
        model.add(minimize(max([end_of(JOB.overall_intv) for JOB in self.JOBs])))

# %%
        # msol = model.solve(TimeLimit=60, trace_log = False)
        # msol = model.solve(TimeLimit=60, trace_log = False, execfile=r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cpoptimizer\bin\x64_win64\cpoptimizer.exe", libfile=r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cpoptimizer\bin\x64_win64\ILOG.CP.dll")
        msol = model.solve(TimeLimit=60, trace_log = False, execfile="/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer")
        for JOB in self.JOBs:
            JOB.verify(msol)
        for NODE in self.NODEs:
            NODE.verify(msol)
        for SITE in self.SITES.values():
            SITE.verify(msol)
            
        return msol
        # msol.print_solution()
            
# msol = model.solve(TimeLimit=36000, trace_log = False, execfile="/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer")
# msol = model.solve(FailLimit=100000, TimeLimit=10, trace_log = False, execfile="/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer")


