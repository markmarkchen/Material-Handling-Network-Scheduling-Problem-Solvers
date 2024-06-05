from statistics import fmean, stdev
import numpy as np
# import editdistance
class logger:
    def __init__(self, **kwargs) -> None:
        '''
        BestSol:        Save the evolution of the best solution so far.
        BestObj:        Save the evolution of the objective value of the best solution so far.
        Pop:            Save the evolution of the population.
        IterBestSol:    Save the evolution of the best solution in each iteration.
        IterBestObj:    Save the evolution of the objective value of the best solution in each iteration.
        IterObjAvg:     Save the evolution of the averaged objective value in each iteration.
        IterObjSTD:     Save the evolution of the standard deviation of objective values in each iteration.
        '''
        self.BestSol = kwargs.get("BestSol", False)
        self.BestObj = kwargs.get("BestObj", True)
        self.Pop = kwargs.get("Pop", False)
        self.PopSTD = kwargs.get("PopSTD", False)
        self.PopMean = kwargs.get("PopMean", False)
        self.IterObj = kwargs.get("IterObj", False)

        self.IterBestSol = kwargs.get("IterBestSol", False)
        self.IterBestObj = kwargs.get("IterBestObj", False)
        self.IterObjAvg = kwargs.get("IterObjAvg", True)
        self.IterObjSTD = kwargs.get("IterObjSTD", True)
        self.RankTurnover = kwargs.get("Turnover", False)
        self.Ages = kwargs.get("Ages", False)
        self.Ranks = None
        
    def Log(self, Copy, BestSol, BestObj, IterBestSol, IterBestObj, Population, PopObjs, Ages):
        log = []
        logstr = ""
        if self.BestSol:
            log.append(Copy(BestSol))
        if self.BestObj:
            log.append(BestObj)
            logstr += f"Best: {BestObj:.8f}    "
        if self.Pop:
            log.append([Copy(p) for p in Population])
        if self.IterBestSol:
            log.append(Copy(IterBestSol))
        if self.IterBestObj:
            log.append(IterBestObj)
            logstr += f"IterBest: {IterBestObj:.8f}    "
        if self.IterObjAvg:
            log.append(fmean(PopObjs))
            # log.append(np.average(PopObjs, weights=Ages))
            logstr += f"IterAvg: {log[-1]:.8f}    "
        if self.IterObjSTD:
            log.append(stdev(PopObjs))
            # log.append(np.sqrt(np.average((PopObjs-np.average(PopObjs, weights=Ages))**2, weights=Ages)))
            logstr += f"IterSTD: {log[-1]:.8f}    "
        # if self.RankTurnover:
        #     if self.Ranks is None:
        #         self.Ranks = np.argsort(PopObjs)
        #     current = np.argsort(PopObjs)
        #     turnover = editdistance.eval(self.Ranks, current)
        #     self.Ranks = current
        #     log.append(turnover)
        #     logstr += f"Turnover: {turnover}    "
        if self.PopMean:
            log += Population.mean(axis=0),
            logstr += f"\n\t\tPopulation Mean:\t[{', '.join([f'{l:.2f}' for l in log[-1].tolist()])}]    "
        if self.PopSTD:
            log += Population.std(axis=0),
            logstr += f"\n\t\tPopulation STD: \t[{', '.join([f'{l:.2f}' for l in log[-1].tolist()])}]    "
        if self.Ages:
            log += Ages,
            logstr += f"\n\t\tPopulation Ages:\t{Ages.tolist()}    "
        if self.IterObj:
            log += PopObjs,
            logstr += f"\n\t\tPopulation Obj: \t[{', '.join([f'{l:.4f}' for l in log[-1].tolist()])}]    "
        logstr += '\n'
        return log, logstr