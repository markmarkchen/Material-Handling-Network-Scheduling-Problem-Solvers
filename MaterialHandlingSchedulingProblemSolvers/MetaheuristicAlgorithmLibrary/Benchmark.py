import numpy as np

class MinimizationProblem:
    def __init__(self, num_variable, bounds, penaltyFactor = 1e6):
        self.num_variable = num_variable
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)
        self.VariableBounds = bounds
        self.PenaltyFactor = penaltyFactor
    
    def DefaultPopluationInit(self, N):
        return self.Repair(np.random.rand(N, len(self.VariableBounds)) * (self.VariableBounds[:,1] - self.VariableBounds[:,0]) + self.VariableBounds[:,0])
    
    def ObjectiveFunction(self, x) -> float:
        raise NotImplementedError
    
    def Repair(self, x):
        x = np.maximum(x, self.VariableBounds[:,0])
        x = np.minimum(x, self.VariableBounds[:,1])
        return x

class PressureVesselProblem(MinimizationProblem):
    def __init__(self, penaltyFactor = 1e6):
        '''
            x1: 
                Ts (thickness of the shell) 
                Range: [1.1, 99]
                Type: discrete (multiple of 0.0625)
            x2: 
                Th (thickness of the head)
                Range: [0.6, 99]
                Type: discrete (multiple of 0.0625)
            x3: 
                R (inner radius)
                Range: [50, 70]
                Type: continuous
            x4: 
                L (length of the cylindrical section of the vessel, not including the head)
                Range: [30, 50]
                Type: continuous
        '''
        # super().__init__(4, [[1.125,99],[0.625,99],[50,70],[30,50]], penaltyFactor)
        super().__init__(4, [[0.0625,99*0.0625],[0.0625,99*0.0625],[10,200],[10,200]], penaltyFactor)

    def ObjectiveFunction(self, x) -> float:
        assert len(x) == self.num_variable \
                and (isinstance(self.PenaltyFactor, float) or isinstance(self.PenaltyFactor, int) or len(self.PenaltyFactor) == self.num_variable)
        # obj = 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]*x[2] + 3.1661*x[0]*x[0]*x[3] + 19.8621*x[0]*x[0]*x[2]
        obj = 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]*x[2] + 3.1661*x[0]*x[0]*x[3] + 19.84*x[0]*x[0]*x[2]
        constraint = np.array([-x[0]+0.0193*x[2], -x[1]+0.00954*x[2], -np.pi*(x[2]*x[2]*x[3]+4/3*x[2]*x[2]*x[2])+1296000, x[3]-240])
        return obj + self.PenaltyFactor * np.abs(constraint[constraint > 0]).sum()
    
    def Repair(self, x):
        x[[0,1]] = 0.0625 * np.trunc(x[[0,1]]/0.0625)
        return super().Repair(x)

    def FullObjFunction(self, x):
        assert len(x) == self.num_variable \
                and (isinstance(self.PenaltyFactor, float) or isinstance(self.PenaltyFactor, int) or len(self.PenaltyFactor) == self.num_variable)
        # obj = 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]*x[2] + 3.1661*x[0]*x[0]*x[3] + 19.8621*x[0]*x[0]*x[2]
        obj = 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]*x[2] + 3.1661*x[0]*x[0]*x[3] + 19.84*x[0]*x[0]*x[2]
        constraint = np.array([-x[0]+0.0193*x[2], -x[1]+0.00954*x[2], -np.pi*(x[2]*x[2]*x[3]+4/3*x[2]*x[2]*x[2])+1296000, x[3]-240])
        return obj, constraint

class SpringDesignProblem(MinimizationProblem):
    def __init__(self, penaltyFactor = 1e6):
        super().__init__(3,[] , penaltyFactor)

