import numpy as np
from linprog.dual_solvers import DualRevisedSimplexSolver 
from linprog.special_solvers import PrimalDualAlgorithm
from linprog.preprocessing import ProblemPreprocessingUtils


from dataclasses import dataclass

@dataclass
class StandardFormLinProg():
    basis: np.array
    c: np.array
    A: np.array
    b: np.array

def is_integer(x: np.array):
    return np.equal(np.mod(x, 1), 0)

def get_fractional_part(x: np.array):
    return x - np.array(x).astype(np.int32)

def get_most_frational_part(x: np.array):
    fractional_part = get_fractional_part(x)
    return np.array([max(f, 1-f) for f in fractional_part])


class BranchAndBound():
    def __init__(self, c, A, b, lb, ub):
        self.c, self.A, self.b = ProblemPreprocessingUtils.add_variables_bounds_to_coefficient_matrix(c, A, b, lb, ub)
        self.problems = list()
        self.best_feasible_soln = None
        self.optimal_cost_upper_bound = np.inf
        self.optimal_cost_lower_bound = np.inf
        self.counter = -1
        self.optimum = False
        
        
    def solve(self, maxiters:int=100, tol:float=1e-3):
        solver = PrimalDualAlgorithm(self.c, self.A, self.b)
        res = solver.solve()
        
        sub_problem = StandardFormLinProg(
            res.basis,
            np.array(self.c),
            np.array(self.A),
            np.array(self.b),
        )
        
        self.problems.append(sub_problem)
        
        while len(self.problems) > 0 and self.counter < maxiters:
            self.counter += 1
            sub_problem = self.problems.pop()
            solver = DualRevisedSimplexSolver(
                sub_problem.c,
                sub_problem.A,
                sub_problem.b,
                sub_problem.basis,
            )
            res = solver.solve()
            if res.cost < self.optimal_cost_lower_bound:
                self.optimal_cost_lower_bound = res.cost
            fractional_parts = get_fractional_part(res.x)
            integer_vars = is_integer(fractional_parts)
            if integer_vars.all(): # this would mean soln is actually optimal? -> no only in original probelm before splits
                if res.cost < self.optimal_cost_upper_bound:
                    self.optimal_cost_upper_bound = res.cost
            else:
                split_idx = np.argmax(~integer_vars)
                lb_ceil = np.repeat(0.0, sub_problem.A.shape[1])
                ub_floor = np.repeat(np.inf, sub_problem.A.shape[1])
                lb_ceil[split_idx] = np.ceil(res.x[split_idx])
                ub_floor[split_idx] = np.floor(res.x[split_idx])
                
                c_ceil, A_ceil, b_ceil = ProblemPreprocessingUtils.add_variables_bounds_to_coefficient_matrix(
                    sub_problem.c,
                    sub_problem.A,
                    sub_problem.b,
                    lb_ceil,
                    None
                )
                
                c_floor, A_floor, b_floor = ProblemPreprocessingUtils.add_variables_bounds_to_coefficient_matrix(
                    sub_problem.c,
                    sub_problem.A,
                    sub_problem.b,
                    None,
                    ub_floor,
                )
                
                self.problems.append(
                    StandardFormLinProg(
                        np.append(res.basis, sub_problem.A.shape[1]), c_ceil, A_ceil, b_ceil
                    )
                )
                
                self.problems.append(
                    StandardFormLinProg(
                        np.append(res.basis, sub_problem.A.shape[1]), c_floor, A_floor, b_floor
                    )
                )
                
                
        


if __name__ == "__main__":
    # pg. 95 integer programming Wosley
    c = np.array([-4, 1, 0, 0, 0])
    A = np.array([[7, -2], [0, 1], [2, -2]])
    A = np.hstack([A, np.eye(A.shape[0])])
    b = np.array([14, 3, 3])
    lb = np.array([1, 1, 0, 0, 0])
    ub = np.repeat(np.inf, A.shape[1])
    
    solver = BranchAndBound(c, A, b, lb, ub)
    res = solver.solve()