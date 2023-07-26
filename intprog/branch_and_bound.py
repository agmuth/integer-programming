import numpy as np
from linprog.dual_solvers import DualRevisedSimplexSolver
from linprog.exceptions import DualIsUnboundedError
from linprog.preprocessing import ProblemPreprocessingUtils
from linprog.special_solvers import PrimalDualAlgorithm
from sortedcontainers import SortedList

from intprog.data_classes import IntProgReturnObj, StandardFormLinProg
from intprog.utils import get_index_of_most_fractional_variable, is_integer


class BranchAndBound:
    def __init__(self, c, A, b, lb, ub):
        (
            self.c,
            self.A,
            self.b,
        ) = ProblemPreprocessingUtils.add_variables_bounds_to_coefficient_matrix(
            c, A, b, lb, ub
        )
        self.m, self.n = A.shape
        self.problems = list()
        self.best_feasible_soln = None
        self.optimal_cost_upper_bound = np.inf
        self.optimal_cost_lower_bound = np.inf
        self.counter = -1
        self.optimum = False

    def solve(self, maxiters: int = 100, tol: float = 1e-3):
        solver = PrimalDualAlgorithm(self.c, self.A, self.b)
        res = solver.solve()

        sub_problem = StandardFormLinProg(
            res.basis,
            np.array(self.c),
            np.array(self.A),
            np.array(self.b),
        )

        # append problem and lower bound on cost as long as `self.problems` is a normal list
        # appending to and popping from the end corresponds to a depth first search strategy
        self.problems += [(-1 * res.cost, sub_problem)]

        while len(self.problems) > 0 and self.counter < maxiters:
            self.counter += 1
            # depth first if `depth_first_search_flag`
            # best soln if `best_soln_search_flag`
            _, sub_problem = self.problems.pop()
            solver = DualRevisedSimplexSolver(
                sub_problem.c,
                sub_problem.A,
                sub_problem.b,
                sub_problem.basis,
            )
            try:
                res = solver.solve()
            except DualIsUnboundedError:
                # primal is infeasible -> prune by infeasibility
                continue

            if res.cost <= self.optimal_cost_upper_bound:
                # prune by bound (if above cond. is not true)
                integer_vars = is_integer(res.x[: self.n])
                if integer_vars.all():
                    # prune by optimality
                    self.optimal_cost_upper_bound = res.cost
                    self.best_feasible_soln = res.x[
                        : self.n
                    ]  # omit added vars from branching

                    self.problems = SortedList([problem for problem in self.problems])
                    # self.problems is now a sorted list + therefore calling `.pop`
                    # pops the subproblem with the best (best_soln_search_flag)
                    # lower bound on the cost of the optimal soln

                else:
                    # split along non-integer variable and add two new subproblems to list
                    split_idx = get_index_of_most_fractional_variable(res.x)
                    lb_ceil = np.repeat(0.0, sub_problem.A.shape[1])
                    ub_floor = np.repeat(np.inf, sub_problem.A.shape[1])
                    lb_ceil[split_idx] = np.ceil(res.x[split_idx])
                    ub_floor[split_idx] = np.floor(res.x[split_idx])
                    updated_problem_dual_basis = np.append(
                        res.basis, sub_problem.A.shape[1]
                    )

                    self.problems += [
                        (
                            -1
                            * res.cost,  # `res.cost` is a lower bound on the optimal soln of the sub problem
                            StandardFormLinProg(
                                updated_problem_dual_basis,
                                *ProblemPreprocessingUtils.add_variables_bounds_to_coefficient_matrix(
                                    sub_problem.c,
                                    sub_problem.A,
                                    sub_problem.b,
                                    lb_ceil,
                                    None,
                                )
                            ),
                        )
                    ]

                    self.problems += [
                        (
                            -1 * res.cost,
                            StandardFormLinProg(
                                updated_problem_dual_basis,
                                *ProblemPreprocessingUtils.add_variables_bounds_to_coefficient_matrix(
                                    sub_problem.c,
                                    sub_problem.A,
                                    sub_problem.b,
                                    None,
                                    ub_floor,
                                )
                            ),
                        )
                    ]
        self.optimum = len(self.problems) == 0
        return IntProgReturnObj(
            self.best_feasible_soln,
            self.optimal_cost_upper_bound,
            self.optimum,
            self.counter,
        )
