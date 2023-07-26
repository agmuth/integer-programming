import numpy as np
from linprog.dual_solvers import DualRevisedSimplexSolver
from linprog.exceptions import DualIsUnboundedError
from linprog.preprocessing import ProblemPreprocessingUtils
from linprog.special_solvers import PrimalDualAlgorithm

from intprog.data_classes import IntProgReturnObj, StandardFormLinProg
from intprog.utils import get_index_of_most_fractional_variable, is_integer


class BranchAndBound:
    """Branch and bound algorithm for pure linear integer programming probelms."""

    def __init__(self, c, A, b, lb, ub):
        """Assumes LP is passed in in standard form (min c'x sbj. Ax = b, lb <= x <= ub)

        Parameters
        ----------
        c : np.array
            (n,) cost vector
        A : np.array
            (m, n) matrix defining linear combinations subject to equality constraints.
        b : np.array
            (m,) vector defining the equality constraints.
        lb : np.array
            (n,) vector specifying lower bounds on x. -np.inf indicates variable is unbounded below.
        ub : np.array
            (n,) vector specifying lower bounds on x. +np.inf indicates variable is unbounded above.
        """
        (
            self.c,
            self.A,
            self.b,
        ) = ProblemPreprocessingUtils.add_variables_bounds_to_coefficient_matrix(
            c, A, b, lb, ub
        )
        self.m, self.n = A.shape
        self.best_feasible_soln = None
        self.optimal_cost_upper_bound = np.inf
        self.optimal_cost_lower_bound = np.inf
        self.counter = -1
        self.optimum = False

    def _get_return_value(self):
        return IntProgReturnObj(
            self.best_feasible_soln,
            self.optimal_cost_upper_bound,
            self.optimum,
            self.counter,
        )

    def solve(self, maxiters: int = 100):
        sub_problems = list()
        solver = PrimalDualAlgorithm(self.c, self.A, self.b)
        res = solver.solve()

        sub_problem = StandardFormLinProg(
            res.basis,
            np.array(self.c),
            np.array(self.A),
            np.array(self.b),
        )

        # append problem and lower bound on cost as long as `sub_problems` is a normal list
        # appending to and popping from the end corresponds to a depth first search strategy
        sub_problems.append((res.cost, sub_problem))

        while len(sub_problems) > 0 and self.counter < maxiters:
            self.counter += 1
            # depth first if `depth_first_search_flag`
            # best soln if `best_soln_search_flag`
            _, sub_problem = sub_problems.pop()
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
                if is_integer(res.x[: self.n]).all():
                    # prune by optimality
                    self.optimal_cost_upper_bound = res.cost
                    self.best_feasible_soln = res.x[
                        : self.n
                    ]  # omit added vars from branching

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

                    sub_problems_to_add = [
                        (
                            # `res.cost` is a lower bound on the optimal soln of the sub problem
                            res.cost,
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
                        ),
                        (
                            res.cost,
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
                        ),
                    ]
                    sub_problems += sub_problems_to_add
                    if self.optimal_cost_upper_bound < np.inf:
                        # at least one feasible soln has been found 
                        # -> switch from depth first search to best first search
                        sub_problems.sort(reverse=True, key=lambda x: x[0])

        self.optimum = len(sub_problems) == 0
        return self._get_return_value()
