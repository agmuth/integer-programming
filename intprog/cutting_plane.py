import numpy as np
from linprog.dual_solvers import DualRevisedSimplexSolver
from linprog.special_solvers import PrimalDualAlgorithm

from intprog.branch_and_bound import BranchAndBound
from intprog.utils import (get_fractional_parts,
                           get_index_of_most_fractional_variable, is_integer)


class CuttingPlane(BranchAndBound):
    """Cutting plane algorithm for pure linear integer programming probelms."""

    def solve(self, maxiters: int = 100):
        self.counter += 1
        solver = PrimalDualAlgorithm(self.c, self.A, self.b)
        res = solver.solve()
        basis = res.basis
        while self.counter < maxiters:
            self.counter += 1
            solver = DualRevisedSimplexSolver(self.c, self.A, self.b, basis)
            res = solver.solve()

            if is_integer(res.x[: self.n]).all():
                # optimal soln found
                self.optimum = True
                self.optimal_cost_upper_bound = res.cost
                self.best_feasible_soln = res.x[
                    : self.n
                ]  # omit added vars from cutting planes
                break

            # otherwise add gomory cut + update basis
            m, n = self.A.shape
            basis = res.basis
            non_basic_vars = np.array([i for i in range(n) if i not in basis])
            inv_B = solver.inv_basis_matrix
            inv_B_at_b = inv_B @ self.b
            cut_index = get_index_of_most_fractional_variable(inv_B_at_b)

            A_cut = np.zeros(n + 1)
            A_cut[non_basic_vars] = get_fractional_parts(
                (inv_B @ solver.A)[cut_index, non_basic_vars]
            )
            A_cut[-1] = -1
            b_cut = get_fractional_parts(inv_B_at_b)[cut_index]
            c_cut = 0.0

            self.A = np.vstack([np.hstack([self.A, np.zeros((m, 1))]), A_cut])
            self.b = np.append(self.b, b_cut)
            self.c = np.append(self.c, c_cut)
            basis = np.append(basis, n)

        return self._get_return_value()
