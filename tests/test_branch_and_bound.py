import numpy as np
import pytest

from intprog.branch_and_bound import BranchAndBound
from tests.problems import TEST_PROBLEMS


@pytest.mark.parametrize("problem", TEST_PROBLEMS)
def test_branch_and_bound(problem):
    solver = BranchAndBound(
        c=problem.c, A=problem.A, b=problem.b, lb=problem.lb, ub=problem.ub
    )
    res = solver.solve()
    assert np.isclose(res.x, problem.x, atol=1e-4).all()


if __name__ == "__main__":
    test_branch_and_bound(TEST_PROBLEMS[-1])
