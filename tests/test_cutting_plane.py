import numpy as np
import pytest

from intprog.cutting_plane import CuttingPlane
from tests.problems import TEST_PROBLEMS


@pytest.mark.parametrize("problem", TEST_PROBLEMS)
def test_cutting_plane(problem):
    solver = CuttingPlane(
        c=problem.c, A=problem.A, b=problem.b, lb=problem.lb, ub=problem.ub
    )
    res = solver.solve()
    assert np.isclose(res.x, problem.x, atol=1e-4).all()
