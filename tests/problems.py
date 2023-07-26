from dataclasses import dataclass

import numpy as np


@dataclass
class IntProgTestProblem:
    c: np.array
    A: np.array
    b: np.array
    lb: np.array
    ub: np.array
    x: np.array


test_problem_1 = IntProgTestProblem(
    # pg. 95 integer programming wosley
    c=np.array([-4, 1, 0, 0, 0]),
    A=np.hstack([np.array([[7, -2], [0, 1], [2, -2]]), np.eye(3)]),
    b=np.array([14, 3, 3]),
    lb=np.array([1, 1, 0, 0, 0]),
    ub=np.repeat(np.inf, 5),
    x=np.array([2.0, 1.0, 2.0, 2.0, 1.0]),
)

test_problem_2 = IntProgTestProblem(
    # pg. 288 linear programming and network flows
    c=np.array([3, 4, 0, 0]),
    A=np.hstack([np.array([[3, 1], [1, 2]]), -1 * np.eye(2)]),
    b=np.array([4, 4]),
    # lb=np.array([1, 1, 0, 0]),
    lb=np.array([0, 0, 0, 0]),
    ub=np.repeat(np.inf, 4),
    x=np.array(
        [
            2.0,
            1.0,
            3.0,
            0.0,
        ]
    ),
)

test_problem_3 = IntProgTestProblem(
    # pg. 330 combinatorial optimization
    c=np.array([0, -1, 0, 0]),
    A=np.hstack([np.array([[3, 2], [-3, 2]]), np.eye(2)]),
    b=np.array([6, 0]),
    lb=np.array([0, 0, 0, 0]),
    ub=np.repeat(np.inf, 4),
    x=np.array([1, 1, 1, 1]),
)

TEST_PROBLEMS = [v for v in locals().values() if isinstance(v, IntProgTestProblem)]
