from dataclasses import dataclass

import numpy as np


@dataclass
class StandardFormLinProg:
    basis: np.array
    c: np.array
    A: np.array
    b: np.array


@dataclass
class IntProgReturnObj:
    x: np.array
    cost: float
    optimum: bool
    iters: int
