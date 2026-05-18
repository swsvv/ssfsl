import numpy as np
import scipy
from scipy.stats import t
from torch import Tensor


def mean_confidence_interval(
    data: list[Tensor], confidence: float = 0.95
) -> tuple[float, float]:
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2.0, n - 1)
    return m, h
