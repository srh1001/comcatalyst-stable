import numpy as np
from typing import Any, Optional

from comcatalyst.vars.vars_metadata import VarsMetadata

def calc_gp_objective_01(
    x: np.ndarray,
    vars_metadata: VarsMetadata,
    z: Any,
    sigma_noise: float,
    rng: Optional[np.random.Generator] = None,
    _F: dict = {0: lambda x: np.sin(5*x), 1: lambda x: np.tanh(0.3*x)},
    _G: dict = {0: lambda x: x - np.cos(3*x), 1: lambda x: -x**2},
    _SCALE_Q1: dict = {0:  1, 1: 1},
    _SCALE_Q2: dict = {0:  1, 1: 1} 
    ) -> np.ndarray:
    """
    Observation bruitée : y = h(X) + eps.
    Mettre sigma_noise = 0 ou None pour calculer h_true, i.e. h non bruitée.
    """
    x = np.atleast_2d(x)  # garantit shape (n, p) meme si x shape (p,) car 1 point
    n = x.shape[0]
    h_true = np.full((n, 1), np.nan)

    for i in range(n):
        qual1_idx = int(x[i, 0])
        qual2_idx = int(x[i, 1])
        quant1_val = float(x[i, 2])
        quant2_val = float(x[i, 3])

        h = (
            _SCALE_Q1[qual1_idx] * _F[qual1_idx](quant1_val + quant2_val)
            + _SCALE_Q2[qual2_idx] * _G[qual2_idx](quant1_val + quant2_val)
        )

        if sigma_noise is not None and sigma_noise > 0:
            if rng is None:
                rng = np.random.default_rng() # pas reproductible
            noise = rng.normal(0.0, sigma_noise)
            h += noise

        h_true[i, 0] = h

    return h_true

def calc_gp_objective_02(
    x: np.ndarray,
    vars_metadata: VarsMetadata,
    z: Any,
    sigma_noise: float,
    rng: Optional[np.random.Generator] = None,
    _F: dict = {
        0: lambda x: np.sin(5*x),
        1: lambda x: np.tanh(0.3*x),
        2: lambda x: 1 - x**2,
        3: lambda x: 0.5*np.sin(5*x) - 0.5*x
    },
    _G: dict = {
        0: lambda x: x - np.cos(3*x),
        1: lambda x: -x**2,
        2: lambda x: x**3-np.exp(x),
        3: lambda x: 1/(1+np.exp(5*x)) - 0.5,
    },
    _SCALE_Q1: dict = {0: 1, 1: 1, 2: 1, 3: 1},
    _SCALE_Q2: dict = {0: 1, 1: 1, 2: 1, 3: 1},
) -> np.ndarray:
    """
    Observation bruitée : y = h(X) + eps.
    Mettre sigma_noise = 0 ou None pour calculer h_true.
    """
    x = np.atleast_2d(x)
    n = x.shape[0]
    h_true = np.full((n, 1), np.nan)

    for i in range(n):
        qual1_idx = int(x[i, 0])
        qual2_idx = int(x[i, 1])
        quant1_val = float(x[i, 2])
        quant2_val = float(x[i, 3])

        h = (
            _SCALE_Q1[qual1_idx] * _F[qual1_idx](quant1_val + quant2_val)
            + _SCALE_Q2[qual2_idx] * _G[qual2_idx](quant1_val + quant2_val)
        )

        if sigma_noise is not None and sigma_noise > 0:
            if rng is None:
                rng = np.random.default_rng()
            h += rng.normal(0.0, sigma_noise)

        h_true[i, 0] = h

    return h_true

def calc_gp_objective_03(
    x: np.ndarray,
    vars_metadata: Any,
    z: Any,
    sigma_noise: float,
    rng: Optional[np.random.Generator] = None,
    _F = {
        0: lambda q1, q2, q3: -(q1 + 0.8)**2 - (q2 - 0.7)**2 - (q3 + 0.6)**2 + 2.5,
        1: lambda q1, q2, q3: -(q1 - 0.7)**2 - (q3 + 0.6)**2 - (q2 - 0.8)**2 + 0.5,
        2: lambda q1, q2, q3: -(q1 - 0.0)**2 - (q2 - 0.8)**2 - (q3 + 0.3)**2 + 0.2,
        3: lambda q1, q2, q3: -(q2 - 0.5)**2 - (q2 + 0.4)**2 - (q3 - 0.5)**2 + 0.3,
    },
    _G = {
        0: lambda q1, q2, q3: -(q1 - 0.6)**2 - (q2 - 0.5)**2 - (q3 - 0.7)**2 + 2.0,
        1: lambda q1, q2, q3: -(q1 + 0.5)**2 - (q2 + 0.7)**2 - (q3 + 0.4)**2 + 0.3,
        2: lambda q1, q2, q3: -(q1 - 0.3)**2 - (q2 - 0.2)**2 - (q3 + 0.6)**2 + 0.1,
        3: lambda q1, q2, q3: -(q1 + 0.4)**2 - (q2 - 0.6)**2 - (q3 - 0.3)**2 + 0.0,
    },
    _SCALE_Q1: dict = {0: 1, 1: 1, 2: 1, 3: 1},
    _SCALE_Q2: dict = {0: 1, 1: 1, 2: 2, 3: 1},
) -> np.ndarray:
    x = np.atleast_2d(x)
    n = x.shape[0]
    h_true = np.full((n, 1), np.nan)

    for i in range(n):
        qual1_idx = int(x[i, 0])
        qual2_idx = int(x[i, 1])
        quant1_val = float(x[i, 2])
        quant2_val = float(x[i, 3])
        quant3_val = float(x[i, 4])

        h = (
            _SCALE_Q1[qual1_idx] * _F[qual1_idx](quant1_val, quant2_val, quant3_val)
            + _SCALE_Q2[qual2_idx] * _G[qual2_idx](quant1_val, quant2_val, quant3_val)
        )

        if sigma_noise is not None and sigma_noise > 0:
            if rng is None:
                rng = np.random.default_rng()
            h += rng.normal(0.0, sigma_noise)

        h_true[i, 0] = h

    return h_true