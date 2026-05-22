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
        _SCALE_Q1: dict = {0:  1, 1: 1},  # qual1 module l'amplitude de F
        _SCALE_Q2: dict = {0:  1, 1: 1}   # qual2 module l'amplitude de G
    ) -> np.ndarray:
        """
        Observation bruitée : y = h(X) + eps.
        Mettre sigma_noise = 0 ou None pour calculer h_true, i.e. h non bruitée.
        """
        x = np.atleast_2d(x)  # garantit shape (n, p) meme si x shape (p,) car 1 point
        n = x.shape[0]
        h_true = np.full(n, np.nan)

        for i in range(n):
            qual1_idx  = int(x[i, 0])
            qual2_idx  = int(x[i, 1])
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

            h_true[i] = h

        return h_true