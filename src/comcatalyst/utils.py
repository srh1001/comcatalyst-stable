import numpy as np
import itertools
import numpy as np
from scipy.stats import norm as scipy_norm
from typing import List, Tuple, Optional, Any

from smt.surrogate_models.krg_based import KrgBased
from botorch.models import MixedSingleTaskGP

from .context.vars_metadata import VarsMetadata

def build_grid(vars_metadata: VarsMetadata, n_grid: int) -> np.ndarray:
    categ_vars_indexes = vars_metadata.get_categ_like_vars_indexes()
    cont_vars_indexes  = vars_metadata.get_continuous_vars_indexes()

    categ_modalities = [
        list(range(len(vars_metadata[i].support)))
        for i in categ_vars_indexes
    ]
    cont_grids = [
        np.linspace(vars_metadata[i].support[0], vars_metadata[i].support[1], n_grid)
        for i in cont_vars_indexes
    ]

    rows = []
    for categ_combo in itertools.product(*categ_modalities):
        for cont_combo in itertools.product(*cont_grids):
            x_vec = np.full(len(vars_metadata), np.nan)
            for i, var_idx in enumerate(categ_vars_indexes):
                x_vec[var_idx] = categ_combo[i]
            for i, var_idx in enumerate(cont_vars_indexes):
                x_vec[var_idx] = cont_combo[i]
            rows.append(x_vec)

    return np.array(rows)

def _build_qual_filter_with_mods_indexes(
    vars_metadata: VarsMetadata,
    qual_filter: List[Tuple[int, Optional[List]]],
) -> List[Tuple[int, List[int]]]:
    if qual_filter is None:
        qual_filter = [(i, None) for i in vars_metadata.get_categ_like_vars_indexes()]
    result = []
    for var_idx, mods in qual_filter:
        full_support = vars_metadata[var_idx].support
        modality_indexes = list(range(len(full_support))) if mods is None else [full_support.index(m) for m in mods]
        result.append((var_idx, modality_indexes))
    return result

def _build_combinations(
    qual_filter_with_mods_indexes : List[Tuple[int, List[int]]],
) -> List[Tuple[int, int]]:
    return list(itertools.product(*[mods for _, mods in qual_filter_with_mods_indexes]))

def _x_in_combo(
    x: np.ndarray,
    comb: Tuple[int, ...],
    qual_filter_with_mods_indexes: List[Tuple[int, List[int]]],
) -> bool:
    """Retourne True si x appartient exactement à la combo comb."""
    return all(
        int(x[var_idx]) == comb[k]
        for k, (var_idx, _) in enumerate(qual_filter_with_mods_indexes)
    )

def _mask_combo(
    x_data : np.ndarray,
    comb: Tuple[int, ...],
    qual_filter_with_mods_indexes: List[Tuple[int, List[int]]],
) -> np.ndarray:
    """Retourne le masque booléen des points de x_data appartenant à la combinaison de modalités."""
    mask = np.ones(len(x_data), dtype=bool)
    for k, (var_idx, _) in enumerate(qual_filter_with_mods_indexes):
        mask &= (x_data[:, var_idx].astype(int) == comb[k])
    return mask

def _prepare_gp_plot_data(
    metric: str,
    vars_metadata: VarsMetadata,
    quant_to_variate_idx_1: int,
    quant_to_variate_idx_2: int,
    quant_fixed_vals: List[Tuple[int, float]] = None, 
    qual_filter: List[Tuple[int, Optional[List]]]=None,
    fn_gp_objective: callable = None,
    z: Any = None,
    sm: Any = None,
    y_data: np.ndarray = None,
    reverse_optim: bool = True,
    n_grid: int=30,

) -> Tuple[
    np.ndarray, # G1
    np.ndarray, # G2
    List[Tuple[int, List[int]]], # qual_filter_with_mods_indexes
    List[Tuple], # combos
    List[np.ndarray], # z_pred_per_combo
]:

    quant_1 = vars_metadata[quant_to_variate_idx_1]
    quant_2 = vars_metadata[quant_to_variate_idx_2]
    grid_1 = np.linspace(quant_1.support[0], quant_1.support[1], n_grid)
    grid_2 = np.linspace(quant_2.support[0], quant_2.support[1], n_grid)
    G1, G2 = np.meshgrid(grid_1, grid_2, indexing="ij")
    n_pts = n_grid * n_grid
    sign = -1.0 if reverse_optim else 1.0
    f_max = float(y_data.ravel().max()) if y_data is not None else 0.0

    qual_filter_with_mods_indexes = _build_qual_filter_with_mods_indexes(vars_metadata, qual_filter)
    combinations = _build_combinations(qual_filter_with_mods_indexes)

    z_pred_per_combo : List[np.ndarray] = []

    for comb in combinations:
        x_ref = np.full(len(vars_metadata), np.nan)
        for k, (var_idx, _) in enumerate(qual_filter_with_mods_indexes):
            x_ref[var_idx] = comb[k]
        if quant_fixed_vals:
            for fix_idx, fix_val in quant_fixed_vals:
                x_ref[fix_idx] = fix_val

        x_pred = np.tile(x_ref, (n_pts, 1))
        x_pred[:, quant_to_variate_idx_1] = G1.ravel()
        x_pred[:, quant_to_variate_idx_2] = G2.ravel()

        if metric == "fn":
            z_val = fn_gp_objective(
                x = x_pred,
                vars_metadata = vars_metadata,
                z = z,
                sigma_noise = 0,
            ).ravel()
        else:
            if isinstance(sm, KrgBased):
                y_mu  = sign * sm.predict_values(x_pred).ravel()
                y_std = np.sqrt(np.maximum(sm.predict_variances(x_pred).ravel(), 0))
            elif isinstance(sm, MixedSingleTaskGP):
                from .xprs.xpr_botorch import XprBotorch
                y_mu, y_std = XprBotorch._predict_botorch(
                    model=sm,
                    x=x_pred
                )
            else:
                raise ValueError(
                    f"Unsupported surrogate type: {type(sm)}. Must be KrgBased (SMT), MixedSingleTaskGP (BoTorch)."
                )

            if metric == "mu":
                z_val = y_mu
            elif metric == "sd":
                z_val = y_std
            elif metric == "ei":
                u = np.where(y_std > 1e-12, (y_mu - f_max) / y_std, 0.0)
                z_val = np.maximum(np.where(
                    y_std > 1e-12,
                    (y_mu - f_max) * scipy_norm.cdf(u) + y_std * scipy_norm.pdf(u),
                    0.0,
                ), 0.0)
            else:
                raise ValueError("Metric must be 'mu', 'sd', 'ei' or 'fn'.")

        z_pred_per_combo.append(z_val)

    return G1, G2, qual_filter_with_mods_indexes, combinations, z_pred_per_combo