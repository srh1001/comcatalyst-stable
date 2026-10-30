import numpy as np
from typing import Any, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools

from smt.applications.mixed_integer import MixedIntegerSamplingMethod
from smt.sampling_methods.sampling_method import SamplingMethod
from smt.design_space import (
    DesignSpace,
    CategoricalVariable,
    FloatVariable,
) 

from ..context.vars_metadata import VarsMetadata
from ..utils import build_grid

@dataclass
class XprResultRun:
    hyperparams: Any
 
    x_true_max: np.ndarray
    y_true_max: float
    x_opt: np.ndarray
    y_opt: float
    y_true_at_x_opt: float
    idx_opt: int

    x_data: np.ndarray
    y_data: np.ndarray
    y_true_at_x_data: np.ndarray

    sm: Any
 
    def to_json_dict(self) -> dict:
        return {
            "hyperparams": self.hyperparams.to_dict(),

            "x_true_max": self.x_true_max.tolist(),
            "y_true_max": self.y_true_max,
            "x_opt": self.x_opt.tolist(),
            "y_opt": self.y_opt,
            "y_true_at_x_opt":  self.y_true_at_x_opt,
            "idx_opt": self.idx_opt,

            "x_data": self.x_data.tolist(),
            "y_data": self.y_data.tolist(),
            "y_true_at_x_data": self.y_true_at_x_data.tolist(),            
        }

class Xpr(ABC):
    Hyperparams: type
  
    @staticmethod
    def calc_true_max(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        n_grid: int,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        x_true_max: np.ndarray, shape (p,)
        y_true_max: float
        X_grid: np.ndarray, shape (n_points, p)
        Y_grid: np.ndarray, shape (n_points,)
        """
        X_grid = build_grid(vars_metadata, n_grid) # shape (n_points, p)

        Y_grid = fn_gp_objective(x=X_grid, vars_metadata=vars_metadata, z=z, sigma_noise=0).ravel() # shape (n_points,)

        best_idx=np.argmax(Y_grid)
        x_true_max=X_grid[best_idx]
        y_true_max=float(Y_grid[best_idx])

        return x_true_max, y_true_max, X_grid, Y_grid
    
    @staticmethod
    def calc_x_init_one_per_combo(
        vars_metadata: VarsMetadata,
        seed: int,
    ) -> np.ndarray:

        rng = np.random.default_rng(seed)
 
        cat_vars = vars_metadata.get_categorical_vars_metadata()
        cont_vars = vars_metadata.get_continuous_vars_metadata()
 
        cat_indexes = vars_metadata.get_categorical_vars_indexes()
        cont_indexes = vars_metadata.get_continuous_vars_indexes()
 
        p = len(vars_metadata)
 
        # Toutes les combinaisons de modalités des variables categorielles
        # Chaque modalité est representée par son indice entier dans le support
        cat_ranges = [range(len(v.support)) for v in cat_vars]
        combos = list(itertools.product(*cat_ranges))
        n_combos = len(combos)
 
        x_init: np.ndarray = np.empty((n_combos, p), dtype=float)
 
        for row_idx, combo in enumerate(combos):
            # Variables categorielles : indice entier de la modalite
            for col_idx, cat_idx in zip(cat_indexes, combo):
                x_init[row_idx, col_idx] = float(cat_idx)
 
            # Variables continues : tirage uniforme dans le domaine de support
            for col_idx, cont_var in zip(cont_indexes, cont_vars):
                lo: float = float(cont_var.support[0])
                hi: float = float(cont_var.support[1])
                x_init[row_idx, col_idx] = rng.uniform(lo, hi)
 
        return x_init
    
    @staticmethod
    def create_design_space(
        vars_metadata: VarsMetadata, 
        sampling_method: SamplingMethod,
        sampling_criterion: str,
        seed: int
    ) -> DesignSpace:
        ds_vars = []
        for v in vars_metadata:
            if v.is_categorical():
                ds_vars.append(CategoricalVariable(values=v.support))
            elif v.is_continuous():
                ds_vars.append(FloatVariable(v.support[0], v.support[1]))
            else:
                raise ValueError(
                    f"Not supported variable type for variable '{v.name}': {v.var_type}"
                )
        ds = DesignSpace(ds_vars, seed=seed)        
        x_limits = ds.get_unfolded_num_bounds()
        ds.sampler = sampling_method(xlimits=x_limits, seed=seed, criterion=sampling_criterion)
        return ds
    
    @staticmethod
    def calc_x_init_with_sampling_method(
        vars_metadata: VarsMetadata,
        n_init: int,
        sampling_method: SamplingMethod,
        sampling_criterion: str,
        seed: int
    ) -> Tuple[np.ndarray, DesignSpace]:
        
        design_space = Xpr.create_design_space(
            vars_metadata=vars_metadata,
            sampling_method=sampling_method,
            sampling_criterion=sampling_criterion,
            seed=seed,
        )    

        sampler = MixedIntegerSamplingMethod(
            sampling_method_class=sampling_method,
            design_space=design_space,
            criterion=sampling_criterion,
            seed=seed,
        )
        x_init = sampler(n_init)
        return x_init, design_space
    
    @staticmethod
    @abstractmethod
    def run(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        x_true_max: np.ndarray,
        y_true_max: float,
        hyperparams: object,
        x_init: Optional[np.ndarray] = None,
    ) -> object:
        raise NotImplementedError()