import warnings
import itertools
import numpy as np
from typing import Any, Tuple, Optional
from dataclasses import dataclass
 
from smt.design_space import (
    DesignSpace,
    CategoricalVariable,
    FloatVariable,
)
from smt.applications import EGO
from smt.applications.mixed_integer import MixedIntegerSamplingMethod
from smt.sampling_methods.sampling_method import SamplingMethod
from smt.surrogate_models.krg_based import KrgBased
from smt.surrogate_models import KRG, MixIntKernelType
 
from ..vars.vars_metadata import VarsMetadata

warnings.filterwarnings("ignore")

class XPR:
  
    @staticmethod
    def create_smt_design_space(
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
    def calc_x_init(
        n_init: int,
        design_space: DesignSpace,
        sampling_method: SamplingMethod,
        sampling_criterion: str,
        seed: int
    ) -> np.ndarray:
        sampler = MixedIntegerSamplingMethod(
            sampling_method_class=sampling_method,
            design_space=design_space,
            criterion=sampling_criterion,
            seed=seed,
        )
        x_init = sampler(n_init)
        return x_init

    @staticmethod
    def calc_y_objective_for_ego(
            fn_gp_objective: callable,
            vars_metadata: VarsMetadata, 
            x: np.ndarray, 
            z: Any, 
            sigma_noise: float,
            seed: int
        ) -> np.ndarray:
        raise NotImplementedError()
    
    @staticmethod
    def calc_true_max(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        n_grille: int,
    ) -> Tuple[np.ndarray, float]:
        raise NotImplementedError()

    
    @staticmethod
    def run_ego(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        design_space: DesignSpace,
        x_init: np.ndarray,
        y_init: np.ndarray,
        z: Any,
        n_iter_bo: int,
        kernel: MixIntKernelType,
        bo_criterion: str,
        qei: str,
        sigma_noise: float,
        n_start: int,
        rng: np.random.Generator,
    ) -> object:
        raise NotImplementedError()

    @staticmethod
    def run_xpr(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        x_true_max: np.ndarray,
        y_true_max: float,
        hyperparams: object,
        seed: int,
        x_init: Optional[np.ndarray] = None,
        y_init: Optional[np.ndarray] = None,
    ) -> object:
        raise NotImplementedError()


    # ------- méthodes d'affichage 

    @staticmethod
    def str_x_point(values: np.ndarray, vars_metada: VarsMetadata) -> str:
        parts = []
        for val, m in zip(values, vars_metada):
            if m.is_categ_like():
                label = m.support[int(val)]
                parts.append(f"{m.name} = {label}")
            else:
                parts.append(f"{m.name} = {val:.4f}")
        return " / ".join(parts)
    
    @staticmethod
    def print_data(x_init: np.ndarray, y_init: np.ndarray, vars_metadata: VarsMetadata, indent: int=2) -> None:
        for i in range(len(x_init)):
            x_str = XPR.str_x_point(values=x_init[i], vars_metada=vars_metadata)
            print(" "*indent + f"{x_str} -> Y = {y_init[i,0]:.4f}")
    
    @staticmethod
    def print_opt_result(x: np.ndarray, y: float, vars_metadata: VarsMetadata, indent: int=2) -> None:
        x_str = XPR.str_x_point(values=x, vars_metada=vars_metadata)
        print(" "*indent + f"x* = {x_str}")
        print(" "*indent + f"y* = {y:.4f}")


 