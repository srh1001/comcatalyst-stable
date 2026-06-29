import numpy as np
from typing import Any, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    convergence_iter_bo: int

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
            "convergence_iter_bo": self.convergence_iter_bo,

            "x_data": self.x_data.tolist(),
            "y_data": self.y_data.tolist(),
            "y_true_at_x_data": self.y_true_at_x_data.tolist(),
            
            "sm": "Not serializable",
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
    @abstractmethod
    def run(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        x_true_max: np.ndarray,
        y_true_max: float,
        hyperparams: object,
        x_init: Optional[np.ndarray] = None,
        y_init: Optional[np.ndarray] = None,
    ) -> object:
        raise NotImplementedError()


    # methodes d'affichage 

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
            x_str = Xpr.str_x_point(values=x_init[i], vars_metada=vars_metadata)
            print(" "*indent + f"{x_str} -> Y = {y_init[i,0]:.4f}")
    
    @staticmethod
    def print_opt_result(x: np.ndarray, y: float, vars_metadata: VarsMetadata, indent: int=2) -> None:
        x_str = Xpr.str_x_point(values=x, vars_metada=vars_metadata)
        print(" "*indent + f"x* = {x_str}")
        print(" "*indent + f"y* = {y:.4f}")