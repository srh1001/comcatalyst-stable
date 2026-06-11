import warnings
import numpy as np
from typing import Any, Tuple, Optional
from dataclasses import dataclass
 
from smt.design_space import DesignSpace
from smt.sampling_methods.sampling_method import SamplingMethod

from smt.applications import EGO
from smt.surrogate_models.krg_based import KrgBased
from smt.surrogate_models import KRG, MixIntKernelType

from .xpr import XPR

from ..vars.vars_metadata import VarsMetadata
from ..utils import build_grid, calc_convergence_iter_bo

@dataclass
class XPRModAHyperparams:
    n_init: int
    n_iter_bo: int
    sampling_method: SamplingMethod
    sampling_criterion: str
    kernel: MixIntKernelType
    bo_criterion: str
    qei: str
    n_start: int
    sigma_noise: float

    def to_dict(self):
        return {
            "n_init": self.n_init,
            "n_iter_bo": self.n_iter_bo,
            "sampling_method": self.sampling_method.__name__,
            "sampling_criterion": self.sampling_criterion,
            "kernel": self.kernel.name,
            "bo_criterion": self.bo_criterion,
            "qei": self.qei,
            "n_start": self.n_start,
            "sigma_noise": self.sigma_noise,
        }

@dataclass
class XPRModAResultRunEgo:
    x_opt: np.ndarray
    y_opt: float
    y_true_at_x_opt: float
    convergence_iter_bo: int
    x_data: np.ndarray
    y_data: np.ndarray
    y_true_at_x_data: np.ndarray
    sm: KrgBased

@dataclass
class XPRModAResultRunXpr:
    seed: int
    seed_ydoe: int
    hyperparams: XPRModAHyperparams
 
    x_true_max: np.ndarray
    y_true_max: float
    x_opt: np.ndarray
    y_opt: float
    y_true_at_x_opt: float
    convergence_iter_bo: int

    x_data: np.ndarray
    y_data: np.ndarray
    y_true_at_x_data: np.ndarray

    sm: KrgBased
 
    def to_json_dict(self) -> dict:
        """Convertit en dict sérialisable json."""
        return {
            "seed": self.seed,
            "seed_ydoe": self.seed_ydoe,
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

class XPRModA(XPR):

    @staticmethod
    def calc_true_max(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        n_grille: int,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        x_true_max: np.ndarray, shape (p,)
        y_true_max: float
        X_grid: np.ndarray, shape (n_points, p)
        Y_grid: np.ndarray, shape (n_points,)
        """
        X_grid = build_grid(vars_metadata, n_grille) # shape (n_points, p)

        Y_grid = fn_gp_objective(x=X_grid, vars_metadata=vars_metadata, z=z, sigma_noise=0).ravel() # shape (n_points,)

        best_idx=np.argmax(Y_grid)
        x_true_max=X_grid[best_idx]
        y_true_max=float(Y_grid[best_idx])

        return x_true_max, y_true_max, X_grid, Y_grid

    @staticmethod
    def calc_y_objective_for_ego(
            fn_gp_objective: callable,
            vars_metadata: VarsMetadata, 
            x: np.ndarray, 
            z: Any, 
            sigma_noise: float,
            rng: np.random.Generator
        ) -> np.ndarray:
        y = fn_gp_objective(x=x, vars_metadata=vars_metadata, z=z, sigma_noise=sigma_noise, rng=rng)
        return -y

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
        seed: int,
    ) -> XPRModAResultRunEgo:
        """ 
        Parameters
        ----------
        kernel: MixIntKernelType
            Noyau pour les variables catégorielles.
        bo_criterion: Fonction d'acquisition gérée par SMT. "EI", "SBO" or "LCB".
        seed: int
        qei: str
            qEI méthode gérée par SMT. "KBUB", "KB", "KBLB", "KBUB", "KBRand".
        """
        rng=np.random.default_rng(seed)

        fn_y_objective_for_ego = lambda x: XPRModA.calc_y_objective_for_ego(
                fn_gp_objective=fn_gp_objective, 
                vars_metadata=vars_metadata, 
                x=x, 
                z=z, 
                sigma_noise=sigma_noise, 
                rng=rng
            )

        sm = KRG(
            design_space=design_space,
            categorical_kernel=kernel,
            print_global=False,
            eval_noise=True,
            hyper_opt="Cobyla"
        )
 
        ego = EGO(
            n_iter=n_iter_bo,
            criterion=bo_criterion,
            xdoe=x_init,
            ydoe=y_init,
            qEI=qei,
            surrogate=sm,
            n_start=n_start,
            verbose=False,
            seed=seed
        )

        x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=fn_y_objective_for_ego)
        
        y_true_at_x_opt = fn_gp_objective(x=x_opt, vars_metadata=vars_metadata, z=z, sigma_noise=0)
        y_true_at_x_data = fn_gp_objective(x=x_data, vars_metadata=vars_metadata, z=z, sigma_noise=0) 
        
        convergence_iter_bo = calc_convergence_iter_bo(y_data=-y_data, n_init=len(y_init))

        return XPRModAResultRunEgo(
            x_opt=x_opt,
            y_opt=-y_opt[0], # ok pour l'instant car une seule sortie y sinon à adapter et dans les fonctions de plots, etc. aussi...
            y_true_at_x_opt=y_true_at_x_opt[0, 0], # ok pour l'instant car une seule sortie y sinon à adapter et dans les fonctions de plots, etc. aussi pour rendre compatible...
            convergence_iter_bo=convergence_iter_bo,
            x_data=x_data,
            y_data=-y_data,
            y_true_at_x_data=y_true_at_x_data,
            sm=sm
        )


    @staticmethod
    def run_xpr(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        x_true_max: np.ndarray,
        y_true_max: float,
        hyperparams: XPRModAHyperparams,
        seed: int,
        seed_ydoe: int,
        x_init: Optional[np.ndarray] = None,
        y_init: Optional[np.ndarray] = None,
    ) -> XPRModAResultRunXpr:

        design_space = XPRModA.create_smt_design_space(
            vars_metadata=vars_metadata, 
            sampling_method=hyperparams.sampling_method,
            sampling_criterion=hyperparams.sampling_criterion,
            seed=seed
        )

        if x_init is None:
            x_init_run = XPRModA.calc_x_init(
                n_init=hyperparams.n_init,
                design_space=design_space,
                sampling_method=hyperparams.sampling_method,
                sampling_criterion=hyperparams.sampling_criterion,
                seed=seed,
            )
        else:
            x_init_run = x_init
        
        if y_init is None:
            rng=np.random.default_rng(seed_ydoe)
            y_init_run = XPRModA.calc_y_objective_for_ego(
                fn_gp_objective=fn_gp_objective,
                vars_metadata=vars_metadata,
                x=x_init_run, 
                z=z,
                sigma_noise=hyperparams.sigma_noise,
                rng=rng
            )
        else:
            y_init_run = -y_init

        res_ego = XPRModA.run_ego(
            fn_gp_objective=fn_gp_objective,
            vars_metadata=vars_metadata,
            design_space=design_space,
            x_init=x_init_run,
            y_init=y_init_run,
            z=z,
            n_iter_bo=hyperparams.n_iter_bo,
            kernel=hyperparams.kernel,
            bo_criterion=hyperparams.bo_criterion,
            qei=hyperparams.qei,
            sigma_noise=hyperparams.sigma_noise,
            n_start=hyperparams.n_start,
            seed=seed,
        )

        return XPRModAResultRunXpr(
            seed=seed,
            seed_ydoe=seed_ydoe,
            hyperparams=hyperparams,
            x_true_max=x_true_max,
            y_true_max=y_true_max,
            x_opt=res_ego.x_opt,
            y_opt=res_ego.y_opt,
            y_true_at_x_opt=res_ego.y_true_at_x_opt,
            convergence_iter_bo=res_ego.convergence_iter_bo,
            x_data=res_ego.x_data,
            y_data=res_ego.y_data,
            y_true_at_x_data=res_ego.y_true_at_x_data,
            sm=res_ego.sm,
        )