import numpy as np
from typing import Any, Optional
from dataclasses import dataclass

from smt.applications import EGO
from smt.applications.mixed_integer import MixedIntegerSamplingMethod
from smt.design_space import DesignSpace
from smt.sampling_methods.sampling_method import SamplingMethod
from smt.surrogate_models.krg_based import KrgBased
from smt.surrogate_models import KRG, MixIntKernelType

from smt.design_space import (
    DesignSpace,
    CategoricalVariable,
    FloatVariable,
) 

from .xpr import Xpr, XprResultRun
from ..context.vars_metadata import VarsMetadata
from ..utils import calc_convergence_iter_bo

@dataclass
class XprSmtHyperparams:
    seed: int
    seed_ydoe_offset: int
    n_init: int
    n_iter_bo: int
    sampling_method: str | SamplingMethod
    sampling_criterion: str
    kernel: MixIntKernelType
    bo_criterion: str
    qei: str
    n_start: int
    sigma_noise: float

    def to_dict(self):
        return {
            "seed": self.seed,
            "seed_ydoe_offset": self.seed_ydoe_offset,
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
class XprSmtResultRunEgo:
    x_opt: np.ndarray
    y_opt: float
    y_true_at_x_opt: float
    convergence_iter_bo: int
    x_data: np.ndarray
    y_data: np.ndarray
    y_true_at_x_data: np.ndarray
    sm: KrgBased

class XprSmt(Xpr):

    Hyperparams = XprSmtHyperparams

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
    ) -> XprSmtResultRunEgo:
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
        fn_y_objective_for_ego = lambda x: XprSmt.calc_y_objective_for_ego(
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

        return XprSmtResultRunEgo(
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
    def run(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        x_true_max: np.ndarray,
        y_true_max: float,
        hyperparams: XprSmtHyperparams,
        x_init: Optional[np.ndarray] = None,
        y_init: Optional[np.ndarray] = None,
    ) -> XprResultRun:

        design_space = XprSmt.create_design_space(
            vars_metadata=vars_metadata, 
            sampling_method=hyperparams.sampling_method,
            sampling_criterion=hyperparams.sampling_criterion,
            seed=hyperparams.seed
        )

        if x_init is None:
            x_init_run = XprSmt.calc_x_init_with_sampling_method(
                n_init=hyperparams.n_init,
                design_space=design_space,
                sampling_method=hyperparams.sampling_method,
                sampling_criterion=hyperparams.sampling_criterion,
                seed=hyperparams.seed,
            )
        else:
            # Pour faire avance la consommation du générateur de design_space.sampler
            # et que on ait résultat reproductible pour même seed sans x_init fourni.
            # Car dans EGO, _sample_valid_x des n_start points utilise le sampler en question.
            XprSmt.calc_x_init_with_sampling_method(
                            n_init=hyperparams.n_init,
                            design_space=design_space,
                            sampling_method=hyperparams.sampling_method,
                            sampling_criterion=hyperparams.sampling_criterion,
                            seed=hyperparams.seed,
                        )
            x_init_run = x_init
        
        if y_init is None:
            rng=np.random.default_rng(hyperparams.seed + hyperparams.seed_ydoe_offset)
            y_init_run = XprSmt.calc_y_objective_for_ego(
                fn_gp_objective=fn_gp_objective,
                vars_metadata=vars_metadata,
                x=x_init_run, 
                z=z,
                sigma_noise=hyperparams.sigma_noise,
                rng=rng
            )
        else:
            y_init_run = -y_init

        res_ego = XprSmt.run_ego(
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
            seed=hyperparams.seed,
        )

        return XprResultRun(
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