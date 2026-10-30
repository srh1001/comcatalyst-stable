import os
import numpy as np
from typing import Any, List, Optional
from dataclasses import dataclass

from smt.sampling_methods.sampling_method import SamplingMethod

from .xpr import Xpr, XprResultRun
from ..context.vars_metadata import VarsMetadata

from ..cocabo.methods.CoCaBO import CoCaBO

@dataclass
class XprCocaboHyperparams:
    seed: int
    seed_ydoe_offset: int
    n_init: int # utilisé seulement si sampling_method == "lhs" ou None
    n_iter_bo: int
    sampling_method: str | SamplingMethod
    sampling_criterion: str
    acq_type: str # "LCB" ou "EI"
    kernel_mix: float # poids du noyau produit vs somme, dans [0, 1]
    sigma_noise: float

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "seed_ydoe_offset": self.seed_ydoe_offset,
            "n_init": self.n_init,
            "n_iter_bo": self.n_iter_bo,
            "sampling_method": (
                self.sampling_method.__name__
                if isinstance(self.sampling_method, type) and issubclass(self.sampling_method, SamplingMethod)
                else self.sampling_method
            ),
            "sampling_criterion": self.sampling_criterion,
            "acq_type": self.acq_type,
            "kernel_mix": self.kernel_mix,
            "sigma_noise": self.sigma_noise,
        }

@dataclass
class XprCocaboResultRun:
    x_opt: np.ndarray
    y_opt: float
    y_true_at_x_opt: float
    idx_opt: int
    x_data: np.ndarray
    y_data: np.ndarray
    y_true_at_x_data: np.ndarray
    sm: None


class XprCocabo(Xpr):

    Hyperparams = XprCocaboHyperparams


    @staticmethod
    def _build_bounds(vars_metadata: VarsMetadata) -> List[dict]:
        """
        Convertit VarsMetadata au format des bornes attendues par CoCaBO.

        Parameters
        ----------
        vars_metadata : VarsMetadata

        Returns
        -------
        List[dict]
            Un dictionnaire par variable, dans l'ordre de vars_metadata.
        """
        bounds: List[dict] = []
        for v in vars_metadata:
            if v.is_categorical():
                bounds.append({
                    "name": v.name,
                    "type": "categorical",
                    "domain": tuple(range(len(v.support))),
                })
            elif v.is_continuous():
                bounds.append({
                    "name": v.name,
                    "type": "continuous",
                    "domain": (float(v.support[0]), float(v.support[1])),
                })
            else:
                raise ValueError(
                    f"Type non supporte pour CoCaBO : '{v.var_type}'"
                    f" (variable '{v.name}')"
                )
        return bounds

    @staticmethod
    def _build_categories(vars_metadata: VarsMetadata) -> List[int]:
        """
        Construit la liste C attendue par CoCaBO.

        Parameters
        ----------
        vars_metadata : VarsMetadata

        Returns
        -------
        List[int]
            Nombre de modalités par variable catégorielle.
            Exemple : [4, 4] pour deux variables à 4 modalités chacune.
        """
        return [len(v.support) for v in vars_metadata if v.is_categorical()]

    @staticmethod
    def _build_objfn(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        sigma_noise: float,
        seed: int,
    ) -> callable:
        """
        Adapte fn_gp_objective à la signature attendue par CoCaBO.

        CoCaBO appelle objfn(ht, x) avec :
            ht : List[int], valeurs categorielles, e.g. [2, 1]
            x : np.ndarray, valeurs continues, e.g. [0.3, -0.5, 0.1]

        La fonction retourne -y de forme (1, 1) car CoCaBO minimise.

        Parameters
        ----------
        fn_gp_objective : callable
        vars_metadata : VarsMetadata
        z : Any
        sigma_noise : float
        seed : int

        Returns
        -------
        callable
        """
        cat_indexes: List[int] = vars_metadata.get_categorical_vars_indexes()
        cont_indexes: List[int] = vars_metadata.get_continuous_vars_indexes()
        p: int = len(vars_metadata)
        rng: np.random.Generator = np.random.default_rng(seed)

        def objfn(ht: List[int], x: np.ndarray) -> np.ndarray:
            x_flat: np.ndarray = np.atleast_1d(x).flatten()
            x_full: np.ndarray = np.empty((1, p), dtype=float)
            for col_idx, cat_val in zip(cat_indexes, ht):
                x_full[0, col_idx] = float(cat_val)
            for col_idx, cont_val in zip(cont_indexes, x_flat):
                x_full[0, col_idx] = float(cont_val)
            y: np.ndarray = fn_gp_objective(
                x=x_full,
                vars_metadata=vars_metadata,
                z=z,
                sigma_noise=sigma_noise,
                rng=rng,
            )
            return -y.reshape(1, 1)

        return objfn

    @staticmethod
    def run_cocabo(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        n_iter_bo: int,
        acq_type: str,
        kernel_mix: float,
        sigma_noise: float,
        seed: int,
        x_init: Optional[np.ndarray],
        y_init: Optional[np.ndarray],
    ) -> "XprCocaboResultRun":
        """
        Lance CoCaBO.

        Si x_init est None, CoCaBO tire son propre plan initial.
        Si x_init est fourni, il est passe directement a runOptim via initData et initResult.

        Parameters
        ----------
        fn_gp_objective : callable
        vars_metadata : VarsMetadata
        z : Any
        n_iter_bo : int
        acq_type : str
        kernel_mix : float
        sigma_noise : float
        seed : int
        seed_ydoe_offset : int
        x_init : Optional[np.ndarray]
        y_init : Optional[np.ndarray]

        Returns
        -------
        XprCocaboResultRun
        """
        bounds: List[dict] = XprCocabo._build_bounds(vars_metadata)
        C: List[int] = XprCocabo._build_categories(vars_metadata)

        objfn: callable = XprCocabo._build_objfn(
            fn_gp_objective=fn_gp_objective,
            vars_metadata=vars_metadata,
            z=z,
            sigma_noise=sigma_noise,
            seed=seed,
        )

        n_init_cocabo: int = x_init.shape[0] if x_init is not None else 0

        mabbo: CoCaBO = CoCaBO(
            objfn=objfn,
            initN=n_init_cocabo,
            bounds=bounds,
            acq_type=acq_type,
            C=C,
            kernel_mix=kernel_mix,
        )

        # on lance qu'une seule optimisation
        mabbo.trial_num = 0

        saving_path: str = "results_cocabo.ignore/"
        os.makedirs(saving_path, exist_ok=True)
        mabbo.saving_path = saving_path

        if x_init is not None and y_init is not None:
            # On passe le plan initial directement à CoCaBO.
            # CoCaBO minimise, donc on passe -y_init.
            init_data: List[np.ndarray] = [x_init.copy()]
            init_result: List[np.ndarray] = [-y_init.copy()]
            mabbo.runOptim(
                budget=n_iter_bo,
                seed=seed,
                batch_size=1,
                initData=init_data,
                initResult=init_result,
            )
            
        else:
            # CoCaBO tire son propre plan initial via self.initialise(seed).
            mabbo.runOptim(
                budget=n_iter_bo,
                seed=seed,
                batch_size=1,
            )

        x_data: np.ndarray = mabbo.data[0]
        y_data: np.ndarray = -mabbo.result[0].ravel()

        best_idx: int = int(np.argmax(y_data))
        x_opt: np.ndarray = x_data[best_idx]
        y_opt: float = float(y_data[best_idx])

        y_true_at_x_opt: float = float(
            fn_gp_objective(
                x=x_opt.reshape(1, -1),
                vars_metadata=vars_metadata,
                z=z,
                sigma_noise=0,
            )[0, 0]
        )

        y_true_at_x_data: np.ndarray = fn_gp_objective(
            x=x_data,
            vars_metadata=vars_metadata,
            z=z,
            sigma_noise=0,
        )

        return XprCocaboResultRun(
            x_opt=x_opt,
            y_opt=y_opt,
            y_true_at_x_opt=y_true_at_x_opt,
            idx_opt=best_idx,
            x_data=x_data,
            y_data=y_data,
            y_true_at_x_data=y_true_at_x_data,
            sm=None,
        )

    @staticmethod
    def run(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        x_true_max: np.ndarray,
        y_true_max: float,
        hyperparams: XprCocaboHyperparams,
        x_init: Optional[np.ndarray] = None,
    ) -> XprResultRun:
        """
        Le plan initial est determiné par hyperparams.sampling_method, sauf si x_init est donné.

        Parameters
        ----------
        fn_gp_objective : callable
        vars_metadata : VarsMetadata
        z : Any
        x_true_max : np.ndarray, shape (p,)
        y_true_max : float
        hyperparams : XprCocaboHyperparams
        x_init : Optional[np.ndarray]

        Returns
        -------
        XprResultRun
        """
        if x_init is not None:
            x_init_run: np.ndarray = x_init

        elif hyperparams.sampling_method == "one_per_combo":
            x_init_run = Xpr.calc_x_init_one_per_combo(
                vars_metadata=vars_metadata,
                seed=hyperparams.seed,
            )
            hyperparams.n_init=x_init_run.shape[0]

        elif hyperparams.sampling_method is None:
            x_init_run = None

        else:
            x_init_run, _ = Xpr.calc_x_init_with_sampling_method(
                vars_metadata=vars_metadata,
                n_init=hyperparams.n_init,
                sampling_method=hyperparams.sampling_method,
                sampling_criterion=hyperparams.sampling_criterion,
                seed=hyperparams.seed,
            )

        if x_init_run is not None:
            rng: np.random.Generator = np.random.default_rng(
                hyperparams.seed + hyperparams.seed_ydoe_offset
            )
            y_init_run: Optional[np.ndarray] = fn_gp_objective(
                x=x_init_run,
                vars_metadata=vars_metadata,
                z=z,
                sigma_noise=hyperparams.sigma_noise,
                rng=rng,
            )
        else:
            y_init_run = None


        res = XprCocabo.run_cocabo(
            fn_gp_objective=fn_gp_objective,
            vars_metadata=vars_metadata,
            z=z,
            n_iter_bo=hyperparams.n_iter_bo,
            acq_type=hyperparams.acq_type,
            kernel_mix=hyperparams.kernel_mix,
            sigma_noise=hyperparams.sigma_noise,
            seed=hyperparams.seed,
            x_init=x_init_run,
            y_init=y_init_run,
        )

        return XprResultRun(
            hyperparams=hyperparams,
            x_true_max=x_true_max,
            y_true_max=y_true_max,
            x_opt=res.x_opt,
            y_opt=res.y_opt,
            y_true_at_x_opt=res.y_true_at_x_opt,
            idx_opt=res.idx_opt,
            x_data=res.x_data,
            y_data=res.y_data,
            y_true_at_x_data=res.y_true_at_x_data,
            sm=res.sm,
        )