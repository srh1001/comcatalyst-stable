import dataclasses
import itertools
import numpy as np
from typing import Any, List, Literal, Optional, Union, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor

from botorch.models import MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize, OutcomeTransform
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.optim import optimize_acqf_mixed
from gpytorch.kernels import PolynomialKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from smt.sampling_methods.sampling_method import SamplingMethod

from .xpr import Xpr, XprResultRun
from ..context.vars_metadata import VarsMetadata

AcqType = Literal["EI", "UCB", "KG"]
GpType  = Literal["rbf", "polynomial"]

class PolynomialMixedGP(MixedSingleTaskGP):

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: List[int],
        outcome_transform: Optional[OutcomeTransform] = None,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            outcome_transform=(
                Standardize(m=1) if outcome_transform is None else outcome_transform
            ),
            cont_kernel_factory=lambda batch_shape, ard_num_dims, active_dims: PolynomialKernel(
                power=2,
                active_dims=torch.tensor(active_dims),
            ),
        )

@dataclass
class XprBotorchHyperparams:
    seed: int
    seed_ydoe_offset: int
    n_init: int  # utilisé seulement si sampling_method est une SamplingMethod SMT
    n_iter_bo: int
    sampling_method: Union[str, type]   # SamplingMethod SMT, "one_per_combo", ou None
    sampling_criterion: str
    acq_type: AcqType    # "EI", "UCB", ou "KG"
    gp_type: GpType   # "matern", "rbf", ou "polynomial"
    num_fantasies: int   # utilise seulement si acq_type == "KG"
    beta: float   # utilise seulement si acq_type == "UCB"
    num_restarts: int  # nombre de relance pour optimize_acqf_mixed
    raw_samples: int    # nombre de points initiaux pour optimize_acqf_mixed
    sigma_noise: float
    use_bb: bool
    prune_every: int
    beta_ucb_pruning: float
    n_grid_cont: int
    allow_reintroduction: bool
    min_points_before_pruning: int

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "seed_ydoe_offset": self.seed_ydoe_offset,
            "n_init": self.n_init,
            "n_iter_bo": self.n_iter_bo,
            "sampling_method": (
                self.sampling_method.__name__
                if isinstance(self.sampling_method, type)
                and issubclass(self.sampling_method, SamplingMethod)
                else self.sampling_method
            ),
            "sampling_criterion": self.sampling_criterion,
            "acq_type": self.acq_type,
            "gp_type": self.gp_type,
            "num_fantasies": self.num_fantasies,
            "beta": self.beta,
            "num_restarts": self.num_restarts,
            "raw_samples": self.raw_samples,
            "sigma_noise": self.sigma_noise,
            "use_bb": self.use_bb,
            "prune_every": self.prune_every,
            "beta_ucb_pruning": self.beta_ucb_pruning,
            "n_grid_cont": self.n_grid_cont,
            "allow_reintroduction": self.allow_reintroduction,
            "min_points_before_pruning": self.min_points_before_pruning
        }


@dataclass
class XprBotorchResultRunEgo:
    x_opt: np.ndarray
    y_opt: float
    y_true_at_x_opt: float
    idx_opt: int
    x_data: np.ndarray
    y_data: np.ndarray 
    y_true_at_x_data: np.ndarray
    sm: MixedSingleTaskGP   


class XprBotorch(Xpr):

    Hyperparams = XprBotorchHyperparams

    @staticmethod
    def _to_tensor(x: np.ndarray) -> Tensor:
        return torch.tensor(x, dtype=torch.float64)

    @staticmethod
    def _to_numpy(t: Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    @staticmethod
    def _predict_botorch(
        model: MixedSingleTaskGP,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        model : MixedSingleTaskGP
        x : np.ndarray shape (n, p)

        Returns
        -------
        mu: np.ndarray shape (n,)
        std: np.ndarray shape (n,)
        """
        model.eval()
        X_torch = torch.tensor(x, dtype=torch.float64)
        with torch.no_grad():
            posterior = model.posterior(X_torch)
            mu  = posterior.mean.squeeze(-1).numpy()
            var = posterior.variance.squeeze(-1).numpy()
        return mu, np.sqrt(np.maximum(var, 0.0))

    @staticmethod
    def _build_fixed_features_list(
        vars_metadata: VarsMetadata,
    ) -> List[dict]:
        """
        Construit la liste de toutes les combinaisons de valeurs des variables
        catégorielles, au format attendu par optimize_acqf_mixed.

        Chaque élément est un dictionnaire {col_idx: valeur} pour une combinaison.

        Returns
        -------
        List[dict]
        """
        cat_indexes: List[int] = vars_metadata.get_categorical_vars_indexes()
        cat_vars = vars_metadata.get_categorical_vars_metadata()
        cat_ranges: List[range] = [range(len(v.support)) for v in cat_vars]

        return [
            {col_idx: float(val) for col_idx, val in zip(cat_indexes, combo)}
            for combo in itertools.product(*cat_ranges)
        ]

    @staticmethod
    def _build_bounds(vars_metadata: VarsMetadata) -> Tensor:
        """
        Construit le tenseur de bornes pour optimize_acqf_mixed.

        Les variables categorielles ont des bornes [0, n_modalités - 1].
        Les variables continues ont leurs bornes issues de vars_metadata.

        Returns
        -------
        Tensor de forme (2, p)
        """
        lowers: List[float] = []
        uppers: List[float] = []

        for v in vars_metadata:
            if v.is_categorical():
                lowers.append(0.0)
                uppers.append(float(len(v.support) - 1))
            elif v.is_continuous():
                lowers.append(float(v.support[0]))
                uppers.append(float(v.support[1]))
            else:
                raise ValueError(
                    f"Type non supporte pour BoTorch : '{v.var_type}'"
                    f" (variable '{v.name}')"
                )

        return torch.tensor([lowers, uppers], dtype=torch.float64)

    @staticmethod
    def _build_gp(
        X_torch: Tensor,
        Y_torch: Tensor,
        cat_dims: List[int],
        gp_type: GpType,
    ) -> MixedSingleTaskGP:
        """
        Construit et entraine un GP selon gp_type sur les donnees fournies.

        Parameters
        ----------
        X_torch: Tensor shape (n, p)
        Y_torch: Tensor shape (n, 1)
        cat_dims: List[int]
            Indices des colonnes categorielles dans X_torch.
        gp_type : GpType
            "rbf": noyau RBF sur les continues.
            "polynomial": noyau polynomial de degre 2 sur les continues.

        Returns
        -------
        MixedSingleTaskGP entrainé
        """
        if gp_type == "rbf":
            model: MixedSingleTaskGP = MixedSingleTaskGP(
                train_X=X_torch,
                train_Y=Y_torch,
                cat_dims=cat_dims,
                outcome_transform=Standardize(m=1),
            )

        elif gp_type == "polynomial":
            model = PolynomialMixedGP(
                train_X=X_torch,
                train_Y=Y_torch,
                cat_dims=cat_dims,
            )
        else:
            raise ValueError(
                f"gp_type unknown: '{gp_type}'. Accepted values: 'rbf', 'polynomial'."
            )

        mll: ExactMarginalLogLikelihood = ExactMarginalLogLikelihood(
            model.likelihood, model
        )
        fit_gpytorch_mll(mll)
        return model

    @staticmethod
    def _build_acqf(
        model: MixedSingleTaskGP,
        acq_type: AcqType,
        y_best: float,
        num_fantasies: int,
        beta: float,
    ) -> Any:
        """
        Instancie la fonction d'acquisition selon acq_type.

        Parameters
        ----------
        model: MixedSingleTaskGP
        acq_type: AcqType
        y_best: float
            Meilleur y observé jusqu'ici (utilisé par EI).
        num_fantasies: int
            Param pour KG.
        beta: float
            Param pour UCB.

        Returns
        -------
        Fonction d'acquisition BoTorch.
        """
        if acq_type == "EI":
            return qLogExpectedImprovement(
                model=model,
                best_f=torch.tensor(y_best, dtype=torch.float64),
            )
        elif acq_type == "UCB":
            return qUpperConfidenceBound(
                model=model,
                beta=beta,
            )
        elif acq_type == "KG":
            return qKnowledgeGradient(
                model=model,
                num_fantasies=num_fantasies,
            )
        else:
            raise ValueError(
                f"Unsupported acq_type: '{acq_type}'. Must be EI', 'UCB', 'KG'."
            )

    @staticmethod
    def train_sm_at_iter(
        result: XprResultRun,
        n_init: int,
        iter_bo: int,
        vars_metadata: VarsMetadata,
    ) -> Tuple[MixedSingleTaskGP, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Reconstruit le GP à partir des k = n_init + iter_bo premiers points,
        avec la même classe que result.sm (donc le même noyau).
        """
        x_data: np.ndarray = result.x_data
        y_data: np.ndarray = result.y_data.ravel()

        k: int = n_init + iter_bo
        x_data_k: np.ndarray = x_data[:k]
        y_data_k: np.ndarray = y_data[:k]

        cat_dims: List[int] = vars_metadata.get_categorical_vars_indexes()
        X_torch: Tensor = XprBotorch._to_tensor(x_data_k)
        Y_torch: Tensor = XprBotorch._to_tensor(y_data_k.reshape(-1, 1))

        model_k: MixedSingleTaskGP = result.sm.__class__(
            train_X=X_torch,
            train_Y=Y_torch,
            cat_dims=cat_dims,
            outcome_transform=Standardize(m=1),
        )
        mll: ExactMarginalLogLikelihood = ExactMarginalLogLikelihood(
            model_k.likelihood, model_k
        )
        fit_gpytorch_mll(mll)
        model_k.eval()

        best_idx_k: int = int(np.argmax(y_data_k))
        x_opt_k: np.ndarray = x_data_k[best_idx_k]
        y_opt_k: float = float(y_data_k[best_idx_k])

        return model_k, x_data_k, y_data_k, x_opt_k, y_opt_k

    @staticmethod
    def run_ego(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        x_init: np.ndarray,
        y_init: np.ndarray,
        n_iter_bo: int,
        acq_type: AcqType,
        gp_type: GpType,
        num_fantasies: int,
        beta: float,
        num_restarts: int,
        raw_samples: int,
        sigma_noise: float,
        seed: int,
    ) -> XprBotorchResultRunEgo:

        torch.manual_seed(seed)

        cat_dims = vars_metadata.get_categorical_vars_indexes()
        bounds: Tensor = XprBotorch._build_bounds(vars_metadata)
        fixed_features_list: List[dict] = XprBotorch._build_fixed_features_list(
            vars_metadata
        )

        X_all: np.ndarray = x_init.copy()
        Y_all: np.ndarray = y_init.copy()

        rng: np.random.Generator = np.random.default_rng(seed)

        for _ in range(n_iter_bo):
            X_torch: Tensor = XprBotorch._to_tensor(X_all)
            Y_torch: Tensor = XprBotorch._to_tensor(Y_all)

            model = XprBotorch._build_gp(X_torch, Y_torch, cat_dims, gp_type)

            y_best = float(Y_all.max())
            acqf = XprBotorch._build_acqf(
                model=model,
                acq_type=acq_type,
                y_best=y_best,
                num_fantasies=num_fantasies,
                beta=beta,
            )

            candidate, _ = optimize_acqf_mixed(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                fixed_features_list=fixed_features_list,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

            x_next: np.ndarray = XprBotorch._to_numpy(candidate.squeeze(0))

            y_next: np.ndarray = fn_gp_objective(
                x=x_next.reshape(1, -1), # pour passer de shape (p,) -> (1,p) car fonction objectif attend (n,p)
                vars_metadata=vars_metadata,
                z=z,
                sigma_noise=sigma_noise,
                rng=rng,
            )

            X_all = np.vstack([X_all, x_next.reshape(1, -1)])
            Y_all = np.vstack([Y_all, y_next])

        best_idx = int(np.argmax(Y_all))
        x_opt: np.ndarray = X_all[best_idx]
        y_opt = float(Y_all[best_idx, 0]) # ok car y 1D pour l'instant

        y_true_at_x_opt = float(
            fn_gp_objective(
                x=x_opt.reshape(1, -1),
                vars_metadata=vars_metadata,
                z=z,
                sigma_noise=0,
            )[0, 0]
        )
        y_true_at_x_data: np.ndarray = fn_gp_objective(
            x=X_all,
            vars_metadata=vars_metadata,
            z=z,
            sigma_noise=0,
        )

        return XprBotorchResultRunEgo(
            x_opt=x_opt,
            y_opt=y_opt,
            y_true_at_x_opt=y_true_at_x_opt,
            idx_opt=best_idx,
            x_data=X_all,
            y_data=Y_all,
            y_true_at_x_data=y_true_at_x_data,
            sm=model,
        )

    @staticmethod
    def _prune_qual_combos_botorch(
        model: MixedSingleTaskGP,
        vars_metadata: VarsMetadata,
        active_combos: List[Tuple[int, ...]],
        all_combos: List[Tuple[int, ...]],
        allow_reintroduction: bool,
        best_y_for_pruning: float,
        beta: float,
        n_grid_cont: int,
        x_data: np.ndarray,
        min_points_before_pruning: int = 0,
    ) -> List[Tuple[int, ...]]:
        cat_indexes= vars_metadata.get_categorical_vars_indexes()
        cont_indexes = vars_metadata.get_continuous_vars_indexes()

        # grille sur l'espace continu
        grids = [
            np.linspace(vars_metadata[i].support[0], vars_metadata[i].support[1], n_grid_cont)
            for i in cont_indexes
        ]
        x_cont: np.ndarray = np.array(list(itertools.product(*grids)))

        combos_to_evaluate = all_combos if allow_reintroduction else active_combos
        new_active_combos: List[Tuple[int, ...]] = []

        for combo in combos_to_evaluate:
            # points deja observés appartenant à cette combinaison
            mask_combo: np.ndarray = np.ones(len(x_data), dtype=bool)
            for k, idx in enumerate(cat_indexes):
                mask_combo &= (x_data[:, idx].astype(int) == combo[k])

            # vérifier min_points_before_pruning
            if min_points_before_pruning > 0 and int(mask_combo.sum()) < min_points_before_pruning:
                new_active_combos.append(combo)
                continue

            # grille continue pour cette combinaison
            x_batch: np.ndarray = np.zeros((len(x_cont), len(vars_metadata)))
            for k, idx in enumerate(cat_indexes):
                x_batch[:, idx] = float(combo[k])
            for k, idx in enumerate(cont_indexes):
                x_batch[:, idx] = x_cont[:, k]

            # ajout des points observés de cette combinaison à la grille
            x_batch = np.vstack([x_batch, x_data[mask_combo]])

            mu, std = XprBotorch._predict_botorch(model, x_batch)
            ucb = float((mu + beta * std).max())

            if ucb >= best_y_for_pruning :
                new_active_combos.append(combo)

        return new_active_combos


    @staticmethod
    def run_ego_with_bb(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        x_init: np.ndarray,
        y_init: np.ndarray,
        n_iter_bo: int,
        acq_type: AcqType,
        gp_type: GpType,
        num_fantasies: int,
        beta: float,
        num_restarts: int,
        raw_samples: int,
        sigma_noise: float,
        seed: int,
        prune_every: int,
        beta_ucb_pruning: float,
        n_grid_cont: int,
        allow_reintroduction: bool,
        min_points_before_pruning: int,
    ) -> XprBotorchResultRunEgo:
        """
        BO avec UCB pruning sur les combinaisons catégorielles.

        Parameters
        ----------
        beta_ucb_pruning : float
            Parametre beta de l'UCB utilisé pour le pruning (différent du beta de la fonction d'acquisition de la BO).
        """
        torch.manual_seed(seed)

        cat_dims = vars_metadata.get_categorical_vars_indexes()
        bounds: Tensor = XprBotorch._build_bounds(vars_metadata)

        # toutes les combinaisons possibles (entiers)
        cat_vars = vars_metadata.get_categorical_vars_metadata()
        all_combos: List[Tuple[int, ...]] = list(itertools.product(
            *[range(len(v.support)) for v in cat_vars]
        ))
        active_combos: List[Tuple[int, ...]] = list(all_combos)
        n_total_combos: int = len(all_combos)

        X_all: np.ndarray = x_init.copy()
        Y_all: np.ndarray = y_init.copy()
        
        rng: np.random.Generator = np.random.default_rng(seed)
        n_iter_remaining: int = n_iter_bo

        while n_iter_remaining > 0:
            n_iter_bloc = min(prune_every, n_iter_remaining)

            # fixed_features_list depuis active_combos
            fixed_features_list: List[dict] = [
                {cat_dims[k]: float(combo[k]) for k in range(len(cat_dims))}
                for combo in active_combos
            ]

            # boucle BO sur ce bloc
            for _ in range(n_iter_bloc):
                X_torch: Tensor = XprBotorch._to_tensor(X_all)
                Y_torch: Tensor = XprBotorch._to_tensor(Y_all)

                model = XprBotorch._build_gp(X_torch, Y_torch, cat_dims, gp_type)

                y_best = float(Y_all.max())
                acqf = XprBotorch._build_acqf(
                    model=model,
                    acq_type=acq_type,
                    y_best=y_best,
                    num_fantasies=num_fantasies,
                    beta=beta,
                )

                candidate, _ = optimize_acqf_mixed(
                    acq_function=acqf,
                    bounds=bounds,
                    q=1,
                    fixed_features_list=fixed_features_list,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )

                x_next: np.ndarray = XprBotorch._to_numpy(candidate.squeeze(0))

                y_next: np.ndarray = fn_gp_objective(
                    x=x_next.reshape(1, -1), 
                    vars_metadata=vars_metadata,
                    z=z,
                    sigma_noise=sigma_noise,
                    rng=rng,
                )

                X_all = np.vstack([X_all, x_next.reshape(1, -1)])
                Y_all = np.vstack([Y_all, y_next])

            n_iter_remaining -= n_iter_bloc

            if n_iter_remaining <= 0:
                break

            # pruning
            X_torch = XprBotorch._to_tensor(X_all)
            Y_torch = XprBotorch._to_tensor(Y_all)
            model_prune = XprBotorch._build_gp(X_torch, Y_torch, cat_dims, gp_type)

            best_idx = int(np.argmax(Y_all[:, 0]))
            x_opt: np.ndarray = X_all[best_idx]
            mu_opt, _ = XprBotorch._predict_botorch(
                model_prune, x_opt.reshape(1, -1) # shape (p,) vers (1,p)
            )
            best_y_for_pruning = float(mu_opt[0])

            new_active_combos = XprBotorch._prune_qual_combos_botorch(
                model=model_prune,
                vars_metadata=vars_metadata,
                active_combos=active_combos,
                all_combos=all_combos,
                allow_reintroduction=allow_reintroduction,
                best_y_for_pruning=best_y_for_pruning,
                beta=beta_ucb_pruning,
                n_grid_cont=n_grid_cont,
                x_data=X_all,
                min_points_before_pruning=min_points_before_pruning,
            )

            print(
                f"  [B&B] iter {n_iter_bo - n_iter_remaining}/{n_iter_bo}"
                f" | best_y_for_pruning={best_y_for_pruning:.3f}"
                f" | active combos={len(new_active_combos)}/{n_total_combos}"
                f" : {new_active_combos}"
            )

            active_combos = new_active_combos

        # résultats finaux
        best_idx = int(np.argmax(Y_all[:, 0]))
        x_opt = X_all[best_idx]
        y_opt = float(Y_all[best_idx, 0])

        y_true_at_x_opt = float(
            fn_gp_objective(
                x=x_opt.reshape(1, -1),
                vars_metadata=vars_metadata,
                z=z,
                sigma_noise=0,
            )[0, 0]
        )
        y_true_at_x_data: np.ndarray = fn_gp_objective(
            x=X_all,
            vars_metadata=vars_metadata,
            z=z,
            sigma_noise=0,
        )

        return XprBotorchResultRunEgo(
            x_opt=x_opt,
            y_opt=y_opt,
            y_true_at_x_opt=y_true_at_x_opt,
            idx_opt=best_idx,
            x_data=X_all,
            y_data=Y_all,
            y_true_at_x_data=y_true_at_x_data,
            sm=model,
        )

    @staticmethod
    def run(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        x_true_max: np.ndarray,
        y_true_max: float,
        hyperparams: XprBotorchHyperparams,
        x_init: Optional[np.ndarray] = None,
    ) -> XprResultRun:
        """
        Le plan initial est determiné par hyperparams.sampling_method, sauf si x_init est donné.

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
            hyperparams = dataclasses.replace(
                hyperparams, n_init=x_init_run.shape[0]
            )

        elif hyperparams.sampling_method is None:
            raise ValueError(
                "XprBotorch needs initial points, sampling_method cannot be None."
            )

        else:
            x_init_run, _ = Xpr.calc_x_init_with_sampling_method(
                vars_metadata=vars_metadata,
                n_init=hyperparams.n_init,
                sampling_method=hyperparams.sampling_method,
                sampling_criterion=hyperparams.sampling_criterion,
                seed=hyperparams.seed,
            )

        rng: np.random.Generator = np.random.default_rng(
            hyperparams.seed + hyperparams.seed_ydoe_offset
        )
        y_init_run: np.ndarray = fn_gp_objective(
            x=x_init_run,
            vars_metadata=vars_metadata,
            z=z,
            sigma_noise=hyperparams.sigma_noise,
            rng=rng,
        )

        if hyperparams.use_bb:
            res = XprBotorch.run_ego_with_bb(
                fn_gp_objective=fn_gp_objective,
                vars_metadata=vars_metadata,
                z=z,
                x_init=x_init_run,
                y_init=y_init_run,
                n_iter_bo=hyperparams.n_iter_bo,
                acq_type=hyperparams.acq_type,
                gp_type=hyperparams.gp_type,
                num_fantasies=hyperparams.num_fantasies,
                beta=hyperparams.beta,
                num_restarts=hyperparams.num_restarts,
                raw_samples=hyperparams.raw_samples,
                sigma_noise=hyperparams.sigma_noise,
                seed=hyperparams.seed,
                prune_every=hyperparams.prune_every,
                beta_ucb_pruning=hyperparams.beta_ucb_pruning,
                n_grid_cont=hyperparams.n_grid_cont,
                allow_reintroduction=hyperparams.allow_reintroduction,
                min_points_before_pruning=hyperparams.min_points_before_pruning
            )
        else:
            res = XprBotorch.run_ego(
                fn_gp_objective=fn_gp_objective,
                vars_metadata=vars_metadata,
                z=z,
                x_init=x_init_run,
                y_init=y_init_run,
                n_iter_bo=hyperparams.n_iter_bo,
                acq_type=hyperparams.acq_type,
                gp_type=hyperparams.gp_type,
                num_fantasies=hyperparams.num_fantasies,
                beta=hyperparams.beta,
                num_restarts=hyperparams.num_restarts,
                raw_samples=hyperparams.raw_samples,
                sigma_noise=hyperparams.sigma_noise,
                seed=hyperparams.seed

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