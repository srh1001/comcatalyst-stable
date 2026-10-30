import numpy as np
from typing import Any, Optional, List, Tuple, Dict
from dataclasses import dataclass
import itertools
import copy

from smt.applications import EGO
from smt.design_space import DesignSpace
from smt.sampling_methods.sampling_method import SamplingMethod
from smt.sampling_methods.lhs import LHS
from smt.surrogate_models.krg_based import KrgBased
from smt.surrogate_models import KRG, MixIntKernelType

from .xpr import Xpr, XprResultRun
from ..context.vars_metadata import VarsMetadata, VarMetadata

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
    sm: KrgBased
    use_bb: bool
    prune_every: int
    beta: float
    n_grid_cont: int
    allow_reintroduction: bool
    min_points_before_pruning: int

    def to_dict(self):

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
            "kernel": self.kernel.name,
            "bo_criterion": self.bo_criterion,
            "qei": self.qei,
            "n_start": self.n_start,
            "sigma_noise": self.sigma_noise,
            "sm": self.sm.__class__.__name__ if self.sm is not None else None,
            "use_bb": self.use_bb,
            "prune_every": self.prune_every,
            "beta": self.beta,
            "n_grid_cont": self.n_grid_cont,
            "allow_reintroduction": self.allow_reintroduction,
            "min_points_before_pruning": self.min_points_before_pruning
        }

@dataclass
class XprSmtResultRunEgo:
    x_opt: np.ndarray
    y_opt: float
    y_true_at_x_opt: float
    idx_opt: int
    x_data: np.ndarray
    y_data: np.ndarray
    y_true_at_x_data: np.ndarray
    sm: KrgBased

class XprSmt(Xpr):

    Hyperparams = XprSmtHyperparams

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
        rng_noise_y: np.random.Generator = None,
        sm: KrgBased=None
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
        
        if rng_noise_y is None:
            rng_noise_y=np.random.default_rng(seed)

        fn_y_objective_for_ego = lambda x: XprSmt.calc_y_objective_for_ego(
                fn_gp_objective=fn_gp_objective, 
                vars_metadata=vars_metadata, 
                x=x, 
                z=z, 
                sigma_noise=sigma_noise, 
                rng=rng_noise_y
            )

        if sm is None:
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
            ydoe=-y_init,
            qEI=qei,
            surrogate=sm,
            n_start=n_start,
            verbose=False,
            seed=seed
        )

        x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=fn_y_objective_for_ego)
        y_opt = -y_opt[0] # ok pour l'instant car une seule sortie

        y_true_at_x_opt = fn_gp_objective(x=x_opt, vars_metadata=vars_metadata, z=z, sigma_noise=0)
        y_true_at_x_data = fn_gp_objective(x=x_data, vars_metadata=vars_metadata, z=z, sigma_noise=0) 
        
        idx_opt = int(np.argmax(y_opt))

        return XprSmtResultRunEgo(
            x_opt=x_opt,
            y_opt=y_opt, 
            y_true_at_x_opt=y_true_at_x_opt[0, 0], # ok pour l'instant car une seule sortie y
            idx_opt=idx_opt,
            x_data=x_data,
            y_data=-y_data,
            y_true_at_x_data=y_true_at_x_data,
            sm=ego.gpr
        )

    @staticmethod
    def _reencoder_os_to_rs(
        x: np.ndarray,
        categ_idxs: List[int],
        vars_metadata_os: VarsMetadata,
        vars_metadata_rs: VarsMetadata,
    ) -> np.ndarray:
        """
        Convertit les indices catégoriels original_shape -> restricted_shape.
        """
        x = x.copy()
        for idx in categ_idxs:
            support_os = vars_metadata_os[idx].support
            support_rs = vars_metadata_rs[idx].support
            for i in range(x.shape[0]):
                level = support_os[int(x[i, idx])]
                x[i, idx] = support_rs.index(level)
        return x

    @staticmethod
    def _reencoder_rs_to_os(
        x: np.ndarray,
        categ_idxs: List[int],
        vars_metadata_os: VarsMetadata,
        vars_metadata_rs: VarsMetadata,
    ) -> np.ndarray:
        """
        Convertit les indices catégoriels restricted_shape -> original_shape.
        """
        x = x.copy()
        for idx in categ_idxs:
            support_rs = vars_metadata_rs[idx].support 
            support_os = vars_metadata_os[idx].support
            for i in range(x.shape[0]):
                level = support_rs[int(x[i, idx])]
                x[i, idx] = support_os.index(level)
        return x

    @staticmethod
    def _prune_qual_combos(
        sm: KrgBased,
        vars_metadata: VarsMetadata,
        active_combos: List[Tuple],
        allow_reintroduction: bool,
        all_combos: List[Tuple],
        best_y_for_pruning: float,
        beta: float,
        n_grid_cont: int,
        x_data: np.ndarray,
        min_points_before_pruning: int,
    ) -> List[Tuple]:
        """
        sm, vars_metadata, best_y_for_pruning et x_data doivent tous être de même shape (au sens support)
        
        Parameters
        -----------
        allow_reintroduction : si True, evaluer all_combos a chaque pruning
        min_points_before_pruning : nb minimum de points avant de pruner une combo
        """
        categ_idxs = [i for i, m in enumerate(vars_metadata) if m.is_categ_like()]
        cont_idxs = [i for i, m in enumerate(vars_metadata) if m.is_continuous()]

        grids  = [
            np.linspace(vars_metadata[i].support[0], vars_metadata[i].support[1], n_grid_cont)
            for i in cont_idxs
        ]
        x_cont = np.array(list(itertools.product(*grids)))

        combos_to_evaluate : List[Tuple] = all_combos if allow_reintroduction else active_combos
        actives_combos : List[Tuple] = []

        for combo in combos_to_evaluate:
            # points deja observes appartenant a cette combinaison
            mask_combo: np.ndarray = np.ones(len(x_data), dtype=bool)
            for k, idx in enumerate(categ_idxs):
                mask_combo &= (x_data[:, idx] == vars_metadata[idx].support.index(combo[k]))

            # verifier min_points_before_pruning
            if min_points_before_pruning > 0 and int(mask_combo.sum()) < min_points_before_pruning:
                actives_combos.append(combo)
                continue

            # grille continue pour cette combinaison
            x_batch: np.ndarray = np.zeros((len(x_cont), len(vars_metadata)))
            for k, idx in enumerate(categ_idxs):
                x_batch[:, idx] = vars_metadata[idx].support.index(combo[k])
            for k, idx in enumerate(cont_idxs):
                x_batch[:, idx] = x_cont[:, k]

            # ajout des points observes de cette combinaison a la grille
            x_batch = np.vstack([x_batch, x_data[mask_combo]])

            mu: np.ndarray = -sm.predict_values(x_batch).ravel()
            sigma: np.ndarray = np.sqrt(np.maximum(sm.predict_variances(x_batch).ravel(), 0))
            ucb: float = float((mu + beta * sigma).max())

            if ucb >= best_y_for_pruning :
                actives_combos.append(combo)

        # construire active_levels_per_var depuis les combos actives
        active_levels_per_var: List[List[str]] = [
            sorted({combo[k] for combo in actives_combos})
            for k in range(len(categ_idxs))
        ]

        # ajouter modalité complémentaire si une variable n'a plus qu'une modalité
        # (contrainte SMT : CategoricalVariable doit avoir >= 2 modalités)
        for k, idx in enumerate(categ_idxs):
            if len(active_levels_per_var[k]) == 1:
                additional_modality = next(
                    l for l in vars_metadata[idx].support
                    if l not in active_levels_per_var[k]
                )
                active_levels_per_var[k] = sorted(active_levels_per_var[k] + [additional_modality])
                print(f"  [B&B] Additional modality: {vars_metadata[idx].name}={additional_modality}")

        # retourner le produit cartesien des modalités actives
        actives_combos: List[Tuple] = list(itertools.product(*active_levels_per_var))

        return actives_combos

    @staticmethod
    def run_ego_with_bb(
        fn_gp_objective_os: callable,
        vars_metadata_os: VarsMetadata,
        z: Any,
        design_space_os: DesignSpace,
        x_init_os: np.ndarray,
        y_init_os: np.ndarray,
        n_iter_bo: int,
        kernel: MixIntKernelType,
        bo_criterion: str,
        qei: str,
        sigma_noise: float,
        n_start: int,
        seed: int,
        prune_every: int,
        beta: float,
        n_grid_cont: int,
        allow_reintroduction: bool,
        min_points_before_pruning: int,
        sm: KrgBased = None,
    ) -> XprSmtResultRunEgo:
        """
        BO avec UCB pruning sur les combinaisons qualitatives.

        Logique par bloc de prune_every iterations :
        1. EGO sur current_shape (os au premier bloc, rs ensuite)
        2. Reencoder x_new cs -> os, ajouter a x_data_os
        3. Pruning sur os -> active_combos
        4. Construire vars_metadata_rs, design_space_rs, x_data_rs, fn_objective_rs
        5. Retour a l'etape 1 avec current_shape = rs
        """
        categ_idxs = [i for i, m in enumerate(vars_metadata_os) if m.is_categ_like()]
        all_combos: List[Tuple] = list(itertools.product(*[
            vars_metadata_os[i].support for i in categ_idxs
        ]))
        n_total_combos = len(all_combos)

        if sm is None:
            sm = KRG(
                design_space=design_space_os,
                categorical_kernel=kernel,
                print_global=False,
                eval_noise=True,
                hyper_opt="Cobyla"
            )
        sm_original: KrgBased = sm

        x_data_os = x_init_os.copy()
        y_data_os = y_init_os.copy()

        # combos actives courantes
        active_combos: List[Tuple] = list(all_combos)

        n_iter_remaining = n_iter_bo
        rng_noise_y = np.random.default_rng(seed)

        # current_shape = os au premier bloc
        vars_metadata_cs = vars_metadata_os
        design_space_cs = design_space_os
        fn_objective_cs = fn_gp_objective_os
        sm_cs = sm_original
        x_data_cs = x_init_os.copy()
        y_data_cs = y_init_os.copy()

        res: XprSmtResultRunEgo = None

        while n_iter_remaining > 0:

            n_iter_bloc : int = min(prune_every, n_iter_remaining)

            # étape 1 : EGO sur current_shape
            res = XprSmt.run_ego(
                fn_gp_objective=fn_objective_cs,
                vars_metadata=vars_metadata_cs,
                design_space=design_space_cs,
                x_init=x_data_cs,
                y_init=y_data_cs,
                z=z,
                n_iter_bo=n_iter_bloc,
                kernel=kernel,
                bo_criterion=bo_criterion,
                qei=qei,
                sigma_noise=sigma_noise,
                n_start=n_start,
                seed=seed,
                rng_noise_y=rng_noise_y,
                sm=sm_cs,
            )

            # étape 2 : reencoder x_new cs -> os, ajouter à x_data_os
            n_before = len(x_data_cs)
            x_new_cs = res.x_data[n_before:]
            y_new = res.y_data[n_before:]

            if vars_metadata_cs is vars_metadata_os:
                x_new_os = x_new_cs # premier bloc donc de reencoding
            else:
                x_new_os = XprSmt._reencoder_rs_to_os(
                    x=x_new_cs,
                    categ_idxs=categ_idxs,
                    vars_metadata_os=vars_metadata_os,
                    vars_metadata_rs=vars_metadata_cs,
                )

            # mettre toutes les données ensemble
            x_data_os = np.vstack([x_data_os, x_new_os])
            y_data_os = np.vstack([y_data_os, y_new])

            n_iter_remaining -= n_iter_bloc

            if n_iter_remaining <= 0:
                break

            # étape 3: pruning sur original_shape
            best_idx_os = int(np.argmax(y_data_os.ravel()))
            x_opt_os = x_data_os[best_idx_os]

            sm_prune = sm_original.__class__(
                design_space=design_space_os,
                categorical_kernel=sm_original.options["categorical_kernel"],
                print_global=False,
                eval_noise=sm_original.options["eval_noise"],
                hyper_opt=sm_original.options["hyper_opt"],
            )
            sm_prune.set_training_values(x_data_os, -y_data_os)
            sm_prune.train()

            best_y_for_pruning = float(
                -sm_prune.predict_values(np.atleast_2d(x_opt_os))[0, 0]
            )

            new_active_combos: List[Tuple] = XprSmt._prune_qual_combos(
                sm=sm_prune,
                vars_metadata=vars_metadata_os,
                active_combos=active_combos,
                allow_reintroduction=allow_reintroduction,
                all_combos=all_combos,
                best_y_for_pruning=best_y_for_pruning,
                beta=beta,
                n_grid_cont=n_grid_cont,
                x_data=x_data_os,
                min_points_before_pruning=min_points_before_pruning,
            )

            print(f"  [B&B] iter {n_iter_bo - n_iter_remaining}/{n_iter_bo}"
                f" | best_y_for_pruning={best_y_for_pruning:.3f}"
                f" | active combos={len(new_active_combos)}/{n_total_combos}"
                f" : {['/'.join(c) for c in new_active_combos]}")

            # verifier qu'on a assez de points dans les nouvelles combos actives
            n_points_new = 0
            for combo in new_active_combos:
                mask = np.ones(len(x_data_os), dtype=bool)
                for k, idx in enumerate(categ_idxs):
                    mask &= (x_data_os[:, idx] == vars_metadata_os[idx].support.index(combo[k]))
                n_points_new += int(mask.sum())

            if n_points_new < 2:
                print(f"  [B&B] Pruning canceled: not enough points ({n_points_new})")
                # garder current_shape inchangé, juste mettre à jour x_data_cs et y_data_cs
                x_data_cs = res.x_data
                y_data_cs = res.y_data
                continue

            active_combos = new_active_combos

            # étape 4 : construire restricted_shape

            # active_levels_per_var depuis active_combos
            active_levels_per_var : List[List[str]] = [
                sorted({combo[k] for combo in active_combos})
                for k in range(len(categ_idxs))
            ]

            vars_metadata_rs = VarsMetadata([
                VarMetadata(
                    name=m.name,
                    var_type=m.var_type,
                    support=active_levels_per_var[categ_idxs.index(i)],
                ) if m.is_categ_like()
                else m
                for i, m in enumerate(vars_metadata_os)
            ])

            design_space_rs = Xpr.create_design_space(
                vars_metadata=vars_metadata_rs,
                sampling_method=LHS,
                sampling_criterion="ese",
                seed=seed,
            )

            # filtrer x_data_os sur active_combos et reencoder os -> rs
            mask_os = np.zeros(len(x_data_os), dtype=bool)
            for combo in active_combos:
                combo_mask = np.ones(len(x_data_os), dtype=bool)
                for k, idx in enumerate(categ_idxs):
                    combo_mask &= (
                        x_data_os[:, idx] == vars_metadata_os[idx].support.index(combo[k])
                    )
                mask_os |= combo_mask

            x_data_rs = XprSmt._reencoder_os_to_rs(
                x=x_data_os[mask_os],
                categ_idxs=categ_idxs,
                vars_metadata_os=vars_metadata_os,
                vars_metadata_rs=vars_metadata_rs,
            )
            y_data_rs = y_data_os[mask_os]

            # wrapper fn_objective pour reencoder x_rs -> x_os avant appel de fn_objective
            def make_fn_objective_rs(
                fn_os: callable,
                vars_metadata_os: VarsMetadata,
                vars_metadata_rs: VarsMetadata,
                categ_idxs: List[int],
            ) -> callable:
                def fn_rs(x, vars_metadata=None, z=None, sigma_noise=None, rng=None):
                    x_os=XprSmt._reencoder_rs_to_os(
                        x=np.atleast_2d(x),
                        categ_idxs=categ_idxs,
                        vars_metadata_os=vars_metadata_os,
                        vars_metadata_rs=vars_metadata_rs,
                    )
                    return fn_os(
                        x=x_os,
                        vars_metadata=vars_metadata_os,
                        z=z,
                        sigma_noise=sigma_noise,
                        rng=rng,
                    )
                return fn_rs

            fn_objective_rs = make_fn_objective_rs(
                fn_os=fn_gp_objective_os,
                vars_metadata_os=vars_metadata_os,
                vars_metadata_rs=vars_metadata_rs,
                categ_idxs=categ_idxs,
            )

            sm_rs = sm_original.__class__(
                design_space=design_space_rs,
                categorical_kernel=sm_original.options["categorical_kernel"],
                print_global=False,
                eval_noise=sm_original.options["eval_noise"],
                hyper_opt=sm_original.options["hyper_opt"],
            )

            # mettre à jour current_shape pour le prochain bloc
            vars_metadata_cs=vars_metadata_rs
            design_space_cs=design_space_rs
            fn_objective_cs=fn_objective_rs
            sm_cs=sm_rs
            x_data_cs=x_data_rs
            y_data_cs=y_data_rs

        # résultat final en original_shape
        y_true_at_x_data=fn_gp_objective_os(
            x=x_data_os,
            vars_metadata=vars_metadata_os,
            z=z,
            sigma_noise=0,
        )

        best_idx = int(np.argmax(y_data_os.ravel()))
        x_opt_os_final = x_data_os[best_idx]
        y_opt = float(y_data_os.ravel()[best_idx])
        y_true_at_x_opt = float(fn_gp_objective_os(
            x=np.atleast_2d(x_opt_os_final),
            vars_metadata=vars_metadata_os,
            z=z,
            sigma_noise=0,
        )[0, 0])

        return XprSmtResultRunEgo(
            x_opt=x_opt_os_final,
            y_opt=y_opt,
            y_true_at_x_opt=y_true_at_x_opt,
            idx_opt=best_idx,
            x_data=x_data_os,
            y_data=y_data_os,
            y_true_at_x_data=y_true_at_x_data,
            sm=res.sm,
        )

    @staticmethod
    def train_sm_at_iter(
        result_ego: XprSmtResultRunEgo,
        n_init: int,
        iter_bo: int,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]:

        x_data = result_ego.x_data
        y_data = result_ego.y_data.ravel()
        k = n_init + iter_bo

        x_data_k = x_data[:k]
        y_data_k = y_data[:k]

        krg_original = result_ego.sm._surrogate
        sm_k = krg_original.__class__(
            design_space = krg_original.design_space,
            categorical_kernel = krg_original.options["categorical_kernel"],
            print_global = False,
            eval_noise = krg_original.options["eval_noise"],
            hyper_opt = krg_original.options["hyper_opt"],
        )
        sm_k.set_training_values(x_data_k, -y_data_k.reshape(-1, 1))
        sm_k.train()

        best_idx_k = int(np.argmax(y_data_k))
        x_opt_k = x_data_k[best_idx_k]
        y_opt_k = float(y_data_k[best_idx_k])

        return sm_k, x_data_k, y_data_k, x_opt_k, y_opt_k

    @staticmethod
    def run(
        fn_gp_objective: callable,
        vars_metadata: VarsMetadata,
        z: Any,
        x_true_max: np.ndarray,
        y_true_max: float,
        hyperparams: XprSmtHyperparams,
        x_init: Optional[np.ndarray] = None,
    ) -> XprResultRun:

        if x_init is None:
            if hyperparams.sampling_method == "one_per_combo":
                x_init_run = Xpr.calc_x_init_one_per_combo(vars_metadata=vars_metadata, seed=hyperparams.seed+hyperparams.seed_ydoe_offset)
                hyperparams.n_init = x_init_run.shape[0]

                design_space = Xpr.create_design_space(
                    vars_metadata=vars_metadata,
                    sampling_method=LHS,
                    sampling_criterion="ese",
                    seed=hyperparams.seed,
                )

            else:
                x_init_run, design_space = Xpr.calc_x_init_with_sampling_method(
                    vars_metadata=vars_metadata,
                    n_init=hyperparams.n_init,
                    sampling_method=hyperparams.sampling_method,
                    sampling_criterion=hyperparams.sampling_criterion,
                    seed=hyperparams.seed,
                )
        else:
            x_init_run = x_init
            hyperparams.n_init = x_init_run.shape[0]

            # Pour faire avance la consommation du générateur de design_space.sampler
            # et que on ait le même résultat pour la même seed sans x_init fourni.
            # Car dans EGO, _sample_valid_x des n_start points utilise le sampler en question.
            _, design_space = Xpr.calc_x_init_with_sampling_method(
                vars_metadata=vars_metadata,
                n_init=hyperparams.n_init,
                sampling_method=LHS,
                sampling_criterion="ese",
                seed=hyperparams.seed,
            )


        rng=np.random.default_rng(hyperparams.seed + hyperparams.seed_ydoe_offset)
        y_init_run = fn_gp_objective(
            x=x_init_run,
            vars_metadata=vars_metadata,
            z=z,
            sigma_noise=hyperparams.sigma_noise,
            rng=rng,
        )

        if hyperparams.use_bb:
            res = XprSmt.run_ego_with_bb(
                fn_gp_objective_os=fn_gp_objective,
                vars_metadata_os=vars_metadata,
                z=z,
                design_space_os=design_space,
                x_init_os=x_init_run,
                y_init_os=y_init_run,
                n_iter_bo=hyperparams.n_iter_bo,
                kernel=hyperparams.kernel,
                bo_criterion=hyperparams.bo_criterion,
                qei=hyperparams.qei,
                sigma_noise=hyperparams.sigma_noise,
                n_start=hyperparams.n_start,
                seed=hyperparams.seed,
                prune_every=hyperparams.prune_every,
                beta=hyperparams.beta,
                n_grid_cont=hyperparams.n_grid_cont,
                allow_reintroduction=hyperparams.allow_reintroduction,
                min_points_before_pruning=hyperparams.min_points_before_pruning,
                sm=hyperparams.sm,
            )
        else:
            res = XprSmt.run_ego(
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
                sm=hyperparams.sm
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
            sm=res.sm
        )