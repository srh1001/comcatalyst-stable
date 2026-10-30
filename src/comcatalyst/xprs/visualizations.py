import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import tempfile
from PIL import Image
import itertools
from typing import List, Tuple, Any, Optional
from smt.surrogate_models.krg_based import KrgBased

from ..context.vars_metadata import VarsMetadata
from ..utils import _x_in_combo, _mask_combo, _prepare_gp_plot_data
from ..xprs.xpr_smt import XprSmtResultRunEgo, XprSmt

def plot_contour_quant_by_qual(
    metric: str,
    vars_metadata: VarsMetadata,
    quant_to_variate_idx_1: int,
    quant_to_variate_idx_2: int,
    quant_fixed_vals: List[Tuple[int, float]] = None,
    qual_filter: List[Tuple[int, Optional[List]]] = None,
    fn_gp_objective: callable = None,
    z: Any = None,
    sm: KrgBased = None,
    reverse_optim: bool = None,
    x_data: np.ndarray = None,
    y_data: np.ndarray = None,
    n_init: int = None,
    x_opt: np.ndarray = None,
    y_opt: float = None,
    x_true_max: np.ndarray = None,
    y_true_max: float = None,
    n_grid: int = 30,
    title: str = "GP contour",
    figsize: Tuple[int, int] = (6, 5),
    n_cols: int = 3,
    fontsize: int = 10,
    save_path: str = None,
) -> None:
    G1, G2, qual_filter_with_mods_indexes, combinations, z_pred_per_combo = _prepare_gp_plot_data(
        metric=metric,
        vars_metadata=vars_metadata,
        quant_to_variate_idx_1=quant_to_variate_idx_1,
        quant_to_variate_idx_2=quant_to_variate_idx_2,
        quant_fixed_vals=quant_fixed_vals,
        qual_filter=qual_filter,
        fn_gp_objective=fn_gp_objective,
        z=z,
        sm=sm,
        reverse_optim=reverse_optim,
        y_data=y_data,
        n_grid=n_grid,
    )

    z_global_min = min(z.min() for z in z_pred_per_combo)
    z_global_max = max(z.max() for z in z_pred_per_combo)
    levels = np.linspace(z_global_min, z_global_max, 20)

    cmap = "Reds" if metric == "ei" else "viridis"

    quant_1 = vars_metadata[quant_to_variate_idx_1]
    quant_2 = vars_metadata[quant_to_variate_idx_2]
    n_combs = len(combinations)
    n_rows = (n_combs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows))
    axes_flat = np.array(axes).ravel() if n_combs > 1 else [axes]
    fig.suptitle(title, fontsize=fontsize)

    for i, (comb, z) in enumerate(zip(combinations, z_pred_per_combo)):
        ax = axes_flat[i]
        Z = z.reshape(n_grid, n_grid)
        cf = ax.contourf(G1, G2, Z, levels=levels, cmap=cmap)
        plt.colorbar(cf, ax=ax, label=metric)
        ax.contour(G1, G2, Z, levels=levels, colors="white", linewidths=0.3, alpha=0.4)

        if x_data is not None and n_init is not None:
            for global_idx in np.where(_mask_combo(x_data, comb, qual_filter_with_mods_indexes))[0]:
                if metric != "ei":
                    color = "white" if global_idx < n_init else "red"
                else:
                    color = "black" if global_idx < n_init else "green"

                ax.annotate(str(global_idx),
                    xy=(x_data[global_idx, quant_to_variate_idx_1], x_data[global_idx, quant_to_variate_idx_2]),
                    xytext=(0, 0), textcoords="offset points",
                    fontsize=fontsize*1.3, color=color, zorder=6, ha="center", va="center",
                )

        if x_opt is not None and y_opt is not None and _x_in_combo(x_opt, comb, qual_filter_with_mods_indexes):
            ax.scatter(x_opt[quant_to_variate_idx_1], x_opt[quant_to_variate_idx_2],
                       color="black", s=100, marker="*", zorder=10, label=f"BO opt (Y={y_opt:.3f})")

        if x_true_max is not None and y_true_max is not None and _x_in_combo(x_true_max, comb, qual_filter_with_mods_indexes):
            ax.scatter(x_true_max[quant_to_variate_idx_1], x_true_max[quant_to_variate_idx_2],
                       color="red", s=100, marker="*", zorder=10, label=f"True max (Y={y_true_max:.3f})")

        title_parts = [f"{vars_metadata[var_idx].name}={vars_metadata[var_idx].support[comb[k]]}"
                       for k, (var_idx, _) in enumerate(qual_filter_with_mods_indexes)]
        ax.set_xlabel(quant_1.name, fontsize=fontsize)
        ax.set_ylabel(quant_2.name, fontsize=fontsize)
        ax.set_title(", ".join(title_parts), fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize - 2)

    for j in range(n_combs, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_3d_quant_by_qual(
    metric: str,
    vars_metadata: VarsMetadata,
    quant_to_variate_idx_1: int,
    quant_to_variate_idx_2: int,
    quant_fixed_vals: List[Tuple[int, float]] = None,
    qual_filter: List[Tuple[int, Optional[List]]] = None,
    fn_gp_objective: callable = None,
    z: Any = None,
    sm: KrgBased = None,
    reverse_optim: bool = None,
    x_data: np.ndarray = None,
    y_data: np.ndarray = None,
    n_init: int = None,
    x_opt: np.ndarray = None,
    y_opt: float = None,
    x_true_max: np.ndarray = None,
    y_true_max: float = None,
    n_grid: int = 30,
    title: str = "GP surface",
    figsize: Tuple[int, int] = (6, 5),
    n_cols: int = 3,
    fontsize: int = 10,
    save_path: str = None,
) -> None:
    G1, G2, qual_filter_with_mods_indexes, combinations, z_pred_per_combo = _prepare_gp_plot_data(
        metric=metric,
        vars_metadata=vars_metadata,
        quant_to_variate_idx_1=quant_to_variate_idx_1,
        quant_to_variate_idx_2=quant_to_variate_idx_2,
        quant_fixed_vals=quant_fixed_vals,
        qual_filter=qual_filter,
        fn_gp_objective=fn_gp_objective,
        z=z,
        sm=sm,
        reverse_optim=reverse_optim,
        y_data=y_data,
        n_grid=n_grid,
    )

    z_global_min = min(z.min() for z in z_pred_per_combo)
    z_global_max = max(z.max() for z in z_pred_per_combo)

    cmap = "Reds" if metric == "ei" else "viridis"

    quant_1 = vars_metadata[quant_to_variate_idx_1]
    quant_2 = vars_metadata[quant_to_variate_idx_2]
    n_combs = len(combinations)
    n_rows = (n_combs + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(figsize[0]*n_cols, figsize[1]*n_rows))
    fig.suptitle(title, fontsize=fontsize)

    for i, (comb, z) in enumerate(zip(combinations, z_pred_per_combo)):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection="3d", computed_zorder=False)
        Z = z.reshape(n_grid, n_grid)
        ax.plot_surface(G1, G2, Z, cmap=cmap, alpha=0.8, zorder=1, vmin=z_global_min, vmax=z_global_max)
        ax.set_zlim(z_global_min, z_global_max)

        if x_data is not None and y_data is not None and n_init is not None:
            for global_idx in np.where(_mask_combo(x_data, comb, qual_filter_with_mods_indexes))[0]:
                if metric != "ei":
                    color = "black" if global_idx < n_init else "red"
                else:
                    color = "black" if global_idx < n_init else "green"

                ax.text(
                    x_data[global_idx, quant_to_variate_idx_1],
                    x_data[global_idx, quant_to_variate_idx_2],
                    y_data.ravel()[global_idx],
                    str(global_idx),
                    fontsize=fontsize*1.3, color=color, zorder=4, ha="center", va="center",
                )
        if x_opt is not None and y_opt is not None and _x_in_combo(x_opt, comb, qual_filter_with_mods_indexes):
            ax.scatter(x_opt[quant_to_variate_idx_1], x_opt[quant_to_variate_idx_2], y_opt,
                       color="black", s=figsize[0]*15, marker="*", zorder=4, label=f"BO opt (Y={y_opt:.3f})")

        if x_true_max is not None and y_true_max is not None and _x_in_combo(x_true_max, comb, qual_filter_with_mods_indexes):
            ax.scatter(x_true_max[quant_to_variate_idx_1], x_true_max[quant_to_variate_idx_2], y_true_max,
                       color="red", s=figsize[0]*15, marker="*", zorder=4, label=f"True max (Y={y_true_max:.3f})")

        title_parts = [f"{vars_metadata[var_idx].name}={vars_metadata[var_idx].support[comb[k]]}"
                       for k, (var_idx, _) in enumerate(qual_filter_with_mods_indexes)]
        ax.set_title(", ".join(title_parts), fontsize=fontsize)
        ax.set_xlabel(quant_1.name, fontsize=fontsize)
        ax.set_ylabel(quant_2.name, fontsize=fontsize)
        ax.set_zlabel(metric, fontsize=fontsize, labelpad=-5)
        ax.tick_params(axis="x", labelsize=fontsize - 2)
        ax.tick_params(axis="y", labelsize=fontsize - 2)
        ax.tick_params(axis="z", labelsize=fontsize - 2)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_gp_at_iter(
    plot_type: str,
    metric: str,
    result_ego: XprSmtResultRunEgo,
    vars_metadata: VarsMetadata,
    n_init: int,
    iter_bo: int,
    quant_to_variate_idx_1: int,
    quant_to_variate_idx_2: int,
    quant_fixed_vals: List[Tuple[int, float]] = None,
    qual_filter: List[Tuple[int, Optional[List]]] = None,
    n_grid: int = 30,
    n_cols: int = 3,
    figsize: Tuple[int, int] = (5, 4),
    title: str = None,
    save_path: str = None,
) -> None:

    if plot_type == 'contour':
        fn_plot = plot_contour_quant_by_qual
    elif plot_type == '3d':
        fn_plot = plot_3d_quant_by_qual
    else:
        raise ValueError("Unsupported plot_type. Must be 'contour' or '3d'.")

    from botorch.models import MixedSingleTaskGP
    from .xpr_botorch import XprBotorch
    if isinstance(result_ego.sm, MixedSingleTaskGP):
        sm_k, x_data_k, y_data_k, x_opt_k, y_opt_k = XprBotorch.train_sm_at_iter(
            result_ego, n_init, iter_bo, vars_metadata
        )
    else:
        sm_k, x_data_k, y_data_k, x_opt_k, y_opt_k = XprSmt.train_sm_at_iter(
            result_ego, n_init, iter_bo
        )

    fn_plot(
        metric=metric,
        vars_metadata=vars_metadata,
        quant_to_variate_idx_1=quant_to_variate_idx_1,
        quant_to_variate_idx_2=quant_to_variate_idx_2,
        quant_fixed_vals=quant_fixed_vals,
        qual_filter=qual_filter,
        sm=sm_k,
        reverse_optim=True,
        x_data=x_data_k,
        y_data=y_data_k.reshape(-1, 1),
        n_init=n_init,
        x_opt=x_opt_k,
        y_opt=y_opt_k,
        n_grid=n_grid,
        n_cols=n_cols,
        figsize=figsize,
        title=title or f"GP contour {metric} | iter_bo={iter_bo}",
        save_path=save_path,
    )

def resize_frame(frame: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(frame)
    img = img.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
    return np.array(img)

def animate_gp_at_iters(
    plot_type: str,
    metric: str,
    result_ego: XprSmtResultRunEgo,
    vars_metadata: VarsMetadata,
    n_init: int,
    quant_to_variate_idx_1: int,
    quant_to_variate_idx_2: int,
    quant_fixed_vals: List[Tuple[int, float]],
    qual_filter: List[Tuple[int, Optional[List]]],
    n_grid: int,
    n_cols: int,
    figsize: Tuple[int, int],
    fps: int,
    save_path: str,
) -> None:

    n_iter_bo = len(result_ego.x_data) - n_init
    tmp_dir = tempfile.mkdtemp()
    frames = []

    for iter_bo in range(n_iter_bo + 1):
        print(f"  Frame {iter_bo}/{n_iter_bo}...")

        tmp_path = os.path.join(tmp_dir, f"frame_{iter_bo:03d}.png")

        plot_gp_at_iter(
            plot_type=plot_type,
            metric=metric,
            result_ego=result_ego,
            vars_metadata=vars_metadata,
            n_init=n_init,
            iter_bo=iter_bo,
            quant_to_variate_idx_1=quant_to_variate_idx_1,
            quant_to_variate_idx_2=quant_to_variate_idx_2,
            quant_fixed_vals=quant_fixed_vals,
            qual_filter=qual_filter,
            n_grid=n_grid,
            n_cols=n_cols,
            figsize=figsize,
            title=f"GP | iter_bo={iter_bo}/{n_iter_bo}",
            save_path=tmp_path,
        )

        frames.append(imageio.imread(tmp_path))

    if frames:
        target_shape = frames[0].shape[:2]
        frames = [resize_frame(f, target_shape) for f in frames]
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Animation saved at: {save_path}")

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)

def plot_convergence(
    y_data: np.ndarray,
    n_init: int,
    label: str = None,
    title: str = "Convergence",
    figsize: Tuple[int, int] = (8, 4),
    fontsize: int = 10,
    save_path: str = None,
) -> None:

    fig, ax = plt.subplots(figsize=figsize)

    y_flat = y_data.ravel()
    best = np.maximum.accumulate(y_flat)
    n = len(best)

    ax.plot(range(1, n + 1), best,
            color="blue", lw=2, marker="o", ms=4,
            label=f"{label} (Y best)" if label else "Y best")
    ax.plot(range(1, n + 1), y_flat,
            color="blue", lw=0.8, ls="--", alpha=0.5,
            label=f"{label} (Y)" if label else "Y")

    ax.axvspan(1, n_init + 0.5, alpha=0.07, color="orange", label="DoE")
    ax.axvspan(n_init + 0.5, n, alpha=0.07, color="green", label="BO")
    ax.axvline(n_init + 0.5, color="blue", ls="--", lw=1, alpha=0.6)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=fontsize - 2)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

##################
# OLD 

def plot_1d_ground_truth(
    fn_gp_objective: callable,
    vars_metadata: VarsMetadata,
    z: Any,
    quant_fixed_vals: List[Tuple[int, float]] = None,
    qual_filter: List[Tuple[int, Optional[List]]] = None,
    n_points: int = 200,
    title: str = "Ground truth",
    figsize: Tuple[int, int] = (9, 5),
    ncol_legend: int = 1,
    fontsize: int = 10,
    save_path: str = None,
) -> None:

    # construire la liste (var_idx, modalities) si qual_filter est None
    if qual_filter is None:
        qual_filter = [(i, None) for i in vars_metadata.get_categ_like_vars_indexes()]
        
    # affecter aux modalités None dans le filtre leur support complet avec les indices des modalités
    qual_filter_with_mods_indexes = []
    for var_idx, mods in qual_filter:
        full_support = vars_metadata[var_idx].support
        if mods is None:
            # récupérer indices de toutes les modalites
            modality_indexes = list(range(len(full_support)))
        else:
            # récupérer indices des modalités spécifiées
            modality_indexes = [full_support.index(m) for m in mods]
        qual_filter_with_mods_indexes.append((var_idx, modality_indexes))

    # combinaison de toutes les modalités entre variables qualitatives
    combinations = list(itertools.product(*[mods for _, mods in qual_filter_with_mods_indexes]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(combinations)))

    # une figure par variable quantitative (les autres sont fixées)
    quant_idxs = vars_metadata.get_continuous_vars_indexes()
    n_quant = len(quant_idxs)

    fig, axes = plt.subplots(1, n_quant, figsize=(figsize[0]*n_quant, figsize[1]))
    if n_quant == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=fontsize)

    # pour chaque variable quantitative
    for ax, quant_idx in zip(axes, quant_idxs):
        quant_meta = vars_metadata[quant_idx]
        quant_grid = np.linspace(quant_meta.support[0], quant_meta.support[1], n_points)

        # pour chaque combinaison de modalité des variables qualitatives
        for comb, color in zip(combinations, colors):
            # initialisation du vecteur de réference
            x_ref = np.full(len(vars_metadata), np.nan)

            # fixer les qualitatives à la combinaison en cours
            label_parts = []
            for k, (var_idx, _) in enumerate(qual_filter_with_mods_indexes):
                x_ref[var_idx] = comb[k]
                label_parts.append(f"{vars_metadata[var_idx].name}={vars_metadata[var_idx].support[comb[k]]}")

            # fixer les autres quantitatives que celle en cours
            if quant_fixed_vals:
                for fix_idx, fix_val in quant_fixed_vals:
                    x_ref[fix_idx] = fix_val

            # calculer h_true pour les valeurs de la grille de la variable quantitative en cours et aux autres valeurs fixées
            x_batch = np.tile(x_ref, (n_points, 1)) # repéter x_ref n_points fois selon l'axe 0 et 1 fois selon l'axe 1 (colonnes)
            x_batch[:, quant_idx] = quant_grid
            y_vals = fn_gp_objective(x=x_batch, vars_metadata=vars_metadata, z=z, sigma_noise=0)       
            ax.plot(quant_grid, y_vals, color=color, lw=2, label=", ".join(label_parts))

        # titre avec les valeurs fixeés des autres quantitatives
        if quant_fixed_vals:
            other_fixed = ", ".join(
                f"{vars_metadata[fix_idx].name}={fix_val:.3f}" for fix_idx, fix_val in quant_fixed_vals
                if fix_idx != quant_idx
            )

        ax.set_xlabel(quant_meta.name)
        ax.set_ylabel("h true")
        ax.set_title(f"h vs {quant_meta.name}\n({other_fixed} fixed)" if other_fixed else "")
        ax.legend(fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncol_legend)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fontsize - 2)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_1d_gp_quant_by_qual(
    sm: KrgBased, # compatible qu'avec smt (KrgBased) pour l'instant
    vars_metadata: VarsMetadata,
    x_opt: np.ndarray,
    quant_to_variate_idx: int,
    reverse_optim: bool,
    qual_filter: List[Tuple[int, Optional[List]]] = None,
    x_data: np.ndarray = None,
    y_data: np.ndarray = None,
    n_init: int = None,
    y_opt: float = None,
    x_true_max: np.ndarray = None,
    y_true_max: float = None,
    ic_factor: int = 1.96,
    n_points: int = 100,
    title: str = 'GP quantitative vs. qualitative variables',
    figsize: Tuple[int, int] = (9, 5),
    ncol_legend: int = 4,
    fontsize: int = 10,
    save_path: str = None,
) -> None:
    
    # grille pour la variable quantitative non fixée
    quant_to_variate_meta = vars_metadata[quant_to_variate_idx]
    quant_to_variate_grid = np.linspace(quant_to_variate_meta.support[0], quant_to_variate_meta.support[1], n_points)

    # construire la liste (var_idx, modalities) si qual_filter est None
    if qual_filter is None:
        qual_filter = [(i, None) for i in vars_metadata.get_categ_like_vars_indexes()]

    # affecter aux variables qualitatives les indices de toutes leurs modalités
    qual_filter_with_mods_indexes = []
    for var_idx, mods in qual_filter:
        full_support = vars_metadata[var_idx].support
        if mods is None:
            modality_indexes = list(range(len(full_support)))
        else:
            modality_indexes = [full_support.index(m) for m in mods]
        qual_filter_with_mods_indexes.append((var_idx, modality_indexes))

    # combinaison de toutes les modalités entre variables qualitatives
    combinations = list(itertools.product(*[mods for _, mods in qual_filter_with_mods_indexes]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(combinations)))

    fig, ax = plt.subplots(figsize=figsize)

    # pour chaque combinaison de modalités des variables qualitatives
    for comb, color in zip(combinations, colors):
        x_pred = np.tile(x_opt, (n_points, 1)) # préparer n points de valeurs x à l'optimum comme baseline
        x_pred[:, quant_to_variate_idx] = quant_to_variate_grid # assigner à cette baseline les valeurs de la grille de la variable quantitative non fixée
        label_parts = [] # labels des courbes pour chaque combinaison de modalités
   
        for k, (var_idx, _) in enumerate(qual_filter_with_mods_indexes): # pour chaque variable qualitative à afficher
            x_pred[:, var_idx] = comb[k] # assigner à l'indice de cette variable, l'indice de sa modalité dans la combinason en cours
            label_parts.append(f'{vars_metadata[var_idx].name}={vars_metadata[var_idx].support[comb[k]]}') # label de la courbe

        curve_label = ', '.join(label_parts) # labels des courbes pour chaque combinaison de modalités

        # calculer mu et sigma (pour la bande d'incertitude) du GP sur ces points créés
        sign = -1.0 if reverse_optim else 1.0 # si modèle trouvé par minimisation (comme EGO) alors qu'on voulait maximiser, le modèle prédit -y (car entrainé sur -y), donc on fait *-1
        mu = sign * sm.predict_values(x_pred).ravel() # ravel() car predict values retourne (n, 1) et ax.plot attend (n,)
        sd = np.sqrt(sm.predict_variances(x_pred).ravel())

        ax.fill_between(quant_to_variate_grid, mu - ic_factor*sd, mu + ic_factor*sd, alpha=0.15, color=color)
        ax.plot(quant_to_variate_grid, mu, color=color, lw=2, label=curve_label)

    if x_data is not None and y_data is not None:
        y_arr = y_data.ravel()

        # filtrer les points selon qual_filter
        mask_filter = np.ones(len(x_data), dtype=bool)
        for var_idx, modality_indexes in qual_filter_with_mods_indexes:
            mask_filter &= np.isin(x_data[:, var_idx].astype(int), modality_indexes)

        x_data_filtered = x_data[mask_filter]
        y_arr_filtered = y_arr[mask_filter]
        idx_filtered = np.where(mask_filter)[0]   # indices globaux

        mask_init = idx_filtered < n_init
        mask_bo = idx_filtered >= n_init

        ax.scatter(
            x_data_filtered[mask_init, quant_to_variate_idx],
            y_arr_filtered[mask_init],
            color='black', s=50, marker='o', zorder=5, label='DoE',
        )
        ax.scatter(
            x_data_filtered[mask_bo, quant_to_variate_idx],
            y_arr_filtered[mask_bo],
            color='black', s=50, marker='x', linewidths=1.5, zorder=5, label='BO',
        )

    # opt
    if y_opt is not None:
        x_opt_in_filter = True
        for var_idx, modality_indexes in qual_filter_with_mods_indexes:
            if int(x_opt[var_idx]) not in modality_indexes:
                x_opt_in_filter = False
                break
        if x_opt_in_filter:
            ax.scatter(
                x_opt[quant_to_variate_idx], y_opt,
                color='black', s=100, marker='*', zorder=10,
                label=f'BO opt (Y={y_opt:.3f})',
            )

    # true max
    if x_true_max is not None and y_true_max is not None:
        ax.scatter(x_true_max[quant_to_variate_idx], y_true_max,
                   color='red', s=100, marker='*', zorder=10, label=f'True max (Y={y_true_max:.3f})')

    ax.set_xlabel(quant_to_variate_meta.name)
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend(fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncol_legend)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=fontsize - 2)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()