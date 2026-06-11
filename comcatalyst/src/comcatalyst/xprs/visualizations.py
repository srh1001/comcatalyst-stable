import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import List, Tuple, Any, Optional

from smt.surrogate_models.krg_based import KrgBased

from ..vars.vars_metadata import VarsMetadata

def plot_comparison_convergence(
    y_datas: list,
    n_init: int,
    runs_labels: List[str] = [None],
    title: str = "Convergence",
    figsize: Tuple[int] = (8,4),
    save_path: str = None,
) -> None:

    fig, ax = plt.subplots(figsize=figsize)
    colors  = plt.cm.viridis(np.linspace(0, 1, len(y_datas)))
 
    all_best = [np.maximum.accumulate(y.ravel()) for y in y_datas]
 
    for best, y_data, label, color in zip(
        all_best, y_datas, runs_labels, colors
    ):
        n = len(best)
        ax.plot(range(1, n+1), best,
                color=color, lw=2, marker="o", ms=4, 
                label=f'{label} (Y best)' if label else 'Y best')
        ax.plot(
            range(1, n+1), y_data.ravel(), 
            color=color, lw=0.8, ls='--', alpha=0.5,
            label=f'{label} (Y)' if label else 'Y',
        )
        # separation DoE / BO avec la plage globale
        ax.axvspan(1, n_init, alpha=0.07, color="orange")
        ax.axvspan(n_init, n, alpha=0.07, color="green")
        ax.axvline(n_init, color=color, ls="--", lw=1, alpha=0.6)
 
    # légendes pour DoE et BO
    ax.axvspan(0, 0, alpha=0.2, color="orange", label="DoE")
    ax.axvspan(0, 0, alpha=0.2, color="green",  label="BO")
 
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_gp_quant_by_qual(
    sm: KrgBased,
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
        x_pred[:, quant_to_variate_idx] = quant_to_variate_grid # assigner à cette baseline les valeurs de la grille de la variable
                                                                # quantitative non fixée
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
        # points initiaux
        ax.scatter(x_data[:n_init, quant_to_variate_idx], y_arr[:n_init],
                   color='black', s=50, marker='o', zorder=5, label='DoE')
        # points bo
        ax.scatter(x_data[n_init:, quant_to_variate_idx], y_arr[n_init:],
                   color='black', s=50, marker='x', linewidths=1.5, zorder=5, label='BO')

    if y_opt is not None:
        ax.scatter(x_opt[quant_to_variate_idx], y_opt,
                   color='black', s=100, marker='*', zorder=10, label=f'BO opt (Y={y_opt:.3f})')

    if x_true_max is not None and y_true_max is not None:
        ax.scatter(x_true_max[quant_to_variate_idx], y_true_max,
                   color='red', s=100, marker='*', zorder=10, label=f'True max (Y={y_true_max:.3f})')

    ax.set_xlabel(quant_to_variate_meta.name)
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncol_legend)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_3d_gp_quant_by_qual(
    sm: KrgBased,
    vars_metadata: VarsMetadata,
    x_opt: np.ndarray,
    quant_to_variate_idx_1: int,
    quant_to_variate_idx_2: int,
    reverse_optim: bool,
    qual_filter: List[Tuple[int, Optional[List]]] = None,
    n_grid: int = 30,
    x_data: np.ndarray = None,
    y_data: np.ndarray = None,
    n_init: int = None,
    y_opt: float = None,
    x_true_max: np.ndarray = None,
    y_true_max: float = None,
    title: str = "GP surface quantitative vs. qualitative variables",
    figsize: Tuple[int, int] = (6, 5),
    n_cols: int = 3,
    save_path: str = None,
) -> None:
    
    quant_1 = vars_metadata[quant_to_variate_idx_1]
    quant_2 = vars_metadata[quant_to_variate_idx_2]

    # grille 2D pour les deux variables quantitatives
    grid_1 = np.linspace(quant_1.support[0], quant_1.support[1], n_grid)
    grid_2 = np.linspace(quant_2.support[0], quant_2.support[1], n_grid)
    G1, G2 = np.meshgrid(grid_1, grid_2)   # shape (n_grid, n_grid)

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
    n_combs = len(combinations)
    n_rows = (n_combs + n_cols - 1) // n_cols # nombre de lignes pour le nombre max de colonne spécifié

    fig = plt.figure(figsize=(figsize[0]*n_cols, figsize[1]*n_rows))
    fig.suptitle(title, fontsize=12)

    # pour chaque combinaison de modalités des variables qualitatives
    for i, comb in enumerate(combinations):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection="3d")

        # préparer n_grid x n_grid points de valeurs x à l'optimum comme baseline        
        x_pred = np.tile(x_opt, (n_grid*n_grid, 1))
        x_pred[:, quant_to_variate_idx_1] = G1.ravel()
        x_pred[:, quant_to_variate_idx_2] = G2.ravel()

        # fixer les variables qualitatives à la combinaison en cours
        subplot_label_parts = [] # labels des plots
        for k, (var_idx, _) in enumerate(qual_filter_with_mods_indexes): # pour chaque variable qualitative à afficher
            x_pred[:, var_idx] = comb[k] # assigner à l'indice de cette variable, l'indice de sa modalité dans la combinason en cours
            subplot_label_parts.append(f"{vars_metadata[var_idx].name}={vars_metadata[var_idx].support[comb[k]]}")

        # calculer mu du GP sur ces points créés
        sign = -1.0 if reverse_optim else 1.0 # si modèle trouvé par minimisation (comme EGO) alors qu'on voulait maximiser, le modèle prédit -y (car entrainé sur -y), donc on fait *-1
        mu = sign * sm.predict_values(x_pred).ravel() # ravel() car predict values retourne (n, 1) (1 car 1 seul y dans notre cas) et ax.plot attend (n,)
        MU = mu.reshape(n_grid, n_grid)

        ax.plot_surface(G1, G2, MU, cmap="viridis", alpha=0.8)

        if x_data is not None and y_data is not None and n_init is not None:
            y_arr = y_data.ravel()
            # points initiaux
            ax.scatter(x_data[:n_init, quant_to_variate_idx_1], x_data[:n_init, quant_to_variate_idx_2], y_arr[:n_init],
                color="black", s=40, marker="o", zorder=5, label="Initial points",
            )
            # points bo
            ax.scatter(x_data[n_init:, quant_to_variate_idx_1], x_data[n_init:, quant_to_variate_idx_2], y_arr[n_init:],
                color="black", s=40, marker="x", zorder=5, label="BO",
            )

        if y_opt is not None:
            ax.scatter(x_opt[quant_to_variate_idx_1], x_opt[quant_to_variate_idx_2], y_opt,
                color="black", s=100, marker="*", zorder=10, label=f"BO opt (Y={y_opt:.3f})",
            )

        if x_true_max is not None and y_true_max is not None:
            ax.scatter(x_true_max[quant_to_variate_idx_1], x_true_max[quant_to_variate_idx_2], y_true_max,
                color="red", s=100, marker="*", zorder=10, label=f"True max (Y={y_true_max:.3f})",
            )

        ax.set_title(", ".join(subplot_label_parts), fontsize=9)
        ax.set_xlabel(quant_1.name, fontsize=8)
        ax.set_ylabel(quant_2.name, fontsize=8)
        ax.set_zlabel("Y", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_ground_truth(
    fn_gp_objective: callable,
    vars_metadata: VarsMetadata,
    z: Any,
    quant_fixed_vals: List[Tuple[int, float]] = None,
    qual_filter: List[Tuple[int, Optional[List]]] = None,
    n_points: int = 200,
    title: str = "Ground truth",
    figsize: Tuple[int, int] = (9, 5),
    ncol_legend: int = 1,
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
    fig.suptitle(title, fontsize=12)

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
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncol_legend)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_3d_ground_truth(
    fn_gp_objective: callable,
    vars_metadata: VarsMetadata,
    z: Any,
    quant_idx_1: int,
    quant_idx_2: int,
    quant_fixed_vals: List[Tuple[int, float]] = None,
    qual_filter: List[Tuple[int, Optional[List]]] = None,
    n_points: int = 100,
    figsize: Tuple[int, int] = (5, 4),
    n_cols: int = 3,
    title: str = "Ground truth 3D",
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
            modality_indexes = list(range(len(full_support)))
        else:
            modality_indexes = [full_support.index(m) for m in mods]
        qual_filter_with_mods_indexes.append((var_idx, modality_indexes))

    # combinaison de toutes les modalités entre variables qualitatives
    combinations = list(itertools.product(*[mods for _, mods in qual_filter_with_mods_indexes]))
    n_combs = len(combinations)
    n_rows = (n_combs + n_cols - 1) // n_cols

    # grilles pour les deux variables quantitatives non fixées
    meta_1 = vars_metadata[quant_idx_1]
    meta_2 = vars_metadata[quant_idx_2]
    grid_1 = np.linspace(meta_1.support[0], meta_1.support[1], n_points)
    grid_2 = np.linspace(meta_2.support[0], meta_2.support[1], n_points)
    G1, G2 = np.meshgrid(grid_1, grid_2)  # shape (n_points, n_points)

    fig = plt.figure(figsize=(figsize[0]*n_cols, figsize[1]*n_rows))
    fig.suptitle(title, fontsize=12)

    # pour chaque combinaison de modalités des variables qualitatives
    for i, comb in enumerate(combinations):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection="3d")

        # vecteur de réference
        x_ref = np.full(len(vars_metadata), np.nan)

        # fixer les qualitatives à la combinaison en cours
        label_parts = []
        for k, (var_idx, _) in enumerate(qual_filter_with_mods_indexes):
            x_ref[var_idx] = comb[k]
            label_parts.append(
                f"{vars_metadata[var_idx].name}={vars_metadata[var_idx].support[comb[k]]}"
            )

        # fixer les autres quantitatives aux valeurs spécifiées
        if quant_fixed_vals:
            for fix_idx, fix_val in quant_fixed_vals:
                x_ref[fix_idx] = fix_val

        # evaluer h_true sur la grille 2D
        n_grid_points = n_points * n_points
        x_batch = np.tile(x_ref, (n_grid_points, 1)) # repéter x_ref n_grid_points fois selon l'axe 0 et 1 fois selon l'axe 1 (colonnes)
        x_batch[:, quant_idx_1] = G1.ravel()
        x_batch[:, quant_idx_2] = G2.ravel()
        H = fn_gp_objective(
            x=x_batch,
            vars_metadata=vars_metadata,
            z=z,
            sigma_noise=0,
        ).reshape(n_points, n_points)

        # maximum local de cette surface
        best_idx = np.argmax(H)
        best_row, best_col = np.unravel_index(best_idx, H.shape)
        best_quant1 = G1[best_row, best_col]
        best_quant2 = G2[best_row, best_col]
        best_h = H[best_row, best_col]

        # plot surface
        ax.plot_surface(G1, G2, H, cmap="viridis", alpha=0.8)
        # plot le maximum sur la surface
        ax.scatter(best_quant1, best_quant2, best_h, color="red", s=100, marker="*", zorder=10)
        ax.scatter(best_quant1, best_quant2, H.min(), color="red", s=100, marker="*", zorder=10)
        ax.plot(
            [best_quant1, best_quant1],
            [best_quant2, best_quant2],
            [H.min(), best_h],
            color="red", lw=1, ls="--",
        )
        ax.set_title(", ".join(label_parts) + f"\nmax={best_h:.3f} at ({best_quant1:.3f}, {best_quant2:.3f})", fontsize=9)
        ax.set_xlabel(meta_1.name, fontsize=8)
        ax.set_ylabel(meta_2.name, fontsize=8)
        ax.set_zlabel("h true", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()