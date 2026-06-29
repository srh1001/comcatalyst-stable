import json
import numpy as np
import itertools
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..utils import calc_convergence_iter_bo
from ..context.vars_metadata import VarsMetadata


def load_results_from_json(json_path: str) -> List[dict]:
    return json.loads(Path(json_path).read_text())


def load_results_from_dir(dir_path: str) -> List[dict]:
    results = []
    for json_path in Path(dir_path).rglob("*.json"):
        data = load_results_from_json(json_path=json_path)
        results.extend(data)
    return results


def slice_results_at_n_points(results: List[dict], n_points: int) -> List[dict]:
    sliced_results = []

    for r in results:
        n_init = r["hyperparams"]["n_init"]

        if n_points < n_init:
            raise ValueError(
                f"n_points (={n_points}) < n_init (={n_init})"
            )

        if n_points > len(r["x_data"]):
            raise ValueError(
                f"n_points (={n_points}) > n_total (={len(r['x_data'])})"
            )

        x_data_new = np.array(r["x_data"])[:n_points]
        y_data_new = np.array(r["y_data"])[:n_points]
        y_true_at_x_data_new = np.array(r["y_true_at_x_data"])[:n_points]

        best_idx = int(np.argmax(y_data_new.ravel()))
        x_opt = x_data_new[best_idx].tolist()
        y_opt = y_data_new.ravel()[best_idx]
        y_true_at_x_opt = y_true_at_x_data_new.ravel()[best_idx]

        n_iter_bo_new = n_points - n_init
        convergence_iter_bo_new = calc_convergence_iter_bo(
            y_data=y_data_new, n_init=n_init
        )

        r_sliced = {
            **r,
            "x_opt": x_opt,
            "y_opt": y_opt,
            "y_true_at_x_opt": y_true_at_x_opt,
            "convergence_iter_bo": convergence_iter_bo_new,
            "x_data": x_data_new.tolist(),
            "y_data": y_data_new.tolist(),
            "y_true_at_x_data": y_true_at_x_data_new.tolist(),
            "hyperparams": {
                **r["hyperparams"],
                "n_iter_bo": n_iter_bo_new,
            },
        }
        sliced_results.append(r_sliced)

    return sliced_results


def filter_results_by_hyperparams(results, filters):
    filtered = []
    for r in results:
        match = all(
            float(r["hyperparams"].get(k, None)) == float(v)
            if isinstance(v, (int, float))
            else str(r["hyperparams"].get(k)) == str(v)
            for k, v in filters.items()
        )
        if match:
            filtered.append(r)
    return filtered


def plot_convergence_curve(
    results: List[dict],
    metric: str, # y_data ou y_true_at_x_data
    filters: dict = None,
    figsize: tuple = (10, 5),
    ylim: Tuple[float, float] = (None, None),
    title: str = None,
    save_path: str = None,
) -> None:
    if filters:
        results = filter_results_by_hyperparams(results, filters)

    if not results:
        print("No results found after filter was applied.")
        return

    n_init = results[0]["hyperparams"]["n_init"]
    n_pts  = min(len(r[metric]) for r in results) # longueur minimale commune
    
    # pour chaque run, calculer la courbe
    all_curves = []
    for r in results:
        y = np.array(r[metric]).ravel()[:n_pts]
        y_best = np.maximum.accumulate(y) # meilleur y jusqu'a chaque iteration
        all_curves.append(y_best)

    all_curves = np.array(all_curves) # shape (n_runs, n_pts)
    mean = all_curves.mean(axis=0)
    std = all_curves.std(axis=0)
    lo = all_curves.min(axis=0)
    hi = all_curves.max(axis=0)
    iters = np.arange(1, n_pts + 1)

    fig, ax = plt.subplots(figsize=figsize)

    # courbe par run en transparence
    for curve in all_curves:
        ax.plot(iters, curve, color="gray", lw=0.8, alpha=0.2)

    # mean
    ax.plot(iters, mean, color="blue", lw=2, label="mean")
    # std
    ax.fill_between(iters, mean - std, mean + std, alpha=0.2, color="blue", label="std")
    # min / max
    ax.fill_between(iters, lo, hi, alpha=0.1, color="blue", label="[min, max]")

    # separation DoE / BO
    ax.axvspan(1, n_init + 0.5, alpha=0.07, color="orange", label="DoE")
    ax.axvspan(n_init + 0.5, n_pts, alpha=0.07, color="green", label="BO")
    ax.axvline(n_init + 0.5, color="black", ls="--", lw=1)

    # true max
    ax.axhline(results[0]["y_true_max"], color="red", ls="--", lw=1)

    if title is None:
        filter_str = ", ".join(f"{k}={v}" for k, v in (filters or {}).items())
        title = f"{metric}" + (f" | {filter_str}" if filter_str else "")

    ax.set_xlabel("iterations")
    ax.set_ylabel("y_best")
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_boxplot(
    results: List[dict],
    metric: str,
    group_by: str,
    filters: Dict[str, Any] = None,
    figsize: Tuple[int, int] = (7, 5),
    ylim: Tuple[float, float] = (None, None),
    title: str = None,
    save_path: str = None,
) -> None:

    if filters:
        results = filter_results_by_hyperparams(results, filters)

    if not results:
        print("No results found after filter was applied.")
        return
        
    # grouper par valeur de group_by
    groups = {}
    for r in results:
        val = r["hyperparams"].get(group_by)
        if val is None:
            continue
        key = str(val)
        if key not in groups:
            groups[key] = []
        groups[key].append(r[metric])

    # trier les groupes par valeur
    try:
        sorted_keys = sorted(groups.keys(), key=lambda x: float(x))
    except ValueError:
        sorted_keys = sorted(groups.keys())

    data   = [groups[k] for k in sorted_keys]
    labels = [f"{k}" for k in sorted_keys]
    labels = [f"{k}\n(tot={len(groups[k])})" for k in sorted_keys]
    
    if title is None:
        filter_str = ", ".join(f"{k}={v}" for k, v in (filters or {}).items())
        title = f"{metric} by {group_by}" + (f" | {filter_str}" if filter_str else "")

    ylabel = metric

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, tick_labels=labels)

    for i, (key, d) in enumerate(zip(sorted_keys, data), start=1):
        median_val = np.median(d)
        mean_val = np.mean(d)
        min_val= np.min(d)
        max_val = np.max(d)

        ax.text(i, median_val, f"med={median_val:.3f}", ha="center", va="bottom", fontsize=7, color="orange")
        ax.text(i, mean_val, f"mean={mean_val:.3f}", ha="center", va="center", fontsize=7, color="blue")
        ax.text(i, min_val, f"{min_val:.3f}", ha="center", va="top", fontsize=7, color="black")
        ax.text(i, max_val, f"{max_val:.3f}", ha="center", va="bottom", fontsize=7, color="black")
    
    # true max
    if metric == "y_opt" or metric == "y_true_at_x_opt":
        ax.axhline(results[0]["y_true_max"], color="red", ls="--", lw=1)

    ax.set_xlabel(group_by)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------- Investigate results

def summarize_results(results: List[dict], epsilon: float, filters: dict = None) -> None:
    if filters:
        results = filter_results_by_hyperparams(results, filters)

    if not results:
        print("No results found after filter was applied.")
        return
    
    y_true_max = results[0]["y_true_max"]
    all_y_true_at_x_opt = [r["y_true_at_x_opt"] for r in results]
    all_y_opt = [r["y_opt"] for r in results]
    all_y_init_max = [np.array(r['y_data'][:r['hyperparams']['n_init']]).max() for r in results]
    all_seeds = [r["hyperparams"]["seed"] for r in results]
    all_convergence_iter_bo = [r['convergence_iter_bo'] for r in results]

    # taux de bons y gap selon un seuil
    n_success = sum(1 for g in all_y_true_at_x_opt if (np.abs(y_true_max - g) < epsilon))

    # means
    print(f"N runs={len(results)}")
    print(f"\ny_opt mean={np.mean(all_y_opt):.3f}, std={np.std(all_y_opt):.3f}, min={np.min(all_y_opt):.3f}, max={np.max(all_y_opt):.3f}")
    print(f"y_true_at_x_opt mean={np.mean(all_y_true_at_x_opt):.3f}, std={np.std(all_y_true_at_x_opt):.3f}, min={np.min(all_y_true_at_x_opt):.3f}, max={np.max(all_y_true_at_x_opt):.3f}")
    print(f"conv_iter_bo mean={np.mean(all_convergence_iter_bo):.1f}, std={np.std(all_convergence_iter_bo):.1f}, min={np.min(all_convergence_iter_bo)}, max={np.max(all_convergence_iter_bo)}")
    print(f"\nSuccess rate gap={epsilon}: {n_success}/{len(results)}={100*n_success/len(results):.1f}%")

    # correlations
    print(f"\nCorrelation y_opt vs conv_iter_bo: {np.corrcoef(all_y_opt, all_convergence_iter_bo)[0,1]:.3f}")
    print(f"Correlation y_true_at_x_opt vs conv_iter_bo: {np.corrcoef(all_y_true_at_x_opt, all_convergence_iter_bo)[0,1]:.3f}")
    
    print(f"\nCorrelation y_opt vs y_init_max: {np.corrcoef(all_y_opt, all_y_init_max)[0,1]:.3f}")
    print(f"Correlation y_true_at_x_opt vs y_init_max: {np.corrcoef(all_y_true_at_x_opt, all_y_init_max)[0,1]:.3f}")
    
    print(f"\nCorrelation y_opt vs seed: {np.corrcoef(all_y_opt, all_seeds)[0,1]:.3f}")
    print(f"Correlation y_true_at_x_opt vs seed: {np.corrcoef(all_y_true_at_x_opt, all_seeds)[0,1]:.3f}")


def stats_x_opt_qual_combos(
    results: List[dict],
    vars_metadata: Any,
    metric: str,
    metric_range: Tuple[float, float],
    filters: dict = None,
) -> None:
    if filters:
        results = filter_results_by_hyperparams(results, filters)

    lb, ub  = metric_range
    results = [r for r in results if lb <= r[metric] <= ub]

    if not results:
        print(f"No result for {metric} in [{lb}, {ub}].")
        return

    categ_idxs = [i for i, m in enumerate(vars_metadata) if m.is_categ_like()]

    combo_counts = {}
    for r in results:
        x_opt = np.array(r["x_opt"])
        combo = "/".join(
            vars_metadata[i].support[int(x_opt[i])]
            for i in categ_idxs
        )
        combo_counts[combo] = combo_counts.get(combo, 0) + 1

    total = len(results)
    print(f"\n{total} runs with {metric} in [{lb:.3f}, {ub:.3f}]")
    print(f"\n{'Combo':<20} {'n':>4}  {'%':>6}")
    print("-" * 35)
    for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1]):
        print(f"{combo:<20} {count:>4}  {100*count/total:>5.1f}%")


def details_doe_by_metric_range(
    results: List[dict],
    vars_metadata: VarsMetadata, 
    metric: str,
    metric_range: Tuple[float, float],
    filters: dict = None,
    top_k: int = None
) -> None:
    if filters:
        results = filter_results_by_hyperparams(results, filters)

    lb, ub  = metric_range
    results = [r for r in results if lb <= r[metric] <= ub]

    if not results:
        print(f"No result for {metric} in [{lb}, {ub}].")
        return

    # tri par ordre décroissant metric
    results = sorted(results, key=lambda r: r[metric], reverse=True)
    if top_k is not None:
        results = results[:top_k] + results[-top_k:]

    print(f"\n{'='*60}")
    print(f"{len(results)} runs with {metric} in [{lb:.3f}, {ub:.3f}]")
    print(f"{'='*60}")

    for (k, r) in enumerate(results):
        n_init = r["hyperparams"]["n_init"]
        x_init = np.array(r["x_data"][:n_init])
        y_init = np.array(r["y_data"][:n_init]).ravel()

        # effectif par combinaison qualitative
        categ_idxs = [i for i, m in enumerate(vars_metadata) if m.is_categ_like()]
        categ_counts = {}
        for x in x_init:
            key = tuple(int(x[idx]) for idx in categ_idxs)
            categ_counts[key] = categ_counts.get(key, 0) + 1

        # range quantitatif par combinaison
        cont_idxs = [i for i, m in enumerate(vars_metadata) if m.is_continuous()]

        if top_k is not None and k==0:
            print(f"\n === BEST TOP {top_k} ===")

        elif top_k is not None and k % top_k == 0:
            print(f"\n === WORST TOP {top_k} ===")

        print(f"\n  --- {metric}={r[metric]:.3f}")
        print(f"  {'Combo':<20} {'n':>3}  {'quant ranges':<50}  {'y range'}")
        print(f"  {'-'*100}")

        for combo, count in sorted(
            categ_counts.items(),
            key=lambda item: "/".join( # affichage dans l'ordre alphabétique des catégorielles
                vars_metadata[idx].support[item[0][j]]
                for j, idx in enumerate(categ_idxs)
            )
        ):            
            # masque pour cette combinaison
            mask = np.ones(len(x_init), dtype=bool)
            for j, idx in enumerate(categ_idxs):
                mask &= (x_init[:, idx] == combo[j])

            # labels des modalités
            combo_label = "/".join(
                vars_metadata[idx].support[combo[j]]
                for j, idx in enumerate(categ_idxs)
            )

            # range quantitatif
            x_combo = x_init[mask]
            y_combo = y_init[mask]
            quant_ranges = ", ".join(
                f"{vars_metadata[idx].name}=[{x_combo[:, idx].min():.2f}, {x_combo[:, idx].max():.2f}]"
                for idx in cont_idxs
            )

            y_range = f"[{y_combo.min():.3f}, {y_combo.max():.3f}]"

            print(f"  {combo_label:<20} {count:>3}  {quant_ranges:<50}  {y_range}")

def plot_boxplot_by_seed(
    results: List[dict],
    metric: str = "y_opt",
    filters: dict = None,
    figsize: tuple = (12, 5),
    title: str = None,
    save_path: str = None,
) -> None:

    if filters:
        results = filter_results_by_hyperparams(results, filters)

    # grouper par seed
    groups = {}
    for r in results:
        seed = r["hyperparams"]["seed"]
        if seed not in groups:
            groups[seed] = []
        groups[seed].append(r[metric])

    sorted_seeds = sorted(groups.keys())
    data = [groups[s] for s in sorted_seeds]
    labels = [str(s) for s in sorted_seeds]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, tick_labels=labels)
    ax.set_xlabel("seed")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by seed")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_trajectory_by_qual(
    result: dict,
    vars_metadata: VarsMetadata,
    quant_idx_1: int,
    quant_idx_2: int,
    x_true_max: np.ndarray = None,
    figsize: tuple = (7, 6),
    title: str = None,
) -> None:
    x_data = np.array(result["x_data"])
    y_true_at_x_data = np.array(result["y_true_at_x_data"]).ravel()
    n_init = result["hyperparams"]["n_init"]
    n_pts  = len(x_data)

    categ_idxs = [i for i, m in enumerate(vars_metadata) if m.is_categ_like()]
    categ_combos = list(itertools.product(*[
        vars_metadata[i].support for i in categ_idxs
    ]))
    n_combos = len(categ_combos)
    n_cols = min(4, n_combos)
    n_rows = (n_combos + n_cols - 1) // n_cols

    meta_1 = vars_metadata[quant_idx_1]
    meta_2 = vars_metadata[quant_idx_2]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows))
    axes = np.array(axes).ravel()
    fig.suptitle(title)

    for ax, combo in zip(axes, categ_combos):
        combo_label = "/".join(combo)

        # masque pour cette combinaison qualitative
        mask = np.ones(n_pts, dtype=bool)
        for k, idx in enumerate(categ_idxs):
            mask &= np.array([
                vars_metadata[idx].support[int(x_data[j, idx])] == combo[k]
                for j in range(n_pts)
            ])

        x1_all = x_data[mask, quant_idx_1]
        x2_all = x_data[mask, quant_idx_2]
        y_all  = y_true_at_x_data[mask]
        idx_all = np.where(mask)[0] # indices globaux

        # points doe et bo
        mask_init = idx_all < n_init
        mask_bo = idx_all >= n_init

        ax.scatter(
            x1_all[mask_init], x2_all[mask_init],
            c=y_all[mask_init], cmap="viridis",
            s=80, marker="x", zorder=5, label="DoE",
            vmin=y_true_at_x_data.min(), vmax=y_true_at_x_data.max(),
        )
        ax.scatter(
            x1_all[mask_bo], x2_all[mask_bo],
            c=y_all[mask_bo], cmap="viridis",
            s=80, marker="o", linewidths=1.5, zorder=5, label="BO",
            vmin=y_true_at_x_data.min(), vmax=y_true_at_x_data.max(),
        )

        # numerotation des points
        for j, (x1, x2, glob_idx) in enumerate(zip(x1_all, x2_all, idx_all)):
            ax.annotate(
                f"{glob_idx + 1} ({y_true_at_x_data[glob_idx]:.2f})",
                xy=(x1, x2), fontsize=6,
                xytext=(4, 4), textcoords="offset points",
            )

        # fleches entre points consecutifs de cette combinaison
        for j in range(len(idx_all) - 1):
            ax.annotate(
                '',
                xy=(x1_all[j+1], x2_all[j+1]),
                xytext=(x1_all[j], x2_all[j]),
                arrowprops=dict(arrowstyle='-|>', color='gray', lw=1.0, alpha=0.8),
            )

        # vrai optimum
        if x_true_max is not None:
            ax.scatter(
                x_true_max[quant_idx_1], x_true_max[quant_idx_2],
                color="red", s=100, marker="*", zorder=10, label="True max",
            )

        ax.set_xlim(meta_1.support)
        ax.set_ylim(meta_2.support)
        ax.set_xlabel(meta_1.name, fontsize=8)
        ax.set_ylabel(meta_2.name, fontsize=8)
        ax.set_title(combo_label, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # masquer les axes vides
    for ax in axes[n_combos:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()