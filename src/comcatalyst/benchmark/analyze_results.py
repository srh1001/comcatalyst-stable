import json
import numpy as np
import itertools
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

        r_sliced = {
            **r,
            "x_opt": x_opt,
            "y_opt": y_opt,
            "y_true_at_x_opt": y_true_at_x_opt,
            "idx_opt": best_idx,
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
    vline: int = None,
    fontsize: int = 10,
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

    if vline:
        ax.axvline(vline, color="red", ls="--", lw=1)

    if title is None:
        filter_str = ", ".join(f"{k}={v}" for k, v in (filters or {}).items())
        title = f"{metric}" + (f" | {filter_str}" if filter_str else "")

    ax.set_xlabel("iterations", fontsize=fontsize)
    ax.set_ylabel("y_best", fontsize=fontsize)
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_title(title, fontsize=fontsize-2)
    ax.legend(fontsize=fontsize - 1)
    ax.tick_params(axis="both", labelsize=fontsize - 1)
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
    fontsize: int = 10,
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

    data = [groups[k] for k in sorted_keys]
    labels = [f"{k}\n(n={len(groups[k])}, std={np.std(groups[k]):.3f})" for k in sorted_keys]    
    
    if title is None:
        filter_str = ", ".join(f"{k}={v}" for k, v in (filters or {}).items())
        title = f"{metric} by {group_by}" + (f" | {filter_str}" if filter_str else "")

    ylabel = metric

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, tick_labels=labels)

    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    y_min, y_max = ax.get_ylim()


    for i, (key, d) in enumerate(zip(sorted_keys, data), start=1):
        median_val = np.median(d)
        mean_val = np.mean(d)
        min_val= np.min(d)
        max_val = np.max(d)

        if y_min <= median_val <= y_max:
            ax.text(i, median_val, f"med={median_val:.3f}", ha="center", va="bottom", fontsize=fontsize-2, color="orange")
        if y_min <= mean_val <= y_max:
            ax.text(i, mean_val, f"mean={mean_val:.3f}", ha="center", va="center", fontsize=fontsize-2, color="blue")
        if y_min <= min_val <= y_max:
            ax.text(i, min_val, f"{min_val:.3f}", ha="center", va="top", fontsize=fontsize-2, color="black")
        if y_min <= max_val <= y_max:
            ax.text(i, max_val, f"{max_val:.3f}", ha="center", va="bottom", fontsize=fontsize-2, color="black")
        
    # true max
    if metric == "y_opt" or metric == "y_true_at_x_opt":
        ax.axhline(results[0]["y_true_max"], color="red", ls="--", lw=1)

    ax.set_xlabel(group_by, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize-2)
    ax.tick_params(axis="both", labelsize=fontsize - 2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def summarize_results(
    results: List[dict],
    vars_metadata: Any,
    epsilon: float,
    filters: dict = None,
) -> None:
    if filters:
        results = filter_results_by_hyperparams(results, filters)

    if not results:
        print("No results found after filter was applied.")
        return

    cat_dims: list[int] = vars_metadata.get_categorical_vars_indexes()
    cont_dims: list[int] = vars_metadata.get_continuous_vars_indexes()

    x_true_max: np.ndarray = np.array(results[0]["x_true_max"])
    y_true_max: float = results[0]["y_true_max"]

    all_y_true_at_x_opt: list[float] = []
    all_y_opt: list[float] = []
    all_y_init_max: list[float] = []
    all_correct_combo_in_init: list[int] = []
    all_correct_modal_in_init: dict[int, list[int]] = {i: [] for i in cat_dims}
    all_dist_best_init_to_opt: list[float] = []
    all_noise: list[float] = []

    for r in results:
        n_init: int = r["hyperparams"]["n_init"]
        x_data: np.ndarray = np.array(r["x_data"])
        y_data: np.ndarray = np.array(r["y_data"])
        y_true_at_x_data: np.ndarray = np.array(r["y_true_at_x_data"]).ravel()
        x_init: np.ndarray = x_data[:n_init]

        all_y_true_at_x_opt.append(r["y_true_at_x_opt"])
        all_y_opt.append(r["y_opt"])
        all_y_init_max.append(float(y_data[:n_init].max()))

        mask: np.ndarray = np.all(x_init[:, cat_dims] == x_true_max[cat_dims], axis=1)
        all_correct_combo_in_init.append(int(mask.sum()))

        for cat_idx in cat_dims:
            all_correct_modal_in_init[cat_idx].append(
                int(np.sum(x_init[:, cat_idx] == x_true_max[cat_idx]))
            )

        best_init_idx: int = int(np.argmax(y_data[:n_init]))
        all_dist_best_init_to_opt.append(
            float(np.linalg.norm(x_data[best_init_idx][cont_dims] - x_true_max[cont_dims]))
        )

        all_noise.append(float(np.mean(np.abs(y_data - y_true_at_x_data))))

    n_success: int = sum(1 for g in all_y_true_at_x_opt if np.abs(y_true_max - g) < epsilon)

    print(f"N runs={len(results)}")
    print(f"\ny_opt mean={np.mean(all_y_opt):.3f}, std={np.std(all_y_opt):.3f}, min={np.min(all_y_opt):.4f}, max={np.max(all_y_opt):.4f}")
    print(f"y_true_at_x_opt mean={np.mean(all_y_true_at_x_opt):.3f}, std={np.std(all_y_true_at_x_opt):.3f}, min={np.min(all_y_true_at_x_opt):.4f}, max={np.max(all_y_true_at_x_opt):.4f}")
    print(f"\nSuccess rate gap={epsilon}: {n_success}/{len(results)}={100*n_success/len(results):.1f}%")

    print(f"\nCorrelation y_true_at_x_opt vs y_init_max: {np.corrcoef(all_y_true_at_x_opt, all_y_init_max)[0,1]:.3f}")
    print(f"Correlation y_true_at_x_opt vs n correct combo in init: {np.corrcoef(all_y_true_at_x_opt, all_correct_combo_in_init)[0,1]:.3f}")

    for cat_idx in cat_dims:
        var_name: str = vars_metadata[cat_idx].name
        modal: int = int(x_true_max[cat_idx])
        print(f"Correlation y_true_at_x_opt vs n '{var_name}={modal}' in init: {np.corrcoef(all_y_true_at_x_opt, all_correct_modal_in_init[cat_idx])[0,1]:.3f}")

    print(f"Correlation y_true_at_x_opt vs dist best init to x_true_max: {np.corrcoef(all_y_true_at_x_opt, all_dist_best_init_to_opt)[0,1]:.3f}")
    print(f"Correlation y_true_at_x_opt vs mean abs noise: {np.corrcoef(all_y_true_at_x_opt, all_noise)[0,1]:.3f}")

def stats_bo_errors(
    results: list[dict],
    vars_metadata: Any,
    filters: dict = None,
) -> dict:
    if filters:
        results = filter_results_by_hyperparams(results, filters)

    cat_dims = vars_metadata.get_categorical_vars_indexes()
    cont_dims = vars_metadata.get_continuous_vars_indexes()

    n_correct_combo = 0
    n_x_opt_in_init = 0
    dist_correct_combo = []
    dist_wrong_combo = []

    for result in results:
        n_init = result["hyperparams"]["n_init"]
        x_opt = np.array(result["x_opt"])
        x_true_max = np.array(result["x_true_max"])

        is_correct_combo = bool(np.all(x_opt[cat_dims] == x_true_max[cat_dims]))

        if is_correct_combo:
            n_correct_combo += 1
            dist_correct_combo.append(float(np.linalg.norm(x_opt[cont_dims] - x_true_max[cont_dims])))
        else:
            dist_wrong_combo.append(float(np.linalg.norm(x_opt[cont_dims] - x_true_max[cont_dims])))

        best_idx = int(np.argmax(np.array(result["y_data"])))
        if best_idx < n_init:
            n_x_opt_in_init += 1

    n_runs = len(results)
    print(f"N runs={n_runs}")

    print(f"\n{'Prop correct combo of x_opt':<{58}}: {n_correct_combo / n_runs:.3f}")
    print(f"{'Mean quant dist to x_true_max for correct combo of x_opt':<{58}}: {float(np.mean(dist_correct_combo)) if dist_correct_combo else None}")
    print(f"{'Std quant dist to x_true_max for correct combo of x_opt':<{58}}: {float(np.std(dist_correct_combo)) if dist_correct_combo else None}")
    print(f"{'Mean quant dist to x_true_max for wrong combo of x_opt':<{58}}: {float(np.mean(dist_wrong_combo)) if dist_wrong_combo else None}")
    print(f"{'Std quant dist to x_true_max for wrong combo of x_opt':<{58}}: {float(np.std(dist_wrong_combo)) if dist_wrong_combo else None}")
    print(f"{'Prop x_opt in x_init':<{58}}: {n_x_opt_in_init / n_runs:.3f}")

def stats_x_opt_qual_combos(
    results: List[dict],
    vars_metadata: Any,
    metric: str,
    metric_range: Tuple[float, float],
    filters: dict = None,
) -> None:
    if filters:
        results = filter_results_by_hyperparams(results, filters)

    lb, ub = metric_range
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
    print(f"\n{total} runs with {metric} in [{lb}, {ub}]")
    print(f"\n{'Combo':<20} {'n':>4}  {'%':>6}")
    print("-" * 35)
    for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1]):
        print(f"{combo:<20} {count:>4}  {100*count/total:>6.1f}%")

def stats_seed_variance(
    results: list[dict],
    metric: str,
    filters: dict = None,
) -> None:
    if filters:
        results = filter_results_by_hyperparams(results, filters)

    if not results:
        print("No results found after filter was applied.")
        return

    # Groupement par seed
    by_seed: dict[int, list[float]] = {}
    for r in results:
        seed = r["hyperparams"]["seed"]
        y = r[metric]
        by_seed.setdefault(seed, []).append(y)

    # Variance intra-seed et inter-seed
    intra_vars: list[float] = []
    seed_means: list[float] = []

    for seed, ys in by_seed.items():
        intra_vars.append(float(np.var(ys)))
        seed_means.append(float(np.mean(ys)))

    var_intra = float(np.mean(intra_vars))
    var_inter = float(np.var(seed_means))

    print(f"N seeds={len(by_seed)}, N runs={len(results)}")
    print(f"\n{'Variance intra-seed':<35} : {var_intra:.4f}")
    print(f"{'Variance inter-seed':<35} : {var_inter:.4f}")
    print(f"{'Ratio inter/intra':<35} : {var_inter/var_intra:.3f}")

def details_doe_by_metric_range(
    results: List[dict],
    vars_metadata: VarsMetadata, 
    metric: str,
    metric_range: Tuple[float, float],
    top_k: int = None,
    filters: dict = None,
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
    print(f"{len(results)} runs with {metric} in [{lb}, {ub}]")
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

        print(f"\n  --- {metric}={r[metric]:.4f}  |  {r["hyperparams"]}")
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

            y_range = f"[{y_combo.min():.4f}, {y_combo.max():.4f}]"

            print(f"  {combo_label:<20} {count:>3}  {quant_ranges:<50}  {y_range}")

def plot_boxplot_by_seed(
    results: List[dict],
    metric: str = "y_opt",
    filters: dict = None,
    figsize: tuple = (12, 5),
    title: str = None,
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