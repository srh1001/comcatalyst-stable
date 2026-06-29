from datetime import datetime
import json
import numpy as np
from pathlib import Path
import os
from typing import Any, List, Optional, Tuple
import itertools

from comcatalyst.context.vars_metadata import VarsMetadata
from comcatalyst.xprs.xpr import Xpr, XprResultRun


def log_error(msg: str, save_path: str) -> None:
    if save_path is None:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line  = f"[{timestamp}] {msg}\n"
    with open(save_path, "a") as f:
        f.write(log_line)
    print(f"  [LOG] {log_line.strip()}")

def benchmark(
    xpr: Xpr,
    fn_gp_objective: callable,
    vars_metadata: VarsMetadata,
    z: Any,
    n_grid_true_max: int,
    fixed_hyperparams_dict: dict,
    varying_hyperparams_dict: List[Tuple[str, List[Any]]],
    save_path_results: str,
    save_path_logs: str,
    max_n_points: int = None,
    x_init: Optional[np.ndarray] = None,
    y_init: Optional[np.ndarray] = None,
) -> Tuple[list, int]:

    x_true_max, y_true_max, _, _ = Xpr.calc_true_max(
        fn_gp_objective=fn_gp_objective, 
        vars_metadata=vars_metadata, 
        z=z, 
        n_grid=n_grid_true_max
    )

    vh_names = [name for name, _ in varying_hyperparams_dict]
    vh_vals  = [vals for _, vals in varying_hyperparams_dict]

    save_path_obj = Path(save_path_results)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
 
    all_results: List[XprResultRun] = []
    vh_combos = list(itertools.product(*vh_vals))
    total = len(vh_combos)

    run_id = 0
    
    print(f"\n--- Experiments for hyperparameters {vh_names} ---")

    for vh_combo in vh_combos:
        hyperparams_dict = {
            **fixed_hyperparams_dict,
            **dict(zip(vh_names, vh_combo)),
        }

        hyperparams = xpr.Hyperparams(**hyperparams_dict)

        if max_n_points is not None:
            if max_n_points < hyperparams.n_init:
                raise ValueError(f"max_n_points (={max_n_points}) cannot be < n_init (={hyperparams.n_init})")
        
            elif hyperparams.n_init + hyperparams.n_iter_bo > max_n_points:
                hyperparams.n_iter_bo = max_n_points - hyperparams.n_init

        run_desc = (
            ", ".join(f"{n}={getattr(hyperparams, n)}" for n in vh_names)
        )

        print("\n", " "*2 + f"\n({run_id+1}/{total}) Running {run_desc}...")

        try:
            result = xpr.run(
                fn_gp_objective=fn_gp_objective,
                vars_metadata=vars_metadata,
                z=z,
                x_true_max=x_true_max,
                y_true_max=y_true_max,
                hyperparams=hyperparams,
                x_init=x_init,
                y_init=y_init,
            )

            if save_path_obj.exists():
                existing = json.loads(save_path_obj.read_text())
            else:
                existing = []
            existing.append(result.to_json_dict())
            save_path_obj.write_text(json.dumps(existing, indent=2))
            all_results.append(result)

            print("\n", " "*4, "Results:")
            xpr.print_opt_result(x=result.x_opt, y=result.y_opt, vars_metadata=vars_metadata, indent=8)
            print(" "*8 + f"y_true_at_x_opt = {result.y_true_at_x_opt:.3f}")

        except Exception as e:
            log_error(
                msg=f"FAILED run {run_id}/{total} | {run_desc} | {type(e).__name__}: {e}",
                save_path=save_path_logs
            )

        run_id += 1

    print(f"Results saved in {save_path_results}")
    return all_results