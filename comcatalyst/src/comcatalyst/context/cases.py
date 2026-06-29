from dataclasses import dataclass
from typing import Any

from .fn_objectives import calc_gp_objective_01, calc_gp_objective_02, calc_gp_objective_03
from .vars_metadata import VarMetadata, VarsMetadata

@dataclass
class Case:
    vars_metadata: VarsMetadata
    z: Any
    fn_gp_objective: callable

def case_01() -> Case:
    vars_metadata = VarsMetadata([
        VarMetadata(name="x_qual1",  var_type="categorical", support=["A","B"]),
        VarMetadata(name="x_qual2",  var_type="categorical", support=["alpha","beta"]),
        VarMetadata(name="x_quant1", var_type="continuous",  support=(-1, 1)),
        VarMetadata(name="x_quant2", var_type="continuous",  support=(-1, 1)),
    ])

    z = {"z1": 1.0, "z2": 1.0}

    fn_gp_objective = lambda x, vars_metadata, z, sigma_noise, rng=None: calc_gp_objective_01(
        x=x,
        vars_metadata=vars_metadata,
        z=z,
        sigma_noise=sigma_noise,
        rng=rng,
        _SCALE_Q1={0: 2, 1: 2},
        _SCALE_Q2={0: 2, 1: 2},
    )

    return Case(vars_metadata=vars_metadata, z=z, fn_gp_objective=fn_gp_objective)


def case_02() -> Case:
    vars_metadata = VarsMetadata([
        VarMetadata(name="x_qual1", var_type="categorical", support=["A","B","C","D"]),
        VarMetadata(name="x_qual2", var_type="categorical", support=["alpha","beta","lambda","phi"]),
        VarMetadata(name="x_quant1", var_type="continuous", support=(-1, 1)),
        VarMetadata(name="x_quant2", var_type="continuous", support=(-1, 1)),
    ])

    z = {"z1": 1.0, "z2": 1.0}

    fn_gp_objective = lambda x, vars_metadata, z, sigma_noise, rng=None: calc_gp_objective_02(
        x=x,
        vars_metadata=vars_metadata,
        z=z,
        sigma_noise=sigma_noise,
        rng=rng,
        _SCALE_Q1={0: 2, 1: 2, 2: 1, 3: 1},
        _SCALE_Q2={0: 2, 1: 2, 2: 1, 3: 1},
    )

    return Case(vars_metadata=vars_metadata, z=z, fn_gp_objective=fn_gp_objective)


def case_03() -> Case:
    vars_metadata = VarsMetadata([
        VarMetadata(name="x_qual1", var_type="categorical", support=["A","B","C","D"]),
        VarMetadata(name="x_qual2", var_type="categorical", support=["alpha","beta","lambda","phi"]),
        VarMetadata(name="x_quant1", var_type="continuous", support=(-1, 1)),
        VarMetadata(name="x_quant2", var_type="continuous", support=(-1, 1)),
        VarMetadata(name="x_quant3", var_type="continuous", support=(-1, 1)),
    ])

    z = {"z1": 1.0, "z2": 1.0}

    fn_gp_objective = lambda x, vars_metadata, z, sigma_noise, rng=None: calc_gp_objective_03(
        x=x,
        vars_metadata=vars_metadata,
        z=z,
        sigma_noise=sigma_noise,
        rng=rng,
        _SCALE_Q1={0: 2, 1: 2, 2: 1, 3: 1},
        _SCALE_Q2={0: 2, 1: 2, 2: 1, 3: 1},
    )
    
    return Case(vars_metadata=vars_metadata, z=z, fn_gp_objective=fn_gp_objective)
