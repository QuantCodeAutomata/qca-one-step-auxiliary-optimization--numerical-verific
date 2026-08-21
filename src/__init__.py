"""
Systemic risk mean-field control: source modules.

Implements:
  - cutoff_operator: T_c cutoff map and threshold solver
  - absorbed_brownian: S_Delta absorbed Brownian semigroup
  - welfare: payoff function library and welfare evaluation
  - benchmarks: alternative feasible allocation rules
"""
from src.cutoff_operator import (
    compute_threshold,
    apply_cutoff,
    cutoff_operator,
    water_filling_cap,
)
from src.absorbed_brownian import (
    absorbed_heat_kernel,
    default_probability,
    evolve_law,
)
from src.welfare import (
    PAYOFF_FUNCTIONS,
    evaluate_welfare,
    evaluate_all_payoffs,
)
from src.benchmarks import (
    proportional_haircut,
    equal_deduction_strongest,
    randomized_feasible,
)

__all__ = [
    "compute_threshold",
    "apply_cutoff",
    "cutoff_operator",
    "water_filling_cap",
    "absorbed_heat_kernel",
    "default_probability",
    "evolve_law",
    "PAYOFF_FUNCTIONS",
    "evaluate_welfare",
    "evaluate_all_payoffs",
    "proportional_haircut",
    "equal_deduction_strongest",
    "randomized_feasible",
]
