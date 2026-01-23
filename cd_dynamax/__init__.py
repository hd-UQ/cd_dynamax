# cd_dynamax/__init__.py

# Nonlinear SSM
from .src.continuous_discrete_nonlinear_gaussian_ssm import (
    ContDiscreteNonlinearGaussianSSM,
    ParamsCDNLGSSM,
    cdnlgssm_filter,
    cdnlgssm_smoother,
    cdnlgssm_forecast,
    cdnlgssm_emissions,
    EKFHyperParams,
    UKFHyperParams,
    EnKFHyperParams,
)

from .src.continuous_discrete_nonlinear_ssm import (
    ContDiscreteNonlinearSSM,
    ParamsCDNLSSM,
    DPFHyperParams,
    cdnlssm_filter,
)

# Linear SSM
from .src.continuous_discrete_linear_gaussian_ssm import (
    ContDiscreteLinearGaussianSSM,
    ParamsCDLGSSM,
    cdlgssm_filter,
    cdlgssm_smoother,
    cdlgssm_forecast,
    cdlgssm_emissions,
    cdlgssm_posterior_sample,
    cdlgssm_joint_sample,
    KFHyperParams,
)

# Discrete-Discrete Linear SSM
from .dynamax.linear_gaussian_ssm import LinearGaussianSSM

# Shared pieces
from .src.ssm_temissions import SSM, Prior

# Utilities (the ones your demos use most)
from .src.utils.diffrax_utils import adjust_rhs
from .src.utils.optimize_utils import make_optimizer
from .src.utils.simulation_utils import make_key_sequence

__all__ = [
    # Models
    "ContDiscreteNonlinearGaussianSSM",
    "ContDiscreteNonlinearSSM",
    "ContDiscreteLinearGaussianSSM",
    "LinearGaussianSSM",
    # Params
    "ParamsCDNLGSSM",
    "ParamsCDNLSSM",
    "ParamsCDLGSSM",
    # Nonlinear algos
    "cdnlgssm_filter",
    "cdnlgssm_smoother",
    "cdnlgssm_forecast",
    "cdnlgssm_emissions",
    "cdnlssm_filter",
    "EKFHyperParams",
    "UKFHyperParams",
    "EnKFHyperParams",
    "DPFHyperParams",
    # Linear algos
    "cdlgssm_filter",
    "cdlgssm_smoother",
    "cdlgssm_forecast",
    "cdlgssm_emissions",
    "cdlgssm_posterior_sample",
    "cdlgssm_joint_sample",
    "KFHyperParams",
    # SSM/emissions
    "SSM",
    "Prior",
    # Utils
    "adjust_rhs",
    "make_optimizer",
    "make_key_sequence",
]
