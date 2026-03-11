# Simple functionality tests

- [test_imports.py](./test_imports.py) checks that key packages can be imported without errors.

- [test_models.py](./test_models.py) checks that basic model creation, parameter setting and sampling data from model works for cd-dynamax models.

- [test_utils_imports.py](./test_utils_imports.py) checks that key (debug and difrax) utility functions can be imported without errors.

# Tests for cd-dynamax performance

- [test_filter_forecast_emissions.py](./test_filter_forecast_emissions.py) checks that the cd-dynamax filter and forecast functions for all model classes (CD-LGSSM, CD-NLGSSM and CD-NLSSM) can be run without errors, and that the forecasted emissions have the expected shapes and are finite.

- [test_cdlgssm_dlgssm_match.py]() checks that cd-dynamax's CD-LGSSM implementation matches the Discrte Linear Gaussian SSM implementation of dynamax, when sampling, filtering, smoothing, forecasting and sampling emissions from model.

- [test_cdnonlinear_cdlinear_match.py](./test_cdnonlinear_cdlinear_match.py) checks that the cd-nonlinear SSM implementations (CD-NLGSSM and CD-NLSSM) match the linear Kalman filter and smoother results when applied to a linear Gaussian SSM. This is a key test to ensure that cd-dynamax nonlinear model implementations are correct, since for linear models it should exactly match the Kalman filter and smoother results.
