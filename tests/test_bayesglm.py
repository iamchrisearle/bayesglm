import numpy as np
from bayesglm.bayesglm import BayesGLM
from scipy.special import expit
import pytest


class TestBayesGLM:
    model_families = [
        "gaussian",
        "logit",
        "probit",
        "poisson",
    ]

    @pytest.fixture
    def get_test_data_with_constant(self):
        np.random.seed(12345)
        n = 100
        x1 = np.random.normal(size=n)
        b0 = 1
        const = np.ones(n)
        y = np.random.binomial(1, expit(b0 + 1.5 * x1 + 2), n)
        X = np.transpose(np.vstack([const, x1]))

        return X, y

    @pytest.fixture
    def get_test_data_without_constant(self):
        np.random.seed(12345)
        n = 100
        x1 = np.random.normal(size=n)
        b0 = 1
        y = np.random.binomial(1, expit(b0 + 1.5 * x1 + 2), n)
        X = np.transpose(np.vstack([x1]))

        return X, y

    @pytest.mark.parametrize("family", model_families)
    def test_every_family_runs_with_constant(
        self, family: str, get_test_data_with_constant
    ) -> None:
        # Arrange
        X, y = get_test_data_with_constant

        # Act
        model = BayesGLM(endog=y, exog=X, family=family)
        training_result = model.fit()
        result = training_result.predict(X)

        # Assert
        assert len(result) == len(y)

    @pytest.mark.parametrize("family", model_families)
    def test_every_family_runs_without_constant(
        self, family: str, get_test_data_without_constant
    ) -> None:
        # Arrange
        X, y = get_test_data_without_constant

        # Act
        model = BayesGLM(endog=y, exog=X, family=family)
        training_result = model.fit(
            weights=None,
            perform_scale=True,
        )
        result = training_result.predict(X)

        # Assert
        assert len(result) == len(y)
