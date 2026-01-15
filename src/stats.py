import numpy as np
from math import sqrt
from scipy import stats


def two_proportion_ztest(x1, n1, x2, n2, alternative="two-sided"):
    """
    Two-proportion z-test.
    x1/n1 vs x2/n2

    alternative:
      - 'two-sided'
      - 'larger'  : p1 > p2
      - 'smaller' : p1 < p2
    """
    p_pool = (x1 + x2) / (n1 + n2)
    se = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = ((x1 / n1) - (x2 / n2)) / se

    if alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "larger":
        p = 1 - stats.norm.cdf(z)
    elif alternative == "smaller":
        p = stats.norm.cdf(z)
    else:
        raise ValueError("alternative must be: two-sided, larger, smaller")

    return z, p


def bootstrap_uplift_binary(control, test, n_boot=5000, seed=0):
    """
    Bootstrap relative uplift distribution for binary outcomes.
    uplift = (p_test - p_control) / p_control
    Returns mean uplift, 95% CI, and the full bootstrap distribution.
    """
    rng = np.random.default_rng(seed)
    control = np.asarray(control)
    test = np.asarray(test)

    uplift = np.empty(n_boot)

    for i in range(n_boot):
        c = rng.choice(control, size=len(control), replace=True)
        t = rng.choice(test, size=len(test), replace=True)
        pc = c.mean()
        pt = t.mean()
        uplift[i] = (pt - pc) / pc if pc > 0 else np.nan

    uplift = uplift[~np.isnan(uplift)]
    ci = np.percentile(uplift, [2.5, 97.5])
    return uplift.mean(), ci, uplift


def mann_whitney(a, b, alternative="two-sided"):
    """Non-parametric test for difference in distributions."""
    return stats.mannwhitneyu(a, b, alternative=alternative)
