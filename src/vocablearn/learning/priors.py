"""Prior confidence calculation using Bayesian updating.

Computes confidence priors for tags based on historical feedback,
using Beta distribution for principled uncertainty quantification.
"""

from typing import Optional

from vocablearn.models import Tag
from vocablearn.storage.sqlite import SQLiteStorage


def calculate_prior(
    confirmed_correct: int,
    confirmed_incorrect: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> tuple[float, float, float]:
    """Calculate prior confidence with credible interval.

    Uses Beta-Binomial conjugate prior for Bayesian updating.

    Args:
        confirmed_correct: Number of correct predictions
        confirmed_incorrect: Number of incorrect predictions
        alpha_prior: Prior alpha (pseudo-successes)
        beta_prior: Prior beta (pseudo-failures)

    Returns:
        Tuple of (prior_mean, lower_bound, upper_bound) for 95% interval
    """
    # Posterior parameters
    alpha = alpha_prior + confirmed_correct
    beta = beta_prior + confirmed_incorrect

    # Posterior mean
    prior_mean = alpha / (alpha + beta)

    # 95% credible interval using Beta quantiles
    try:
        from scipy import stats

        lower = stats.beta.ppf(0.025, alpha, beta)
        upper = stats.beta.ppf(0.975, alpha, beta)
    except ImportError:
        # Fallback: use normal approximation
        import math

        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std = math.sqrt(variance)
        lower = max(0.0, prior_mean - 1.96 * std)
        upper = min(1.0, prior_mean + 1.96 * std)

    return prior_mean, lower, upper


def update_all_priors(
    storage: SQLiteStorage,
    min_samples: int = 5,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> int:
    """Update prior confidence for all tags with sufficient feedback.

    Args:
        storage: Storage backend
        min_samples: Minimum feedback samples required
        alpha_prior: Prior alpha for Beta distribution
        beta_prior: Prior beta for Beta distribution

    Returns:
        Number of tags updated
    """
    updated = 0

    # Get all tags with feedback
    tags = storage.search_vocabulary(min_occurrences=1, limit=100000)

    for tag in tags:
        if tag.sample_size < min_samples:
            continue

        prior, lower, upper = calculate_prior(
            tag.confirmed_correct,
            tag.confirmed_incorrect,
            alpha_prior,
            beta_prior,
        )

        storage.update_tag_prior(tag.tag_id, prior, lower, upper)
        updated += 1

    return updated


def get_effective_prior(
    tag: Tag,
    model_agreement_boost: float = 0.1,
    min_prior: float = 0.1,
    max_prior: float = 0.95,
) -> float:
    """Get effective prior confidence for a tag.

    Combines learned prior with model agreement signal.

    Args:
        tag: Tag object
        model_agreement_boost: Bonus for tags with model agreement
        min_prior: Minimum prior floor
        max_prior: Maximum prior ceiling

    Returns:
        Effective prior confidence
    """
    # Start with learned prior
    prior = tag.prior_confidence

    # Boost for model agreement
    if tag.model_agreement_count > 0 and tag.total_occurrences > 0:
        agreement_rate = tag.model_agreement_count / tag.total_occurrences
        prior = prior + (model_agreement_boost * agreement_rate)

    # Clamp to valid range
    return max(min_prior, min(max_prior, prior))


def estimate_sample_size_needed(
    current_correct: int,
    current_incorrect: int,
    target_interval_width: float = 0.1,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> int:
    """Estimate additional samples needed to achieve target precision.

    Args:
        current_correct: Current correct count
        current_incorrect: Current incorrect count
        target_interval_width: Target 95% CI width
        alpha_prior: Prior alpha
        beta_prior: Prior beta

    Returns:
        Estimated additional samples needed
    """
    # Current posterior parameters
    alpha = alpha_prior + current_correct
    beta = beta_prior + current_incorrect

    # Current interval width
    _, lower, upper = calculate_prior(current_correct, current_incorrect, alpha_prior, beta_prior)
    current_width = upper - lower

    if current_width <= target_interval_width:
        return 0

    # Estimate samples needed (approximate)
    # Width scales roughly as 1/sqrt(n)
    current_n = current_correct + current_incorrect
    if current_n == 0:
        current_n = 1

    # Solve: target_width = current_width * sqrt(current_n / (current_n + additional))
    ratio = (current_width / target_interval_width) ** 2
    target_n = current_n * ratio
    additional = int(target_n - current_n)

    return max(0, additional)
