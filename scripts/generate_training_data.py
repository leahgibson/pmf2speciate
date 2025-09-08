"""
Generate synthetic training data from SPECIATE profiles using uncertainties.

This script takes the processed SPECIATE data and created synthetic samples my sampling
from normal distributions based on the uncertainties. For profiles without uncertainties,
assumes 15% relative error.

Original profiles are also included in training data.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 10

profiles = pd.read_pickle("./data/raw/profiles.pkl")
mechanism_lookup = dict(
    zip(profiles["PROFILE_CODE"], profiles["CATEGORY_LEVEL_1_Generation_Mechanism"])
)


def load_data(filename):
    df = pd.read_parquet(f"./data/processed/{filename}.parquet")
    df = df.set_index("code")
    return df


def generate_samples_with_uncertainty(weight_row, uncertainty_row):
    """
    Generate synthetic samples for a profile using reported uncertainties.

    Parameters:
    weight_row : pd.Series
        Weight percentages for compounds
    uncertainty_row : pd.Series
        Absolute uncertainties for compounds

    Returns:
    pd.DataFrame
        DataFrame with synthetic samples as rows
    """

    samples = np.random.normal(
        loc=weight_row.values,
        scale=uncertainty_row.values,
        size=(n_samples, len(weight_row)),
    )

    # If negatives -> 0
    samples = np.maximum(samples, 0)

    # Renormalize weights to maintail total %
    original_sum = np.sum(weight_row.values)
    row_totals = np.sum(samples, axis=1, keepdims=True)

    mask = row_totals.flatten() > 0
    samples[mask] = samples[mask] * (original_sum / row_totals[mask])

    all_samples = np.vstack([weight_row.values, samples])
    synthetic_df = pd.DataFrame(all_samples, columns=weight_row.index)

    return synthetic_df


def generate_samples_with_re(weight_row, relative_error=0.15):
    """
    Generate synthetic samples for a profile using assumed relative error.

    Parameters:
    weight_row : pd.Series
        Weight percentages for compounds
    relative_error : float
        Relative Error as a fraction

    Returns:
    pd.DataFrame
        DataFrame with synthetic samples as rows
    """

    absolute_uncertainties = weight_row.values * relative_error

    samples = np.random.normal(
        loc=weight_row.values,
        scale=absolute_uncertainties,
        size=(n_samples, len(weight_row)),
    )

    # If negatives -> 0
    samples = np.maximum(samples, 0)

    # Renormalize weights to maintail total %
    original_sum = np.sum(weight_row.values)
    row_totals = np.sum(samples, axis=1, keepdims=True)

    mask = row_totals.flatten() > 0
    samples[mask] = samples[mask] * (original_sum / row_totals[mask])

    all_samples = np.vstack([weight_row.values, samples])
    synthetic_df = pd.DataFrame(all_samples, columns=weight_row.index)

    return synthetic_df


# Load the datasets
weights = load_data(filename="profile_compounds")
uncertainties = load_data(filename="uncertainty")


all_samples = []
# Generate synthetic data
for i, (profile_id, weight_row) in enumerate(weights.iterrows()):
    generation_mechanism = mechanism_lookup[profile_id]
    uncertainty_row = uncertainties.loc[profile_id]
    has_uncertainties = not (uncertainty_row == 0).all()

    if has_uncertainties:
        samples = generate_samples_with_uncertainty(weight_row, uncertainty_row)

        # Add in code and generation mechanism
        samples["code"] = [profile_id] * len(samples)
        samples["generation_mechanism"] = [generation_mechanism] * len(samples)

    else:
        # Use 15% relative error
        samples = generate_samples_with_re(weight_row)

        # Add in code and generation mechanism
        samples["code"] = [profile_id] * len(samples)
        samples["generation_mechanism"] = [generation_mechanism] * len(samples)

    all_samples.append(samples)

combined_samples = pd.concat(all_samples, ignore_index=True).fillna(0)

combined_samples.to_parquet(
    "./data/processed/synthetic_samples.parquet", compression="snappy"
)
