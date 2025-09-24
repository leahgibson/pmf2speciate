"""
Builds training and testing datasets for RF model.

Train/test split is applied such that each class is split accoring to the TT amount.

Since each class has a different number of samples, the following apporach is applied to balance classes:
- If the classes are the generation mechanisms, a stratified sampling apporach is used
- If the classes are sources, then each is balanced equally:
    - If the required number of sacmples < size(class) -> generate synthetic samples using uncertainty
    - If the required number of samples > size(class) -> randomly select from class samples
"""

import pickle
import warnings
import pandas as pd
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


np.random.seed(42)

profiles = pd.read_pickle("./data/raw/profiles.pkl")
weights = pd.read_parquet("./src/pmf2speciate/data/profile_compounds.parquet")
uncertainties = pd.read_parquet("./data/processed/uncertainty.parquet")
uncertainties = uncertainties.set_index("code")


mechanism_lookup = dict(zip(weights["code"], weights["generation_mechanism"]))
mechanism_counts = Counter(mechanism_lookup.values())

categorical_columns = ["generation_mechanism", "source", "code"]
species_names = [col for col in weights.columns if col not in categorical_columns]
uncert_names = [col for col in uncertainties.columns if col not in categorical_columns]


# Map species names to CAS numbers
species_properties = pd.read_pickle("./data/raw/species_properties.pkl")
with open("./src/pmf2speciate/data/species_lookup.pkl", "rb") as f:
    cas_map = pickle.load(f)
cas_cols = [cas_map[name] for name in species_names]


mechanisms = ["Combustion", "Ash", "Dust", "Microbial", "Volatilization"]
mech_source_counts = {}
mech_value_counts = {}
for mech in mechanisms:
    value_counts = weights[weights["generation_mechanism"] == mech][
        "source"
    ].value_counts()
    mech_value_counts[mech] = value_counts
    mech_source_counts[mech] = len(value_counts.keys())


def build_datasets(mechanism, min_samples=10):
    """
    Builds training/testing dataset for each generation mechanism category.
    Ensures enough samples per mechanism so that training can use stratified sampling for balance.

    Parameters:
    mechanism: str
        Name of generation mechanism
    min_samples: int
        Minimum number of samples per source class (before applying train/test split) (recommend n=multiple of 5)

    Returns: pd.DataFrame
    """

    n_samples_per_source = {}
    n_sources = mech_source_counts[mechanism]

    n_samples_per_source = calculate_balance(
        n_sources=n_sources, min_samples=min_samples
    )
    n_train = int(n_samples_per_source * 0.8)
    n_test = n_samples_per_source - n_train
    print(f"Training set size: {n_train}. Testing set size: {n_test}.")

    # Preallocate train & test datasets
    total_train_rows = n_train * n_sources
    total_test_rows = n_test * n_sources
    source_names_train, compound_data_train = preallocate(
        n_compounds=len(species_names), n_rows=total_train_rows
    )
    source_names_test, compound_data_test = preallocate(
        n_compounds=len(species_names), n_rows=total_test_rows
    )

    # Sample from df
    samples = weights[weights["generation_mechanism"] == mechanism]
    source_groups = samples.groupby("source")
    train_row_idx = 0
    test_row_idx = 0
    for source, df in source_groups:
        print(f"Collection samples for {source}")
        n_source_samples = len(df)
        # print(f"Number of samples from speciate: {n_source_samples}")

        if n_source_samples >= n_samples_per_source:
            sampled_df = df.sample(
                n=n_samples_per_source, random_state=101
            ).reset_index()

            train_samples = sampled_df[species_names].iloc[:n_train].values
            test_samples = (
                sampled_df[species_names].iloc[n_train : n_train + n_test].values
            )
        else:
            # Split speciate samples 50/50 for generating synthetic samples
            df = df.reset_index()
            half_n = int(len(df) / 2)

            test_set = df.sample(n=half_n, random_state=17)
            train_set = df.drop(test_set.index)

            train_samples = generate_synthetic_samples(train_set, n_train)
            test_samples = generate_synthetic_samples(test_set, n_test)

        # Fill preallocated arrays
        train_end_idx = train_row_idx + n_train
        compound_data_train[train_row_idx:train_end_idx] = train_samples
        source_names_train[train_row_idx:train_end_idx] = source

        train_row_idx = train_end_idx

        test_end_idx = test_row_idx + n_test
        compound_data_test[test_row_idx:test_end_idx] = test_samples
        source_names_test[test_row_idx:test_end_idx] = source

        test_row_idx = test_end_idx

    print("Creating final dataframes")
    combined_samples_train = pd.DataFrame(compound_data_train, columns=cas_cols).fillna(
        0
    )
    combined_samples_train["source"] = source_names_train

    combined_samples_test = pd.DataFrame(compound_data_test, columns=cas_cols).fillna(0)
    combined_samples_test["source"] = source_names_test

    print("Saving files")
    combined_samples_train.to_parquet(
        f"./data/processed/{mechanism}_train.parquet", compression="snappy"
    )
    combined_samples_test.to_parquet(
        f"./data/processed/{mechanism}_test.parquet", compression="snappy"
    )


def calculate_balance(n_sources, min_samples):
    """
    Calculates the number of synthetic samples per profile based on imbalance of generation mechanism classes

    Parameters:
    n_sources: int
        Number of sources per mechanism class
    min_samples: int
        Minimum number of samples per profile ID

    Returns: int
        The number of samples needed per source to balance the classes
    """

    max_count = max(mech_source_counts.values())
    max_size = max_count * min_samples
    class_size = n_sources * min_samples

    scale_factor = np.sqrt(max_size / class_size)
    n_samples = int(scale_factor * min_samples)

    return n_samples


def preallocate(n_compounds, n_rows):
    """
    Make np arrays for preallocation of data

    n_compounds: int
        Number of species/compounds in dataset
    n_rows: number of rows in dataset

    Returns: 2 np arrays
        1 - 1xn_rows of dtype object
        1 - n_compoundsxn_rows of 0's
    """

    source_names = np.empty(n_rows, dtype=object)
    compound_data = np.zeros((n_rows, n_compounds), dtype=np.float32)

    return source_names, compound_data


def generate_synthetic_samples(data, n_samples):
    """
    Generate synthetic samples for a profile using reported uncertainties

    Note: training data will include the samples from SPECIATE while test data will include only synthetic samples

    Parameters:
    data: pd.DataFrame
        Sample data from SPECIATE
    n_samples: int
        Total number of samples needed per source
    type: str
        "train" or "test"

    Returns: np.array
        All necessary samples are rows
    """

    # Check if desired size is small enough to be sampled
    if n_samples < len(data):
        return data.sample(n=n_samples, random_state=1)[species_names].values

    all_samples = []
    n_synthetic_samples = max(n_samples - len(data), 0)

    # print(
    #     f"The desired number of samples is {n_samples}, The number of synthetic samples: {n_synthetic_samples} and there are {len(data)} SPECIATE samples."
    # )

    sampled_df = data.sample(n=n_synthetic_samples, replace=True, random_state=25)
    consolidated_df = sampled_df.copy().value_counts().reset_index(name="count")

    for i, row in consolidated_df.iterrows():
        code = row["code"]
        weight_row = row[species_names]
        weight_row = weight_row.astype(float)

        uncertainty_row = uncertainties[species_names].loc[code]
        has_uncertainties = not (uncertainty_row == 0).all()

        if has_uncertainties:
            samples = np.random.normal(
                loc=weight_row.values,
                scale=uncertainty_row.values,
                size=(row["count"], len(weight_row)),
            )

        else:
            # generate uncertainties using 15% relative error
            absolute_uncertainties = weight_row.values * 0.15

            samples = np.random.normal(
                loc=weight_row.values,
                scale=absolute_uncertainties,
                size=(row["count"], len(weight_row)),
            )

        # If negative samples -> 0
        samples = np.maximum(samples, 0)

        # Renormalize weights to maintail total %
        original_sum = np.sum(weight_row.values)
        row_totals = np.sum(samples, axis=1, keepdims=True)

        mask = row_totals.flatten() > 0
        samples[mask] = samples[mask] * (original_sum / row_totals[mask])

        all_samples.append(samples)

    if all_samples:
        final_samples = np.vstack(all_samples)
        original_data = data[species_names].values
        return np.vstack([final_samples, original_data])
    else:
        return data[species_names].values


for mechanism in mechanisms:
    print(mechanism)
    build_datasets(mechanism, min_samples=50)
