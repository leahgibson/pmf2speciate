"""
Data preperation for model training.

It transforms profile-based species data into a standardized matrix format where:
- Each row represents a unique chemical profile (identified by PROFILE_CODE)
- Each column represents a specific chemical compound (identified by CAS number)
- Values represent weight percentages and uncertainty percentages for each compound in each profile

Input files (expected in ./data/raw/):
- profiles.pkl: Contains profile metadata and identifiers
- species.pkl: Contains species data linked to profiles via PROFILE_CODE
- species_properties.pkl: Contains mapping between SPECIES_ID and CAS numbers

Output files (saved to ./data/processed/):
1. profile_compounds.pkl: Matrix of weight percentages where rows are profiles and columns are compounds.
   Missing compounds for a profile are filled with 0 (not NaN) to indicate absence.
2. uncertainty.pkl: Corresponding matrix of uncertainty percentages with the same structure.

The script handles profiles with different compound compositions by creating a unified matrix
where missing values are explicitly set to 0 rather than NaN, making it suitable for ML training.
"""

import pandas as pd

profiles = pd.read_pickle("./data/raw/profiles.pkl")
species = pd.read_pickle("./data/raw/species.pkl")
species_properties = pd.read_pickle("./data/raw/species_properties.pkl")

species["UNCERTAINTY_PERCENT"] = species["UNCERTAINTY_PERCENT"].where(
    species["UNCERTAINTY_PERCENT"] >= 0, pd.NA
)

species_grouped = species.groupby("PROFILE_CODE")
species_lookup = dict(zip(species_properties["SPECIES_ID"], species_properties["CAS"]))

profile_codes = profiles["PROFILE_CODE"].to_list()

profile_dfs = []
uncertainty_dfs = []
for code in profile_codes:
    profile = species_grouped.get_group(code)

    # Filter out results where the compound is unknowns
    valid_mask = profile["SPECIES_ID"].map(species_lookup).notna()
    profile = profile[valid_mask]

    weights = dict(
        zip(profile["SPECIES_ID"].map(species_lookup), profile["WEIGHT_PERCENT"])
    )
    weights["code"] = code
    profile_df = pd.DataFrame(weights, index=[0])
    profile_dfs.append(profile_df)

    uncerts = dict(
        zip(profile["SPECIES_ID"].map(species_lookup), profile["UNCERTAINTY_PERCENT"])
    )
    uncerts["code"] = code
    uncertainty_df = pd.DataFrame(uncerts, index=[0])
    uncertainty_dfs.append(uncertainty_df)

all_profiles = pd.concat(profile_dfs, ignore_index=True).fillna(0)
all_profiles.to_parquet(
    "./data/processed/profile_compounds.parquet", compression="snappy"
)

all_uncertainties = pd.concat(uncertainty_dfs, ignore_index=True).fillna(0)
all_uncertainties.to_parquet(
    "./data/processed/uncertainty.parquet", compression="snappy"
)
