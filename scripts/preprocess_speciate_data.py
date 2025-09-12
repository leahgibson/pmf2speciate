"""
Data preperation for model training.

Transforms profile-based species data into a standardized format where:
- Each row represents a unique chemical profile (identified by PROFILE_CODE) and classified by "Generation Mechanism" and "Source"
- Each column represents a specific chemical compound
- Values represent weight percentages and uncertainty percentages for each compound in each profile

Cleaning includes:
- If a species is UNKNOWN, it's contribution is removed
- If a profile only has one species with 100% contribution, it is removed
- If a profile has a species contributing > 100%, it is removed
- If sum of species is < 95%, profile is removed
- Remove misc category

Input files (expected in ./data/raw/):
- profiles.pkl: Contains profile metadata and identifiers
- species.pkl: Contains species data linked to profiles via PROFILE_CODE
- species_properties.pkl: Contains mapping between SPECIES_ID and name

Output files (saved to ./data/processed/):
1. profile_compounds.pkl: Matrix of weight percentages where rows are profiles and columns are compounds.
   Missing compounds for a profile are filled with 0 (not NaN) to indicate absence.
2. uncertainty.pkl: Corresponding matrix of uncertainty percentages with the same structure.

The script handles profiles with different compound compositions by creating a unified matrix
where missing values are explicitly set to 0 rather than NaN, making it suitable for ML training.
"""

import pandas as pd

from profile_grouper import id_profiles

profiles = pd.read_pickle("./data/raw/profiles.pkl")
species = pd.read_pickle("./data/raw/species.pkl")
species_properties = pd.read_pickle("./data/raw/species_properties.pkl")

species["UNCERTAINTY_PERCENT"] = species["UNCERTAINTY_PERCENT"].where(
    species["UNCERTAINTY_PERCENT"] >= 0, pd.NA
)

species_grouped = species.groupby("PROFILE_CODE")
species_cas_lookup = dict(
    zip(species_properties["SPECIES_ID"], species_properties["CAS"])
)
species_name_lookup = dict(
    zip(species_properties["SPECIES_ID"], species_properties["SPECIES_NAME"])
)

generatioin_mechanism_lookup = dict(
    zip(profiles["PROFILE_CODE"], profiles["CATEGORY_LEVEL_1_Generation_Mechanism"])
)
profile_name_lookup = dict(zip(profiles["PROFILE_CODE"], profiles["PROFILE_NAME"]))
profile_map = id_profiles(
    mechanism_lookup=generatioin_mechanism_lookup,
    profile_name_lookup=profile_name_lookup,
)

profile_total_lookup = dict(zip(profiles["PROFILE_CODE"], profiles["TOTAL"]))

profile_dfs = []
uncertainty_dfs = []
profiles_to_remove = set()  # profiles with 100% single species contribution

for code, categories in profile_map.items():
    profile = species_grouped.get_group(code)

    # Filter out results where the compound is unknowns
    valid_mask = profile["SPECIES_ID"].map(species_cas_lookup).notna()
    profile = profile[valid_mask]

    if (profile["WEIGHT_PERCENT"] >= 100).any():
        profiles_to_remove.add(code)
        continue

    if profile_total_lookup[code] < 90:
        profiles_to_remove.add(code)
        continue

    weights = dict(
        zip(profile["SPECIES_ID"].map(species_name_lookup), profile["WEIGHT_PERCENT"])
    )
    weights["code"] = code
    weights["source"] = categories[0]
    weights["generation_mechanism"] = categories[1]
    profile_df = pd.DataFrame(weights, index=[0])
    profile_dfs.append(profile_df)

    uncerts = dict(
        zip(
            profile["SPECIES_ID"].map(species_name_lookup),
            profile["UNCERTAINTY_PERCENT"],
        )
    )
    uncerts["code"] = code
    uncertainty_df = pd.DataFrame(uncerts, index=[0])
    uncertainty_dfs.append(uncertainty_df)

print(f"Removed {len(profiles_to_remove)} profiles that did not meat criteria.")
print(f"Remaining profiles: {len(profile_dfs)}")

all_profiles = pd.concat(profile_dfs, ignore_index=True).fillna(0)
all_profiles.to_parquet(
    "./src/pmf2speciate/data/profile_compounds.parquet", compression="snappy"
)

all_uncertainties = pd.concat(uncertainty_dfs, ignore_index=True).fillna(0)
all_uncertainties.to_parquet(
    "./data/processed/uncertainty.parquet", compression="snappy"
)
