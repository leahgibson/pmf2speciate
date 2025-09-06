import pandas as pd

profiles = pd.read_pickle("./data/raw/profiles.pkl")
species = pd.read_pickle("./data/raw/species.pkl")
species_properties = pd.read_pickle("./data/raw/species_properties.pkl")

species_grouped = species.groupby("PROFILE_CODE")
species_lookup = dict(
    zip(species_properties["SPECIES_ID"], species_properties["SPECIES_NAME"])
)

profile_codes = profiles["PROFILE_CODE"].to_list()

dfs = []
for code in profile_codes:
    profile = species_grouped.get_group(code)

    weights = dict(
        zip(profile["SPECIES_ID"].map(species_lookup), profile["WEIGHT_PERCENT"])
    )
    weights["code"] = code

    df = pd.DataFrame(weights, index=[0])
    dfs.append(df)

all_profiles = pd.concat(dfs, ignore_index=True)

all_profiles.to_pickle("./data/processed/profile_compounds.pkl")
