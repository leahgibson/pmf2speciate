"""
Helper code for mapping different labels for compounds (because SPECIES ID -> CAS is not 1 to 1)
"""

import pandas as pd
from collections import Counter
import pickle

species_properties = pd.read_pickle("./data/raw/species_properties.pkl")

# make 1 to 1 mapping between CAS and SPECIES_NAME

cas_list = species_properties["CAS"].to_list()
duplicates = [item for item, count in Counter(cas_list).items() if count > 1]

cas_map = {}
species_to_cas_map = {}
for i, row in species_properties.iterrows():
    cas_num = row["CAS"]
    id = row["SPECIES_ID"]
    species_name = str(row["SPECIES_NAME"])
    if cas_num in duplicates:
        if "-duplicate" in species_name:
            continue
    cas_map[cas_num] = species_name
    species_to_cas_map[species_name] = cas_num

species_id_map = {}
for i, row in species_properties.iterrows():
    species_id = row["SPECIES_ID"]
    cas_num = row["CAS"]
    species_name = cas_map[cas_num]
    species_id_map[species_id] = species_name


with open("./src/pmf2speciate/data/cas_lookup.pkl", "wb") as f:
    pickle.dump(cas_map, f)

with open("./src/pmf2speciate/data/id_lookup.pkl", "wb") as f:
    pickle.dump(species_id_map, f)

with open("./src/pmf2speciate/data/species_lookup.pkl", "wb") as f:
    pickle.dump(species_to_cas_map, f)
