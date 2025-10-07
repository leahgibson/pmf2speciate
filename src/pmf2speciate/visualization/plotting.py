import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import pickle


def plot_profiles(generation_mechanism):
    """
    Plot SPECIATE profiles for a given generation mechanism with dropdown to select source.

    Parameters:
    generation_mechanism: str
        The generation mechanism to filter profiles by

    Returns:
    plotly.graph_objects.Figure
        Interactive plot with dropdown menu for source selection
    """

    filepath = Path(__file__).parent.parent / "data" / "profile_compounds.parquet"
    df = pd.read_parquet(filepath)

    valid_mechanisms = df["generation_mechanism"].unique()
    if generation_mechanism not in valid_mechanisms:
        raise KeyError(
            f"'{generation_mechanism}' not a valid generation mechanism. Pick from {valid_mechanisms}"
        )

    df = df[df["generation_mechanism"] == generation_mechanism]
    if df.empty:
        raise ValueError(f"No profiles to display for {generation_mechanism}")

    metadata_cols = ["generation_mechanism", "code", "source"]
    species_cols = list(df.columns)
    for md_col in metadata_cols:
        species_cols.remove(md_col)

    # For plotting, only include values with contributions > 1%
    df[species_cols] = df[species_cols].where(df[species_cols] >= 5, 0)

    sources = df["source"].unique()

    # Create figure
    fig = go.Figure()

    for i, source in enumerate(sources):
        source_data = df[df["source"] == source]

        avg_values = source_data[species_cols].replace(0, np.nan).mean().dropna()

        # Bar chat of averages
        fig.add_trace(
            go.Bar(
                x=avg_values.index,
                y=avg_values.values,
                name=f"{source} (Average)",
                visible=(i == 0),
                opacity=0.7,
                showlegend=False,
            )
        )

        # Scatter plot traces for individual profiles
        for _, profile in source_data.iterrows():
            profile_values = profile[species_cols][profile[species_cols] != 0]
            # profile_values = profile[species_cols].replace(0, np.nan).dropna()

            fig.add_trace(
                go.Scatter(
                    x=profile_values.index,
                    y=profile_values.values,
                    mode="markers",
                    name=f"Profile {profile['code']}",
                    visible=(i == 0),
                    opacity=0.6,
                    marker=dict(size=4, color="red"),
                    showlegend=False,
                )
            )

    # Build visibility list
    buttons = []
    for i, source in enumerate(sources):
        visible = []
        for j, s in enumerate(sources):
            s_data = df[df["source"] == s]
            s_n_traces = 1 + len(s_data)

            if j == i:
                visible.extend([True] * s_n_traces)
            else:
                visible.extend([False] * s_n_traces)

        buttons.append(dict(label=source, method="update", args=[{"visible": visible}]))

    # Update layout
    fig.update_layout(
        xaxis=dict(title="Species", tickangle=45, tickmode="linear"),
        yaxis_title="% of Total",
        updatemenus=[
            dict(buttons=buttons, direction="down", showactive=True, x=0.1, y=1.15)
        ],
        # height=600,
        margin=dict(b=150),
        showlegend=True,
    )
    fig.show()


def plot_factor(profile, generation_mechanism, source):
    """
    Plot factor profile compared to source profile.

    Parameters:
    profile: Dict
        CAS numbers as keys and percent weights as values
    generation_mechanism: str
        Name of generation mechanism identified by classification
    source: str
        Name of source identified by classification

    Returns:
        plotly.graph_objects.Figure
    """

    filepath = Path(__file__).parent.parent / "data" / "profile_compounds.parquet"
    df = pd.read_parquet(filepath)

    valid_mechanisms = df["generation_mechanism"].unique()
    if generation_mechanism not in valid_mechanisms:
        raise KeyError(
            f"'{generation_mechanism}' not a valid generation mechanism. Pick from {valid_mechanisms}"
        )

    df = df[df["generation_mechanism"] == generation_mechanism]
    if df.empty:
        raise ValueError(f"No profiles to display for {generation_mechanism}")

    valid_sources = df["source"].unique()
    if source not in valid_sources:
        raise KeyError(
            f"'{source}' not a valid generation mechanism. Pick from {valid_sources}"
        )

    df = df[df["source"] == source]
    df = df.drop(columns=["source", "generation_mechanism", "code"])
    avg_df = df.replace(0, np.nan).mean()
    avg_df = (avg_df / avg_df.sum()) * 100  # renormalize
    avg_profile = avg_df.to_dict()

    filepath = Path(__file__).parent.parent / "data" / "cas_lookup.pkl"
    with open(filepath, "rb") as file:
        cas_map = pickle.load(file)

    # map cas id -> species name
    factor_profile = dict(zip(map(cas_map.get, profile.keys()), profile.values()))

    # Determine which species to include in plot
    species_to_plot = set(factor_profile.keys())
    for species, value in avg_profile.items():  # add any species with weight > 5%
        if value > 5:
            species_to_plot.add(species)

    species_to_plot = sorted(species_to_plot)

    # Create aigned lists for plotting
    factor_values = [factor_profile.get(species, 0) for species in species_to_plot]
    avg_values = [avg_profile.get(species, 0) for species in species_to_plot]

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Factor", x=species_to_plot, y=factor_values, marker_color="steelblue"
        )
    )

    fig.add_trace(
        go.Bar(name=f"{source}", x=species_to_plot, y=avg_values, marker_color="coral")
    )

    fig.update_layout(
        title=f"Profile Comparison: {generation_mechanism} - {source}",
        xaxis_title="Species",
        yaxis_title="Percent Weight (%)",
        barmode="group",
        xaxis_tickangle=-45,
        height=600,
        showlegend=True,
    )

    fig.show()
