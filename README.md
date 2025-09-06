# pmf2speciate

Automatically classify and map Positive Matrix Factorization (PMF) factor profiles from [esat](https://github.com/quanted/esat) to emission sources from [EPA's SPECIATE database](https://www.epa.gov/air-emissions-modeling/speciate) using trained machine learning models.

## Overview

PMF analysis decomposes ambient air pollution data into factor profiles, but identifying the actual emission sources these factors represent requires expert interpretation. This package bridges that gap by using a Random Forest classifier trained on EPA SPECIATE profiles to automatically suggest the most likely source types for your PMF factors.

## Features

- Pre-trained Random Forest model for immediate use
- Compatible with EPA PMF5 esat output formats
- Maps factors to comprehensive EPA SPECIATE source categories
- Retrain models with custom datasets
- Confidence scores and uncertainty quantification

## Quick Start

Updates will come as progress continues!