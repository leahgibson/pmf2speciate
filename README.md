# pmf2speciate

Automatically classify and map Positive Matrix Factorization (PMF) factor profiles from [esat](https://github.com/quanted/esat) to emission sources from [EPA's SPECIATE database](https://www.epa.gov/air-emissions-modeling/speciate) using trained machine learning models.

## Overview

PMF analysis decomposes ambient air pollution data into factor profiles, but identifying the actual emission sources these factors represent requires expert interpretation. This package bridges that gap by using a hierarchical Random Forest classifier trained on EPA SPECIATE profiles to automatically suggest the most likely source types for your PMF factors.

## Features

- Hierarchical classification system that first identifies generation mechanism (Combustion, Microbial, Volatilization), then specific source within that mechanism
- Pre-trained Random Forest model for immediate use
- Single function call to identify sources from factor profiles
- Confidence scoring metrics with each classification level
- Compatible with EPA PMF5 esat output formats
- Visualization of dominant species in each source

## Installation

Since this package is not yet on PyPI, you can install it directly from the GitHub repository. This is the simplest way for a user to get started with the package and its dependencies.

```
pip install git+https://github.com/leahgibson/pmf2speciate.git
```

## Quick Start

### Basic Usage - Source Identification

Users provide the percent weights of a factor profile, identifying each compound by its [CAS registry number](https://www.cas.org/cas-data/cas-registry).

 ```python
from pmf2speciate import SourceClassifier

factor_profile = {
    "71-43-2": 15.2,
    "108-88-3": 8.7,
    "100-41-4": 5.1,
    "1330-20-7": 12.3,
    # ... more species
}

classifier = SourceClassifier()
result = classifier.identify_source(factor_profile)

print("Classification Result:")
print(f"Generation Mechanism: {result['generation_mechanism']} (confidence: {result['generation_confidence']:.3f})")
if result["specific_source"]:
    print(f"Specific Source: {result['specific_source']} (confidence: {result['source_confidence']:.3f})")
    print(f"Overall Confidence: {result['overall_confidence']:.3f}")
 ```

 ### Basic Usage - Source Visualization

 It is possible to easily visualize the source profiles categorized by generation mechanism.

 ```python
from pmf2speciate import plot_profiles

plot_profiles("Combustion")
plot_profiles("Microbial")
plot_profiles("Volatilization")
 ```

 Once a source has been identified, the user's factor profile can be visualized against the average profile for that source (or any generation mechanism and source they specify).
 ```python
 from pmf2speciate import plot_factor

 plot_factor(factor_profile, result["generation_mechanism"], result["specific_source"])
 ```

## Model Architecture

The classification uses a two-level hierarchy:

1. Generation Mechanism Model: classifies profiles into broad categories:
- Combustion
- Microbial
- Volatilization

2. Source-Specific Models: For each generation mechanism, a specialized model identifies specific sources (e.g., "Vehicle Exhaust", "Biomass Burning", "Landfill", etc.)

This hierarchical approach improves accuracy by first narrowing down the general type of emission source, then applying specialized knowledge for fine-grained classification.

To view a summary of the classifications within both levels:
```python
from pmf2speciate import SourceClassifier

classifier = SourceClassifier()
print(classifier.get_model_info())
```

## Contributions

Contributions are welcome! Here is some important information to help you get started with the development environment.

### Project Structure

This project uses a standard `src/` layout. The Python package code is located inside the `src/pmf2speciate` directory, while the project's metadata and other files are at the root.

It is important to be aware of this structure, especially when running tests or scripts. To avoid import issues, all development commands should be executed from the root directory of the project (the directory containing `pyproject.toml`).

### Setting Up a Development Environment

1. Clone the repository
2. Create and activate a virtual environment
3. Install `requirements.txt`
4. Install the package in editable mode: `pip install -e .`

### Running Tests
`pytest` is used for all testing. When running tests, the `ModuleNotFoundError` is a common issue for projects with this structure due to module shadowing. To prevent this, there is a `conftest.py` file in the `tests/` director that correctly configures the Python path.

To run all tests
```
pytest
```