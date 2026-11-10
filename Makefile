.PHONY: all run analyze figures test data

PYTHON ?= python3

all: run analyze figures

run:
	$(PYTHON) scripts/run_scenarios.py

analyze:
	$(PYTHON) scripts/analyze_results.py

figures:
	$(PYTHON) scripts/make_figures.py

test:
	$(PYTHON) -m pytest tests

# Re-download the public inputs and rebuild the derived series (network access
# and the optional gridstatus/xlrd packages are needed).
data:
	$(PYTHON) data/_raw/fetch_ercot_dam.py
	$(PYTHON) data/_raw/fetch_eia930_erco.py
	$(PYTHON) data/_raw/fetch_henry_hub.py
	$(PYTHON) data/_raw/fetch_open_meteo_houston.py
	$(PYTHON) data/_raw/build_real_carbon_intensity.py
	$(PYTHON) data/_raw/build_performance_curves.py
