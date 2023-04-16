# Audio classifier template project

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-lightblue.svg)](https://github.com/meanvoid/nix-flakes)
[![lightning](https://img.shields.io/badge/Powered%20by-Lightning-blueviolet.svg)](https://www.pytorchlightning.ai/index.html)
[![optuna](https://img.shields.io/badge/Powered%20by-Optuna-blue.svg)](https://optuna.org/)


My playground project for Pytorch Lightning, Optuna and other tools

Currently only works with Google's Speech Commands dataset, but I'll add something else later 

## How to run

Edit `n_trials`, `suggest_args` and `experiment_config` in [__main__.py](audio_classifier%2F__main__.py) the way you like

Execute:
```commandline
python -m audio_classifier
```

Results would be saved to `optuna_results.db`, `lightning_logs` and `report_<datetime>.txt`