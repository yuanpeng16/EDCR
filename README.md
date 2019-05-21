# Efficiently and Robustly Disentangle Causality

This repository is extended from https://github.com/authors-1901-10912/A-Meta-Transfer-Objective-For-Learning-To-Disentangle-Causal-Mechanisms.

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Main Experiments
Causality direction prediction
```
cd comparison_experiments/direction_prediction
python baseline.py
python proposed.py
python plots.py
```

Representation learning
```
cd comparison_experiments/representation_learning
python3 baseline.py
python3 proposed.py
python3 plots.py
```

Robustness
```
cd comparison_experiments/counter_example_discrete
python baseline.py
python proposed.py
```

## Appendix experiments
Causality direction prediction with N=100
```
cd comparison_experiments/direction_prediction/
python baseline.py --N 100
python proposed.py --N 100
python plots_N=100.py
```

Causality direction prediction with continuous variable
```
cd comparison_experiments/direction_prediction_continuous
python baseline.py
python proposed.py
```

Other metrics
```
cd comparison_experiments/other_metrics
python kl_divergence.py
python grad_l2_norm.py
```
