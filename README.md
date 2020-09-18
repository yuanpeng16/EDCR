# Efficiently Disentangle Causal Representations

## Install dependency
```buildoutcfg
pip install -r requirements.txt
```

## Main experiments
Causality direction prediction
```buildoutcfg
cd experiments/direction_prediction
python baseline.py
python proposed.py
python plots.py
```

Representation learning
```buildoutcfg
cd experiments/representation_learning
python3 baseline.py
python3 proposed.py
python3 plots.py
```

## Discussion experiments

Other metrics
```buildoutcfg
cd experiments/other_metrics
python kl_divergence.py
python grad_l2_norm.py
```

Robustness
```buildoutcfg
cd experiments/counter_example_discrete
python baseline.py
python proposed.py
```

Temperature and altitude data
```buildoutcfg
cd experiments/altitude_temperature
python altitude_temperature.py
```

Network architecture
```buildoutcfg
cd experiments/altitude_temperature
python altitude_temperature.py --hidden_layers 2
```

Noise
```buildoutcfg
cd experiments/altitude_temperature
python altitude_temperature.py --noise 1.0
```

## Appendix experiments
Causality direction prediction with N=100
```buildoutcfg
cd experiments/direction_prediction/
python baseline.py --N 100
python proposed.py --N 100
python plots_N=100.py
```

Causality direction prediction with continuous variable
```buildoutcfg
cd experiments/direction_prediction_continuous
python baseline.py
python proposed.py
```
