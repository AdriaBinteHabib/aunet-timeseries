# AUNET: Attention-based Unified Network
AUNET, introduced in [AUNET (Attention-based Unified Network): Leveraging Attention Based N-BEATS for Enhanced Univariate Time Series Forecasting](https://ieeexplore.ieee.org/document/11016718), is a custom attention-based deep learning architecture for time series forecasting, built using TensorFlow and Keras. Inspired by Transformer-based models, AUNET supports multistep forecasting with lag features and engineered temporal signals.

![AUNET full architecture](https://github.com/user-attachments/assets/89d4c159-050b-4bb6-98c5-3176c6a13bce)

## Features
- Multi-head attention
- Sliding window input
- Rolling multi-step forecasting
- Scikit-learn-like `.fit()` and `.predict()` interface
- Evaluation + Visualization methods included

## Requirements
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from aunet_forecaster import AUNET
model = AUNET(input_length=30, forecast_horizon=7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.plot_predictions(y_test, y_pred, scaler_y=scaler_y)
```

## Evaluation
```python
metrics = model.evaluate(X_test, y_test, scaler_y=scaler_y)
print(metrics)
```

## Learning Curve
```python
model.plot_learning_curve()
```

## Authors
- Adria Binte Habib
- Dr. Golam Rabiul Alam
- Dr. Zia Uddin

## License
MIT

## Citation
If you refer to or apply AUNET in any context, kindly cite the following paper:
```
@ARTICLE{11016718,
  author={Habib, Adria Binte and Alam, Md. Golam Rabiul and Uddin, Md. Zia},
  journal={IEEE Access}, 
  title={AUNET (Attention-based Unified Network): Leveraging Attention Based N-BEATS for Enhanced Univariate Time Series Forecasting}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Forecasting;Time series analysis;Predictive models;Biological system modeling;Finance;Meteorology;Accuracy;Deep learning;Computational modeling;Attention mechanisms;AUNET;Deep Learning;Interpretability;Multi-Head Self-Attention;N-BEATS;Temporal Dependency;Time-Series Forecasting;Univariate Forecasting},
  doi={10.1109/ACCESS.2025.3574459}}
```
