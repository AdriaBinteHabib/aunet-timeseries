# AUNET: Attention-based Unified Network
AUNET is a custom attention-based deep learning architecture for time series forecasting, built using TensorFlow and Keras. Inspired by Transformer-based models, AUNET supports multistep forecasting with lag features and engineered temporal signals.

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
