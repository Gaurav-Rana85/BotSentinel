\# BotSentinel Architecture



BotSentinel uses a machine learning pipeline

to identify potentially malicious network traffic.



\## Main Components



1\. Dataset: Network traffic records.

2\. Training: Model training using train.py.

3\. Model: Saved machine learning model.

4\. API: FastAPI application in app.py.

5\. Dashboard: Web interface for predictions.

6\. Monitoring: Prediction monitoring.



\## Workflow



Network Traffic → Feature Processing →

ML Model → Prediction → API Response

