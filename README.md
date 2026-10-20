# Machine Learning Portfolio

A collection of projects in machine learning, deep learning, recommender systems, time-series forecasting, and LLM applications. The work combines data exploration, model development, and evaluation, with attention to interpretability and practical decision-making.

The projects below begin with applied systems and advanced modeling, followed by predictive analytics and foundational work. Projects are organized as Jupyter notebooks and modular Python projects. 

## Featured Projects

| Project | Focus | Key Methods |
| --- | --- | --- |
| [Commercial Support Chatbot](./commercial_chatbot/) | Customer-support workflows with grounded responses | RAG, hybrid retrieval, citations, guardrails, intent routing, evaluation, modular Python project |
| [Food Review Recommendation System](./recommend_system_food_review/) | Personalized recommendations from review interactions | Collaborative filtering, user/item embeddings, neural recommenders, PyTorch, Hit Rate@K, modular Python project |
| [CIFAR-10 Image Classification](./cifar10_cnn.ipynb) | Custom CNNs and transfer learning | PyTorch, ResNet18, data augmentation, confusion-matrix analysis |
| [Sales Forecasting](./sales_forecast/) | Sales prediction and daily demand forecasting | XGBoost, lag and rolling-window features, exponential smoothing, Holt-Winters, modular Python project |
| [Employee Attrition Prediction](./employee_attrition.ipynb) | Attrition risk ranking and intervention prioritization | MLP, SVC, XGBoost, LightGBM, RandomizedSearchCV, PR-AUC, calibration |
| [Credit Default Risk Modeling](./default_of_credit_card_clients.ipynb) | Default prediction and probability evaluation | Logistic regression, HistGradientBoosting, calibration, threshold selection |
| [Adult Income Classification](./adult_income_uci_gb.ipynb) | Interpretable income classification | Gradient boosting, preprocessing pipelines, SHAP, LIME, bootstrapping |
| [Credit Card Fraud Detection](./Credit%20Card%20Fraud.ipynb) | Classification under class imbalance | Logistic regression, random forest, PR-AUC, ROC-AUC, threshold analysis |
| [Neural Network from Scratch](./2_layer_nn.ipynb) | Neural network implementation and training | NumPy, backpropagation, gradient checking, PyTorch, SGD, Adam |
| [Marketing A/B Test](./ab_test_marketing.ipynb) | Conversion analysis and hypothesis testing | Two-proportion z-test, treatment/control comparisons |
| [Ridge and Lasso Regression](./ridge_lasso.ipynb) | Regression and regularization | Ridge regression, lasso regression, L1 and L2 regularization |

## Project Summaries

### Commercial Support Chatbot

Built a customer-support chatbot with retrieval-augmented generation and hybrid retrieval to produce responses grounded in source documents. Implemented citation-based answers, guardrails for insufficient evidence, intent routing, conversation memory, and tool or escalation workflows. Added an evaluation harness to assess system behavior.

### Food Review Recommendation System

Developed a recommendation workflow using Amazon food reviews, from streaming ingestion and exploratory analysis to user-item matrix construction. Compared user-based and item-based collaborative filtering using leave-one-out evaluation and Hit Rate@K. Built neural recommenders in PyTorch with learned user and item embeddings, then compared plain and bias-enhanced models through ablation experiments.

### CIFAR-10 Image Classification

Built image classification workflows in PyTorch using a custom CNN and transfer learning with ResNet18. Applied normalization and data augmentation, and compared model behavior through training curves and confusion-matrix analysis.

### Sales Forecasting

Developed workflows for row-level sales prediction and daily total sales forecasting. Compared baseline, linear regression, and XGBoost models, including lag and rolling-window features. Evaluated moving average, simple exponential smoothing, and Holt-Winters methods for daily forecasting using MAE, RMSE, WMAPE, and MAPE.

### Employee Attrition Prediction

Analyzed employee characteristics and employment factors to examine attrition patterns and build preprocessing pipelines. Compared classifiers including SVC, MLP, XGBoost, and LightGBM, tuned hyperparameters with RandomizedSearchCV, and selected MLP based on cross-validation PR-AUC. Evaluated probability calibration, decision thresholds, and ranked predictions for potential retention interventions under limited capacity. Used permutation importance to interpret predictive associations without treating them as causal explanations.

### Credit Default Risk Modeling

Developed a credit default prediction workflow using logistic regression and HistGradientBoosting. Evaluated both risk ranking and probability quality through calibration analysis, threshold selection, confusion-matrix interpretation, and model explainability.

### Adult Income Classification

Built an income classification workflow using preprocessing pipelines and gradient boosting. Examined model predictions through feature importance, SHAP, and LIME, and used bootstrapping to explore uncertainty.

### Credit Card Fraud Detection

Built and compared logistic regression and random forest models on imbalanced transaction data. Evaluated fraud detection using precision-recall performance and ROC-AUC, and examined decision thresholds to understand the tradeoff between detecting fraud and generating false alerts.

### Neural Network from Scratch

Implemented a two-layer neural network in NumPy, including forward propagation, backpropagation, and gradient checking. Extended the workflow with a PyTorch MLP for MNIST, compared SGD and Adam, and tracked training with TensorBoard.

### Marketing A/B Test

Analyzed a marketing experiment by validating the data, summarizing treatment and control groups, and comparing conversion rates. Used a two-proportion z-test to assess statistical evidence of a difference between groups.

### Ridge and Lasso Regression

Explored ridge and lasso regression to examine how regularization influences regression models. Compared L1 and L2 penalties and their effects on model coefficients.


## Technical Skills

| Area | Tools and Methods |
| --- | --- |
| Programming and data analysis | Python, pandas, NumPy |
| Machine learning | scikit-learn, preprocessing pipelines, logistic regression, SVC, random forest, XGBoost, LightGBM, ridge and lasso |
| Deep learning | PyTorch, MLPs, CNNs, transfer learning, embeddings, TensorBoard |
| Model evaluation | Cross-validation, hyperparameter tuning, PR-AUC, ROC-AUC, calibration, threshold selection |
| Interpretability | Permutation importance, SHAP, LIME |
| Recommender systems | Collaborative filtering, neural recommenders, leave-one-out evaluation, Hit Rate@K |
| Forecasting | Lag features, rolling-window features, exponential smoothing, Holt-Winters |
| Statistical analysis | A/B testing, hypothesis testing, bootstrapping |
| LLM applications | RAG, hybrid retrieval, citations, guardrails, intent routing, evaluation |

## Repository Guide

Projects are organized as standalone notebooks and project folders. Use the links above to explore each project.

```text
.
├── commercial_chatbot/
├── recommend_system_food_review/
├── sales_forecast/
├── cifar10_cnn.ipynb
├── employee_attrition.ipynb
├── default_of_credit_card_clients.ipynb
├── adult_income_uci_gb.ipynb
├── Credit Card Fraud.ipynb
├── 2_layer_nn.ipynb
├── ab_test_marketing.ipynb
└── ridge_lasso.ipynb
```

Several notebooks retain their original learning-sprint structure to document the progression of experiments.
