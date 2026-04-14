# Machine Learning Portfolio

This repository is a collection of hands-on projects in machine learning, deep learning, statistical testing, explainability, recommender system, and LLM application design. The work spans classical supervised learning, neural networks, computer vision, credit-risk modeling, experimentation, and retrieval-augmented chatbot systems.

## Repository Overview

The portfolio includes:

- tabular classification projects with real-world business datasets
- regression and regularization experiments
- neural network implementations in NumPy and PyTorch
- computer vision models on CIFAR-10
- explainability workflows using SHAP and LIME
- model calibration, thresholding, and evaluation for risk-sensitive tasks
- statistical A/B testing
- recommender system with Hit Rate@K ranking evaluation, collaborative fltering, and embedding-based models
- an end-to-end commercial support chatbot with RAG, guardrails, and evaluation

## Featured Projects

| Project | Focus | Main Techniques |
| --- | --- | --- |
| [`Credit Card Fraud.ipynb`](./Credit%20Card%20Fraud.ipynb) | Fraud detection on imbalanced transaction data | Logistic regression, random forest, ROC-AUC, PR-AUC, threshold analysis |
| [`adult_income_uci_gb.ipynb`](./adult_income_uci_gb.ipynb) | Income classification with explainability | Gradient boosting, SHAP, LIME, bootstrapping, preprocessing pipelines |
| [`default_of_credit_card_clients.ipynb`](./default_of_credit_card_clients.ipynb) | Credit default risk modeling | Logistic regression, HistGradientBoosting, calibration, thresholding, explainability |
| [`employee_attrition.ipynb`](./employee_attrition.ipynb) | Employee attrition prediction | PR-AUC model selection, calibration, thresholding, feature importance |
| [`cifar10_cnn.ipynb`](./cifar10_cnn.ipynb) | Image classification | CNNs, data augmentation, transfer learning, ResNet18, PyTorch |
| [`2_layer_nn.ipynb`](./2_layer_nn.ipynb) | Neural network fundamentals | Forward pass, backpropagation, gradient checking, MNIST, SGD vs Adam |
| [`ab_test_marketing.ipynb`](./ab_test_marketing.ipynb) | Marketing experiment analysis | Two-proportion z-test, conversion analysis, hypothesis testing |
| [`commercial_chatbot/`](./commercial_chatbot) | Customer-support chatbot system | RAG, hybrid retrieval, citations, guardrails, intent routing, evaluation |
| [`recommend_system_food_review/`] | Food recommendation system from review interactions | User-based CF, item-based CF, leave-one-out evaluation, Hit Rate@K, neural collaborative filtering, user/item embeddings, PyTorch |

## Project Summaries

### 1. Credit Card Fraud Detection
Built and evaluated fraud detection models on an imbalanced dataset. The notebook compares baseline and tree-based methods, with emphasis on metrics that matter for rare-event detection such as precision-recall performance, ROC curves, threshold tradeoffs, and practical operating points.

### 2. Adult Income Classification
Developed an interpretable income prediction workflow using the Adult dataset. This project combines preprocessing pipelines with boosting models and focuses on model explanation through feature importance, SHAP, LIME, and bootstrap-based uncertainty analysis.

### 3. Default of Credit Card Clients
Created a credit-risk modeling project centered on probability quality as well as ranking quality. In addition to classification performance, the workflow includes calibration analysis, threshold selection, confusion-matrix interpretation, and explainability.

### 4. Employee Attrition Prediction
Built a human-resources risk model for ranking employees by attrition risk. The project compares multiple classifiers, uses PR-AUC for model selection under class imbalance, and explores calibration and decision-threshold strategies for capacity-limited intervention.

### 5. CIFAR-10 CNN and Transfer Learning
Implemented image classification workflows in PyTorch, starting with a custom CNN and extending to transfer learning with ResNet18. The notebook includes normalization, augmentation, training curves, confusion-matrix analysis, and model comparison.

### 6. Two-Layer Neural Network from Scratch
Implemented a two-layer neural network in NumPy to understand the mechanics of forward propagation, backpropagation, and gradient checking, then extended the workflow to a PyTorch MLP on MNIST with optimizer comparison and TensorBoard logging.

### 7. Marketing A/B Test
Analyzed a marketing experiment using a two-proportion z-test to compare conversion rates between groups. The notebook focuses on clean statistical workflow: loading data, validating structure, summarizing treatment and control groups, and testing for significance.

### 8. Commercial Support Chatbot
Built a portfolio-style support chatbot that goes beyond basic chat completion. The system includes retrieval-augmented generation, chunked document retrieval, citation-based answering, refusal logic for weak evidence, routing to tools or escalation workflows, conversation memory, and an evaluation harness.

### 9. Food Review Recommendation System 
Built a recommender-systems project on the Amazon food reviews dataset using a staged workflow: streaming ingestion, exploratory analysis, recommendation-focused EDA, user-item matrix construction, user-based collaborative filtering, item-based collaborative filtering, direct comparison of recommendation lists, leave-one-out offline evaluation with Hit Rate@K, and deep learning recommenders in PyTorch. The deep learning stage uses learned user and item embeddings, then extends the architecture with global mean, user bias, item bias, bounded predictions, and an ablation-style comparison of plain versus bias-enhanced models.


## Skills Demonstrated

- Python
- pandas and NumPy
- scikit-learn pipelines
- supervised learning for classification and regression
- logistic regression and tree-based models
- regularization with ridge and lasso
- gradient boosting
- recommender systems
- collaborative filtering
- ranking evaluation and Hit Rate@K
- neural networks and CNNs
- PyTorch
- embedding-based models
- transfer learning
- TensorBoard / experiment tracking
- model evaluation for imbalanced classification
- calibration and threshold tuning
- SHAP and LIME explainability
- statistical hypothesis testing
- retrieval-augmented generation
- information retrieval / hybrid retrieval
- LLM guardrails, routing, and workflow orchestration

## Repository Structure

Most projects are currently organized as standalone notebooks in the repository root, with the chatbot project implemented as a Python application folder:

```text
.
├── 2_layer_nn.ipynb
├── Credit Card Fraud.ipynb
├── ab_test_marketing.ipynb
├── adult_income_uci_gb.ipynb
├── cifar10_cnn.ipynb
├── commercial_chatbot/
├── default_of_credit_card_clients.ipynb
├── employee_attrition.ipynb
├── recommend_system_food_review/
└── ridge_lasso.ipynb
```

## Notes

- Several notebooks are structured as learning sprints, with day-by-day progress and experiments.
- Some projects emphasize model interpretability and business-facing evaluation rather than only raw accuracy.
- The chatbot project is a systems project and is intentionally different from the notebook-based ML work in the rest of the repository.
