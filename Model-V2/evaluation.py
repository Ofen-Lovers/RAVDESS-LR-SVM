import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC

def train_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test, model_name, label_encoder, output_dir, save_artifact_func):
    """Generic function to train a model with GridSearchCV and evaluate it."""
    print(f"\n=== Training and Evaluating {model_name} ===")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    print(f"Test Accuracy for {model_name}: {metrics['accuracy']:.4f}")
    
    model_prefix = model_name.lower().replace(' ', '_')
    save_artifact_func(pd.DataFrame(grid_search.cv_results_), f"03_{model_prefix}_cv_results.csv", output_dir, f"{model_name} CV results")
    save_artifact_func(pd.DataFrame([metrics]), f"03_{model_prefix}_metrics.csv", output_dir, f"{model_name} metrics")
    
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    save_artifact_func(fig, f"03_{model_prefix}_confusion_matrix.png", output_dir)
    
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    save_artifact_func(pd.DataFrame(report).transpose(), f"03_{model_prefix}_classification_report.csv", output_dir, f"{model_name} report")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return best_model, metrics

def analyze_feature_importance(model, X_test, y_test, feature_names, model_name, output_dir, random_state, save_artifact_func):
    """Analyzes and plots feature importance for a given model."""
    print(f"\nAnalyzing feature importance for {model_name}...")
    model_prefix = model_name.lower().replace(' ', '_')
    
    if isinstance(model, SVC) and model.kernel == 'linear':
        importance = np.abs(model.coef_).mean(axis=0)
        importance_type = "Coefficient-based"
    else:
        print("Using permutation importance (model-agnostic method)...")
        importance_type = "Permutation-based"
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1)
        importance = result.importances_mean

    feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)
    save_artifact_func(feat_imp_df, f"04_{model_prefix}_feature_importance.csv", output_dir, f"{model_name} feature importance")
    
    fig = plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feat_imp_df.head(20))
    plt.title(f'Top 20 Feature Importance ({importance_type}) - {model_name}')
    plt.tight_layout()
    save_artifact_func(fig, f"04_{model_prefix}_feature_importance.png", output_dir)