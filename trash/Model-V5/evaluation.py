import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC

def train_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test, model_name, label_encoder, output_dir, save_artifact_func, cv_strategy):
    """
    Generic function to train a model within a pipeline (scaler + selector + model) 
    using GridSearchCV and a specified CV strategy, then evaluate it.
    """
    print(f"\n=== Training and Evaluating {model_name} ===")
    
    # --- CREATE A PIPELINE WITH FEATURE SELECTION ---
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif)),
        ('model', model)
    ])
    
    # Add selector parameters to the grid search
    param_grid['selector__k'] = [40, 60, 80, 100, 'all'] # Test different numbers of features
    # -------------------------------------------------

    # Use the provided cross-validation strategy (GroupKFold)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train) # Note: We fit on the unscaled data
    
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

def analyze_feature_importance(model_pipeline, X_test, y_test, feature_names, model_name, output_dir, random_state, save_artifact_func):
    """Fixed feature importance analysis that handles pipelines correctly."""
    print(f"\nAnalyzing feature importance for {model_name}...")
    model_prefix = model_name.lower().replace(' ', '_')
    
    try:
        # Get selected features from the pipeline
        selector = model_pipeline.named_steps['selector']
        selected_mask = selector.get_support()
        selected_features = np.array(feature_names)[selected_mask]
        
        # Get the final model from the pipeline
        final_model = model_pipeline.named_steps['model']
        
        # Use permutation importance for all models - more reliable
        print("Using permutation importance (model-agnostic method)...")
        importance_type = "Permutation-based"
        
        result = permutation_importance(
            model_pipeline, X_test, y_test, 
            n_repeats=10, 
            random_state=random_state, 
            n_jobs=-1
        )
        
        # Create importance dataframe
        feat_imp_df = pd.DataFrame({
            'feature': feature_names, 
            'importance': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance', ascending=False)
        
        save_artifact_func(
            feat_imp_df, 
            f"04_{model_prefix}_feature_importance.csv", 
            output_dir, 
            f"{model_name} feature importance"
        )
        
        # Plot top 20 features
        fig = plt.figure(figsize=(12, 8))
        top_features = feat_imp_df.head(20)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importance ({importance_type}) - {model_name}')
        plt.gca().invert_yaxis()  # Most important at top
        plt.tight_layout()
        
        save_artifact_func(
            fig, 
            f"04_{model_prefix}_feature_importance.png", 
            output_dir
        )
        
        print(f"Top 5 features for {model_name}:")
        for i, row in feat_imp_df.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
            
    except Exception as e:
        print(f"Error in feature importance analysis for {model_name}: {e}")
        import traceback
        traceback.print_exc()