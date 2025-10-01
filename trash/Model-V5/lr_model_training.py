# lr_model_training.py

from sklearn.linear_model import LogisticRegression
from evaluation import train_and_evaluate_model # Import the generic evaluator

def train_logistic_regression(X_train, y_train, X_test, y_test, label_encoder, output_dir, random_state, save_artifact_func, cv_strategy):
    """Sets up and trains a Logistic Regression model with an expanded grid."""
    print("\n--- Training Logistic Regression Model ---")
    
    # --- EXPANDED HYPERPARAMETER GRID ---
    lr_param_grid = {
        'model__C': [0.1, 1, 10, 100], 
        'model__solver': ['liblinear', 'saga'],
        'model__penalty': ['l1', 'l2'],
        'model__max_iter': [1000, 2500], 
        'model__random_state': [random_state]
    }
    # Note: We use 'model__' prefix because we'll be using a Pipeline
    
    lr = LogisticRegression()
    
    lr_model, lr_metrics = train_and_evaluate_model(
        model=lr,
        param_grid=lr_param_grid,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model_name="Logistic Regression",
        label_encoder=label_encoder,
        output_dir=output_dir,
        save_artifact_func=save_artifact_func,
        cv_strategy=cv_strategy # Pass the cross-validation strategy
    )
    
    save_artifact_func(lr_model, "03_logistic_regression_model.pkl", output_dir, "Trained LR model")
    return lr_model, lr_metrics