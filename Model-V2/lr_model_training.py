from sklearn.linear_model import LogisticRegression
from evaluation import train_and_evaluate_model # Import the generic evaluator

def train_logistic_regression(X_train, y_train, X_test, y_test, label_encoder, output_dir, random_state, save_artifact_func):
    """Sets up and trains a Logistic Regression model."""
    print("\n--- Training Logistic Regression Model ---")
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10], 
        'solver': ['liblinear', 'lbfgs'], 
        'max_iter': [1000], 
        'random_state': [random_state]
    }
    
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    
    lr_model, lr_metrics = train_and_evaluate_model(
        model=lr,
        param_grid=lr_param_grid,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model_name="Logistic Regression",
        label_encoder=label_encoder,
        output_dir=output_dir,
        save_artifact_func=save_artifact_func
    )
    
    save_artifact_func(lr_model, "03_logistic_regression_model.pkl", output_dir, "Trained LR model")
    return lr_model, lr_metrics