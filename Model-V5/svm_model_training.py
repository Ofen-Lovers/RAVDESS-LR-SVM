from sklearn.svm import SVC
from evaluation import train_and_evaluate_model # Import the generic evaluator

def train_svm(X_train, y_train, X_test, y_test, label_encoder, output_dir, random_state, save_artifact_func, cv_strategy):
    """Sets up and trains an SVM model with an expanded grid."""
    print("\n--- Training SVM Model ---")
    
    # --- EXPANDED HYPERPARAMETER GRID ---
    svm_param_grid = {
        'model__C': [1, 10, 50, 100], 
        'model__kernel': ['rbf', 'linear'], 
        'model__gamma': ['scale', 'auto', 0.1, 1],
        'model__probability': [True], 
        'model__random_state': [random_state]
    }
    # Note: We use 'model__' prefix because we'll be using a Pipeline
    
    svm = SVC()
    
    svm_model, svm_metrics = train_and_evaluate_model(
        model=svm,
        param_grid=svm_param_grid,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model_name="SVM",
        label_encoder=label_encoder,
        output_dir=output_dir,
        save_artifact_func=save_artifact_func,
        cv_strategy=cv_strategy # Pass the cross-validation strategy
    )
    
    save_artifact_func(svm_model, "03_svm_model.pkl", output_dir, "Trained SVM model")
    return svm_model, svm_metrics