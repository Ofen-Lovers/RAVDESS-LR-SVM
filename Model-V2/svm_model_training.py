from sklearn.svm import SVC
from evaluation import train_and_evaluate_model # Import the generic evaluator

def train_svm(X_train, y_train, X_test, y_test, label_encoder, output_dir, random_state, save_artifact_func):
    """Sets up and trains an SVM model."""
    print("\n--- Training SVM Model ---")
    svm_param_grid = {
        'C': [0.1, 1, 10], 
        'kernel': ['linear', 'rbf'], 
        'gamma': ['scale', 'auto'], 
        'probability': [True], 
        'random_state': [random_state]
    }
    
    svm = SVC(probability=True, class_weight='balanced') 
    
    svm_model, svm_metrics = train_and_evaluate_model(
        model=svm,
        param_grid=svm_param_grid,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model_name="SVM",
        label_encoder=label_encoder,
        output_dir=output_dir,
        save_artifact_func=save_artifact_func
    )
    
    save_artifact_func(svm_model, "03_svm_model.pkl", output_dir, "Trained SVM model")
    return svm_model, svm_metrics