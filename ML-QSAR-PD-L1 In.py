import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from xgboost import plot_importance


def load_data(file_path):
    """Load SMILES strings and activity values from a file.

    Args:
        file_path (str): Path to the input file.

    Returns:
        tuple: Two lists containing SMILES strings and activity values.
    """
    smiles_list = []
    activity_list = []
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue  
                parts = line.split()
                if len(parts) != 2:
                    print(f"Skipping invalid line {line_num}: {line}")
                    continue
                smiles, activity = parts
                smiles_list.append(smiles)
                try:
                    activity_list.append(float(activity))
                except ValueError:
                    print(f"Invalid activity value in line {line_num}: {activity}")
                    continue
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        raise
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

    if len(smiles_list) == 0:
        raise ValueError("No valid data found in the input file.")
    
    return smiles_list, activity_list


def calculate_descriptors(smiles_list):
    """Calculate molecular descriptors for valid SMILES strings.

    Args:
        smiles_list (list): List of SMILES strings.

    Returns:
        tuple: (numpy array of descriptors, list of valid SMILES indices, list of valid SMILES)
    """
    descriptor_list = []
    valid_indices = []
    valid_smiles = []
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                descriptors = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumAliphaticRings(mol),
                    Descriptors.FractionCSP3(mol),
                    Descriptors.HeavyAtomCount(mol),
                    Descriptors.RingCount(mol),
                ]
                descriptor_list.append(descriptors)
                valid_indices.append(idx)
                valid_smiles.append(smiles)
            except Exception as e:
                print(f"Error calculating descriptors for {smiles}: {e}")
        else:
            print(f"Invalid SMILES skipped: {smiles}")

    if len(descriptor_list) == 0:
        raise ValueError("No valid SMILES could be processed.")
    
    return np.array(descriptor_list), valid_indices, valid_smiles


def preprocess_data(descriptors, activities):
    """Create DataFrame and clean data.

    Args:
        descriptors (np.array): Array of molecular descriptors.
        activities (list): List of activity values.

    Returns:
        pd.DataFrame: Cleaned DataFrame with descriptors and activities.
    """
    data = pd.DataFrame(descriptors, columns=[
        'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
        'NumRotatableBonds', 'NumAromaticRings', 'NumAliphaticRings',
        'FractionCSP3', 'HeavyAtomCount', 'RingCount'
    ])
    data['activity'] = activities
    data = data.dropna()
    if data.empty:
        raise ValueError("No valid data remaining after preprocessing.")
    return data


def select_features(data):
    """Separate features and target variable.

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        tuple: Feature matrix (X) and target vector (y).
    """
    X = data.drop('activity', axis=1)
    y = data['activity']
    return X, y


def train_xgboost(X_train, y_train):
    """Train initial XGBoost model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        XGBRegressor: Trained model.
    """
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model




def evaluate_model(model, X_train, y_train, X_test, y_test, kfold_eval=False, n_splits=5):
    """Evaluate model performance with optional K-Fold Cross-Validation.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        kfold_eval (bool): Whether to perform K-Fold evaluation
        n_splits (int): Number of folds for K-Fold
        
    Returns:
        tuple: (y_pred_train, y_pred_test)
    """
    if kfold_eval:
        print("\nK-Fold Cross-Validation Evaluation:")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        r2_scores, mse_scores = [], []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            r2 = r2_score(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            
            r2_scores.append(r2)
            mse_scores.append(mse)
            
            print(f"  Fold {fold + 1}: R² = {r2:.4f}, MSE = {mse:.4f}")
        
        print(f"\n  Mean R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
        print(f"  Mean MSE: {np.mean(mse_scores):.4f} (±{np.std(mse_scores):.4f})")
    
    
    print("\nStandard Evaluation:")
    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    
    y_pred_test = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    print(f"  Train R²: {r2_train:.4f}, Train MSE: {mse_train:.4f}")
    print(f"  Test R²: {r2_test:.4f}, Test MSE: {mse_test:.4f}")
    
    return y_pred_train, y_pred_test


def optimize_and_predict(X, y):
    """Perform hyperparameter optimization with K-Fold and evaluate model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    
    initial_model = train_xgboost(X_train, y_train)
    print("\nInitial Model Evaluation:")
    evaluate_model(initial_model, X_train, y_train, X_test, y_test, kfold_eval=True)
    
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }
    
    grid_search = GridSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,  
        verbose=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f'\nBest Parameters: {grid_search.best_params_}')
    
    print("\nOptimized Model Evaluation:")
    evaluate_model(best_model, X_train, y_train, X_test, y_test, kfold_eval=True)
    y_pred_initial = initial_model.predict(X_test)
    y_pred_optimized = best_model.predict(X_test)
    
    return best_model, X_test, y_test, y_pred_initial, y_pred_optimized
    


def load_new_smiles(file_path):
    """Load new SMILES strings for prediction.

    Args:
        file_path (str): Path to file containing SMILES.

    Returns:
        list: List of SMILES strings.
    """
    smiles_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                smiles = line.strip()
                if smiles:
                    smiles_list.append(smiles)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        raise
    return smiles_list


def predict_new_data(best_model, new_smiles_list):
    """Predict activities for new SMILES strings.

    Args:
        best_model (XGBRegressor): Trained model.
        new_smiles_list (list): List of SMILES strings.

    Returns:
        list: Predicted activities.
    """
    new_descriptors, valid_indices, valid_smiles = calculate_descriptors(new_smiles_list)
    if not valid_smiles:
        print("No valid SMILES found for prediction.")
        return []
    
    predictions = best_model.predict(new_descriptors)
    pred_sd = np.std(predictions)  
    
    print("\nPredictions:")
    for smiles, pred in zip(valid_smiles, predictions):
        print(f"SMILES: {smiles}, Predicted Activity: {pred:.4f}")
    
    print(f"\nStandard Deviation of Predictions: {pred_sd:.4f}")
    
    return predictions


def main(train_file_path, predict_file_path):
    """Main workflow function.

    Args:
        train_file_path (str): Path to training data.
        predict_file_path (str): Path to prediction data.

    Returns:
        tuple: Best model and predictions.
    """
    
    print("Loading training data...")
    smiles_list, activity_list = load_data(train_file_path)
    print(f"Loaded {len(smiles_list)} entries.")
    
    print("\nCalculating descriptors...")
    descriptors, valid_indices, valid_smiles = calculate_descriptors(smiles_list)
    valid_activities = [activity_list[i] for i in valid_indices]
    print(f"Computed descriptors for {len(valid_smiles)} valid molecules.")
    
    print("\nPreprocessing data...")
    data = preprocess_data(descriptors, valid_activities)
    print(f"Preprocessed data shape: {data.shape}")
    
    X, y = select_features(data)
    
    
    print("\nTraining and optimizing model...")
    best_model, X_test, y_test, _, _ = optimize_and_predict(X, y)
    
    
    joblib.dump(best_model, 'best_xgboost_model.pkl')
    print("\nSaved best model to 'best_xgboost_model.pkl'")
    
    
    print("\nLoading new SMILES for prediction...")
    new_smiles_list = load_new_smiles(predict_file_path)
    predictions = predict_new_data(best_model, new_smiles_list)
    
    return best_model, predictions


if __name__ == "__main__":
    train_file_path = 'mol1.txt'
    predict_file_path = 'ligands.txt'
    best_model, predictions = main(train_file_path, predict_file_path)

    
plot_importance(best_model)
plt.show()
