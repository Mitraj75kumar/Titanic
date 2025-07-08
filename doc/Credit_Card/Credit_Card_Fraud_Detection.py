import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# Configuration
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
np.random.seed(42)

def load_data():
    """Load credit card data from zip file"""
    try:
        zip_path = r'C:\Users\mitra\Downloads\creditcard.csv.zip'
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open('creditcard.csv') as f:
                df = pd.read_csv(f)
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess transaction data"""
    # Normalize 'Amount' feature
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Drop 'Time' as it may not be useful
    df = df.drop(['Time'], axis=1)
    
    return df

def handle_imbalance(X_train, y_train):
    """Handle class imbalance using SMOTE"""
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res

def train_model(X_train, y_train):
    """Train Random Forest classifier"""
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Plot precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    print("\nPreprocessing data...")
    df = preprocess_data(df)
    
    # Check class distribution
    print("\nClass Distribution:")
    print(df['Class'].value_counts())
    
    # Split data
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance
    print("\nBalancing classes...")
    X_train_res, y_train_res = handle_imbalance(X_train, y_train)
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train_res, y_train_res)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Feature importance
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Important Features')
    plt.show()

if __name__ == "__main__":
    main()