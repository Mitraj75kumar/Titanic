import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, roc_auc_score)
import zipfile
import os
import warnings

# Configure settings for better display
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data(zip_path):
    """Load and prepare the Titanic dataset"""
    # Load data from zip file
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No CSV file found in the zip archive")
        
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f)
    
    # Standardize column names
    df.columns = df.columns.str.lower()
    col_mapping = {
        'pclass': 'class',
        'sibsp': 'siblings_spouses',
        'parch': 'parents_children',
        'embarked': 'embark_port',
        'embark_town': 'embark_port'
    }
    df.rename(columns=col_mapping, inplace=True)
    
    return df

def clean_data(df):
    """Clean and preprocess the data"""
    # Handle missing values
    if 'age' in df.columns:
        df['age'] = df['age'].fillna(df['age'].median())
    
    if 'embark_port' in df.columns:
        df['embark_port'] = df['embark_port'].fillna(df['embark_port'].mode()[0])
    
    if 'fare' in df.columns:
        df['fare'] = df['fare'].fillna(df['fare'].median())
    
    return df

def engineer_features(df):
    """Create new features"""
    # Family features
    if all(col in df.columns for col in ['siblings_spouses', 'parents_children']):
        df['family_size'] = df['siblings_spouses'] + df['parents_children'] + 1
        df['is_alone'] = (df['family_size'] == 1).astype(int)
    
    # Age bins
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                                labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    
    # Title extraction from name
    if 'name' in df.columns:
        df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_map = {
            'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
            'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
            'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
        }
        df['title'] = df['title'].replace(title_map)
    
    # Drop unnecessary columns
    cols_to_drop = ['passengerid', 'name', 'ticket', 'cabin', 'siblings_spouses', 'parents_children']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    return df

def perform_eda(df):
    """Perform exploratory data analysis"""
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Survival count
    plt.subplot(2, 3, 1)
    sns.countplot(x='survived', data=df)
    plt.title('Survival Count')
    
    # Plot 2: Survival by class
    plt.subplot(2, 3, 2)
    if 'class' in df.columns:
        sns.countplot(x='class', hue='survived', data=df)
        plt.title('Survival by Class')
    
    # Plot 3: Survival by sex
    plt.subplot(2, 3, 3)
    if 'sex' in df.columns:
        sns.countplot(x='sex', hue='survived', data=df)
        plt.title('Survival by Gender')
    
    # Plot 4: Age distribution
    plt.subplot(2, 3, 4)
    if 'age' in df.columns:
        sns.histplot(df['age'].dropna(), bins=30, kde=True)
        plt.title('Age Distribution')
    
    # Plot 5: Fare by survival
    plt.subplot(2, 3, 5)
    if 'fare' in df.columns:
        sns.boxplot(x='survived', y='fare', data=df)
        plt.title('Fare Distribution by Survival')
    
    # Plot 6: Survival by embarkation port
    plt.subplot(2, 3, 6)
    if 'embark_port' in df.columns:
        sns.countplot(x='embark_port', hue='survived', data=df)
        plt.title('Survival by Embarkation Port')
    
    plt.tight_layout()
    plt.savefig('titanic_eda.png')
    plt.show()

def prepare_model_data(df):
    """Prepare data for modeling"""
    X = df.drop('survived', axis=1)
    y = df['survived']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_model_pipeline():
    """Build the machine learning pipeline"""
    # Numeric features
    numeric_features = []
    if 'age' in df.columns:
        numeric_features.append('age')
    if 'fare' in df.columns:
        numeric_features.append('fare')
    if 'family_size' in df.columns:
        numeric_features.append('family_size')
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    # Categorical features
    categorical_features = []
    if 'class' in df.columns:
        categorical_features.append('class')
    if 'sex' in df.columns:
        categorical_features.append('sex')
    if 'embark_port' in df.columns:
        categorical_features.append('embark_port')
    if 'age_group' in df.columns:
        categorical_features.append('age_group')
    if 'title' in df.columns:
        categorical_features.append('title')
    if 'is_alone' in df.columns:
        categorical_features.append('is_alone')
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the model"""
    # Define parameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        build_model_pipeline(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    print("\n=== Best Parameters ===")
    print(grid_search.best_params_)
    
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    return best_model

def plot_feature_importance(model, X):
    """Plot feature importance"""
    # Get feature names
    preprocessor = model.named_steps['preprocessor']
    preprocessor.fit(X)
    
    # Numeric features
    numeric_features = []
    if 'age' in X.columns:
        numeric_features.append('age')
    if 'fare' in X.columns:
        numeric_features.append('fare')
    if 'family_size' in X.columns:
        numeric_features.append('family_size')
    
    # Categorical features
    categorical_features = []
    if 'class' in X.columns:
        categorical_features.append('class')
    if 'sex' in X.columns:
        categorical_features.append('sex')
    if 'embark_port' in X.columns:
        categorical_features.append('embark_port')
    if 'age_group' in X.columns:
        categorical_features.append('age_group')
    if 'title' in X.columns:
        categorical_features.append('title')
    if 'is_alone' in X.columns:
        categorical_features.append('is_alone')
    
    ohe_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(ohe_features)
    
    # Get importances
    importances = model.named_steps['classifier'].feature_importances_
    
    # Create DataFrame
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(15)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title('Top 15 Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

def make_prediction(model, passenger_data):
    """Make a prediction for a single passenger"""
    passenger_df = pd.DataFrame([passenger_data])
    prediction = model.predict(passenger_df)[0]
    probability = model.predict_proba(passenger_df)[0, 1]
    
    print("\n=== Example Prediction ===")
    print("Passenger Details:")
    for k, v in passenger_data.items():
        print(f"{k}: {v}")
    print(f"\nPrediction: {'Survived' if prediction == 1 else 'Did not survive'}")
    print(f"Probability: {probability:.1%}")

if __name__ == "__main__":
    # File path - modify as needed
    zip_path = r'C:\Users\mitra\Downloads\archive (1).zip'
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"File not found at {zip_path}")
    
    try:
        print("⏳ Loading and preparing data...")
        df = load_and_prepare_data(zip_path)
        
        print("🧹 Cleaning data...")
        df = clean_data(df)
        
        print("🛠️ Engineering features...")
        df = engineer_features(df)
        
        print("🔍 Performing EDA...")
        perform_eda(df)
        
        print("🤖 Preparing model data...")
        X_train, X_test, y_train, y_test = prepare_model_data(df)
        
        print("🏋️ Training model...")
        model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        
        print("📊 Plotting feature importance...")
        plot_feature_importance(model, X_train)
        
        print("🔮 Making example prediction...")
        example_passenger = {
            'class': 1,
            'sex': 'female',
            'age': 28,
            'fare': 50,
            'embark_port': 'S',
            'family_size': 2,
            'is_alone': 0,
            'age_group': 'Young Adult',
            'title': 'Mrs'
        }
        make_prediction(model, example_passenger)
        
        print("\n✅ Analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")