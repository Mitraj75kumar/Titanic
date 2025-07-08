import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import chardet 

# Configuration
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

def detect_encoding(filepath):
    """Detect file encoding"""
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read(10000))  # Read first 10k bytes to guess encoding
    return result['encoding']

def load_movie_data(zip_path):
    """Load movie data from zip file with encoding detection"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Get first CSV file in zip
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV file found in zip archive")
            
            # Extract to temporary file
            temp_path = 'temp_movie_data.csv'
            with z.open(csv_files[0]) as zf, open(temp_path, 'wb') as f:
                f.write(zf.read())
            
            # Detect encoding
            try:
                encoding = detect_encoding(temp_path)
                df = pd.read_csv(temp_path, encoding=encoding)
            except UnicodeDecodeError:
                # Try fallback encodings if detection fails
                for enc in ['latin1', 'iso-8859-1', 'windows-1252']:
                    try:
                        df = pd.read_csv(temp_path, encoding=enc)
                        break
                    except:
                        continue
                else:
                    raise ValueError("Failed to decode file with common encodings")
            
            # Clean up
            os.remove(temp_path)
            return df
            
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"Error loading data: {str(e)}")

def preprocess_data(df):
    """Clean and prepare movie data"""
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Handle missing values
    text_cols = ['director', 'actors', 'genre', 'title']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
    
    # Ensure we have rating column
    rating_cols = ['rating', 'average_rating', 'imdb_rating', 'score']
    for col in rating_cols:
        if col in df.columns:
            df['rating'] = pd.to_numeric(df[col], errors='coerce')
            break
    else:
        raise ValueError("No rating column found")
    
    # Drop rows with missing ratings
    df = df.dropna(subset=['rating'])
    
    # Feature engineering
    if 'actors' in df.columns:
        df['actor_count'] = df['actors'].apply(lambda x: len(str(x).split(',')))
    if 'genre' in df.columns:
        df['genre_count'] = df['genre'].apply(lambda x: len(str(x).split(',')))
    
    return df

def build_model():
    """Build the ML pipeline"""
    text_features = []
    if 'director' in df.columns:
        text_features.append('director')
    if 'actors' in df.columns:
        text_features.append('actors')
    if 'genre' in df.columns:
        text_features.append('genre')
    
    numeric_features = []
    if 'actor_count' in df.columns:
        numeric_features.append('actor_count')
    if 'genre_count' in df.columns:
        numeric_features.append('genre_count')
    
    transformers = []
    if text_features:
        transformers.append(('text', TfidfVectorizer(), text_features[0]))
        if len(text_features) > 1:
            for i, feat in enumerate(text_features[1:], 2):
                transformers.append((f'text{i}', TfidfVectorizer(), feat))
    
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    
    preprocessor = ColumnTransformer(transformers)
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

def main():
    try:
        # Load data
        print("Loading data...")
        filepath = r'C:\Users\mitra\Downloads\Moving rating Prediction.zip'
        movies = load_movie_data(filepath)
        
        # Show available columns
        print("\nAvailable columns:", movies.columns.tolist())
        
        # Preprocess
        print("Preprocessing data...")
        global df  # Make available for model building
        df = preprocess_data(movies)
        
        # Prepare features
        features = []
        if 'director' in df.columns:
            features.append('director')
        if 'actors' in df.columns:
            features.append('actors')
        if 'genre' in df.columns:
            features.append('genre')
        if 'actor_count' in df.columns:
            features.append('actor_count')
        if 'genre_count' in df.columns:
            features.append('genre_count')
        
        if not features:
            raise ValueError("No usable features found in dataset")
        
        # Train-test split
        X = df[features]
        y = df['rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build and train model
        print("Training model...")
        model = build_model()
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        print(f"\nModel Performance:")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
        print(f"R²: {r2_score(y_test, predictions):.2f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=predictions)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Movie Rating Predictions')
        plt.show()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'df' in globals():
            print("\nSample of processed data:")
            print(df.head())

if __name__ == "__main__":
    # Install chardet if not available
    try:
        import chardet # type: ignore
    except ImportError:
        print("Installing chardet for encoding detection...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"]) # type: ignore
        import chardet # type: ignore
    
    main()