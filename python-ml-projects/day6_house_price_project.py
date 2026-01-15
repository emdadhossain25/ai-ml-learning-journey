"""
Day 6: Complete ML Project - House Price Prediction
End-to-end machine learning workflow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class HousePricePredictor:
    """Complete ML project class"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_data(self, n_samples=200):
        """Generate realistic house price dataset"""
        print("=" * 60)
        print("GENERATING REALISTIC HOUSE DATASET")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Generate features
        data = {
            'size_sqft': np.random.randint(800, 4000, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.uniform(1, 4, n_samples),
            'age_years': np.random.randint(0, 50, n_samples),
            'garage_spaces': np.random.randint(0, 4, n_samples),
            'lot_size': np.random.randint(2000, 15000, n_samples),
            'distance_to_city_km': np.random.uniform(1, 30, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate target with realistic relationships
        df['price'] = (
            250 * df['size_sqft'] +
            40000 * df['bedrooms'] +
            30000 * df['bathrooms'] +
            -1500 * df['age_years'] +
            15000 * df['garage_spaces'] +
            20 * df['lot_size'] +
            -3000 * df['distance_to_city_km'] +
            np.random.randn(n_samples) * 50000  # Noise
        )
        
        # Ensure realistic price range
        df['price'] = df['price'].clip(100000, 1500000)
        
        print(f"✅ Generated {n_samples} houses with 7 features")
        print("\nFirst few houses:")
        print(df.head())
        
        return df
    
    def explore_data(self, df):
        """Exploratory data analysis"""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        print("\nDataset shape:", df.shape)
        print("\nStatistical summary:")
        print(df.describe())
        
        print("\nPrice statistics:")
        print(f"  Mean: ${df['price'].mean():,.0f}")
        print(f"  Median: ${df['price'].median():,.0f}")
        print(f"  Min: ${df['price'].min():,.0f}")
        print(f"  Max: ${df['price'].max():,.0f}")
        
        # Correlations
        print("\nFeature correlations with price:")
        correlations = df.corr()['price'].sort_values(ascending=False)
        print(correlations)
        
        return correlations
    
    def prepare_data(self, df, test_size=0.2):
        """Prepare data for modeling"""
        print("\n" + "=" * 60)
        print("PREPARING DATA")
        print("=" * 60)
        
        # Separate features and target
        X = df.drop('price', axis=1)
        y = df['price']
        
        self.feature_names = X.columns
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Features scaled")
        
        return X_train_scaled, X_test_scaled
    
    def train_model(self, X_train_scaled):
        """Train the model"""
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, self.y_train)
        
        print("✅ Model trained: Linear Regression")
        print("\nLearned coefficients:")
        for feature, coef in zip(self.feature_names, self.model.coef_):
            print(f"  {feature}: {coef:,.0f}")
        print(f"  Intercept: {self.model.intercept_:,.0f}")
        
    def evaluate_model(self, X_train_scaled, X_test_scaled):
        """Comprehensive model evaluation"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Training metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        # Test metrics
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print("TRAINING PERFORMANCE:")
        print(f"  RMSE: ${train_rmse:,.0f}")
        print(f"  MAE:  ${train_mae:,.0f}")
        print(f"  R²:   {train_r2:.4f}")
        
        print("\nTEST PERFORMANCE:")
        print(f"  RMSE: ${test_rmse:,.0f}")
        print(f"  MAE:  ${test_mae:,.0f}")
        print(f"  R²:   {test_r2:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, self.y_train,
                                     cv=5, scoring='r2')
        print(f"\nCross-validation R² (5-fold):")
        print(f"  Mean: {cv_scores.mean():.4f}")
        print(f"  Std:  {cv_scores.std():.4f}")
        
        # Interpretation
        print("\n📊 INTERPRETATION:")
        print(f"  Model explains {test_r2*100:.1f}% of price variance")
        print(f"  Average prediction error: ${test_mae:,.0f}")
        print(f"  Typical error range: ±${test_rmse:,.0f}")
        
        return y_test_pred, test_rmse, test_r2
    
    def visualize_results(self, X_test_scaled, y_test_pred):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('House Price Prediction - Complete Analysis', 
                     fontsize=20, fontweight='bold')
        
        # 1. Actual vs Predicted
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(self.y_test, y_test_pred, alpha=0.6, s=50)
        ax1.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 'r--', linewidth=2)
        ax1.set_xlabel('Actual Price ($)', fontsize=11)
        ax1.set_ylabel('Predicted Price ($)', fontsize=11)
        ax1.set_title('Predictions vs Actual', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # 2. Residuals
        residuals = self.y_test - y_test_pred
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(y_test_pred, residuals, alpha=0.6, s=50)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Price ($)', fontsize=11)
        ax2.set_ylabel('Residuals ($)', fontsize=11)
        ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # 3. Residual distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Residual ($)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Feature importance
        ax4 = fig.add_subplot(gs[1, :])
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        colors = ['green' if x > 0 else 'red' for x in importance['Coefficient']]
        ax4.barh(importance['Feature'], importance['Coefficient'], color=colors, alpha=0.7)
        ax4.set_xlabel('Coefficient Value', fontsize=11)
        ax4.set_title('Feature Impact on Price', fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax4.grid(axis='x', alpha=0.3)
        
        # 5. Prediction errors by price range
        ax5 = fig.add_subplot(gs[2, 0])
        error_pct = np.abs(residuals / self.y_test) * 100
        ax5.scatter(self.y_test, error_pct, alpha=0.6, s=50)
        ax5.set_xlabel('Actual Price ($)', fontsize=11)
        ax5.set_ylabel('Error (%)', fontsize=11)
        ax5.set_title('Prediction Error by Price', fontsize=12, fontweight='bold')
        ax5.grid(alpha=0.3)
        
        # 6. Price distribution
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(self.y_test, bins=30, alpha=0.5, label='Actual', edgecolor='black')
        ax6.hist(y_test_pred, bins=30, alpha=0.5, label='Predicted', edgecolor='black')
        ax6.set_xlabel('Price ($)', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title('Price Distribution', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Model performance metrics
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        metrics_text = f"""
        MODEL PERFORMANCE
        
        Training R²: {r2_score(self.y_train, self.model.predict(self.scaler.transform(self.X_train))):.4f}
        Test R²: {r2_score(self.y_test, y_test_pred):.4f}
        
        Test RMSE: ${np.sqrt(mean_squared_error(self.y_test, y_test_pred)):,.0f}
        Test MAE: ${mean_absolute_error(self.y_test, y_test_pred):,.0f}
        
        Mean Error %: {error_pct.mean():.1f}%
        """
        ax7.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig('plots/26_house_price_complete_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: plots/26_house_price_complete_analysis.png")
    
    def predict_new_houses(self, new_data):
        """Make predictions for new houses"""
        print("\n" + "=" * 60)
        print("PREDICTING NEW HOUSE PRICES")
        print("=" * 60)
        
        print("\nNew houses:")
        print(new_data)
        
        # Scale features
        new_data_scaled = self.scaler.transform(new_data)
        
        # Predict
        predictions = self.model.predict(new_data_scaled)
        
        print("\nPredicted prices:")
        for i, price in enumerate(predictions):
            print(f"  House {i+1}: ${price:,.0f}")
        
        return predictions
    
    def run_complete_project(self):
        """Execute complete ML pipeline"""
        # 1. Generate data
        df = self.generate_data(n_samples=200)
        
        # 2. Explore data
        correlations = self.explore_data(df)
        
        # 3. Prepare data
        X_train_scaled, X_test_scaled = self.prepare_data(df)
        
        # 4. Train model
        self.train_model(X_train_scaled)
        
        # 5. Evaluate model
        y_test_pred, rmse, r2 = self.evaluate_model(X_train_scaled, X_test_scaled)
        
        # 6. Visualize results
        self.visualize_results(X_test_scaled, y_test_pred)
        
        # 7. Test on new houses
        new_houses = pd.DataFrame({
            'size_sqft': [2500, 1800, 3200],
            'bedrooms': [4, 3, 5],
            'bathrooms': [3, 2, 4],
            'age_years': [5, 15, 2],
            'garage_spaces': [2, 1, 3],
            'lot_size': [8000, 5000, 12000],
            'distance_to_city_km': [8, 15, 5]
        })
        predictions = self.predict_new_houses(new_houses)
        
        print("\n" + "=" * 60)
        print("🎉 PROJECT COMPLETE! 🎉")
        print("=" * 60)
        print(f"\nFinal Model Performance:")
        print(f"  R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)")
        print(f"  RMSE: ${rmse:,.0f}")
        print("\n✅ You completed an end-to-end ML project!")


# Run the complete project
if __name__ == "__main__":
    predictor = HousePricePredictor()
    predictor.run_complete_project()