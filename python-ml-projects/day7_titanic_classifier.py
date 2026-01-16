"""
Day 7: Titanic Survival Prediction
Complete binary classification project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                            roc_curve, roc_auc_score, precision_recall_curve)

class TitanicSurvivalPredictor:
    """Complete Titanic classification project"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_and_prepare_data(self):
        """Load and prepare Titanic dataset"""
        print("=" * 60)
        print("TITANIC SURVIVAL PREDICTION")
        print("=" * 60)
        
        # Load data
        df = pd.read_csv('data/titanic.csv')
        print(f"\n✅ Loaded {len(df)} passengers")
        
        # Clean data
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        
        # Feature engineering
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Simplify titles
        df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                           'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                           'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # Age groups
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100],
                                labels=[0, 1, 2, 3])
        
        # Fare groups
        df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3])
        
        print("\n✅ Data cleaned and engineered")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def select_features(self, df):
        """Select and encode features"""
        print("\n" + "=" * 60)
        print("FEATURE SELECTION")
        print("=" * 60)
        
        # Select features
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
                   'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup']
        
        # Create feature dataframe
        X = df[features].copy()
        
        # Encode categorical variables
        X = pd.get_dummies(X, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
        
        # Target
        y = df['Survived']
        
        self.feature_names = X.columns.tolist()
        
        print(f"Features selected: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names}")
        
        return X, y
    
    def split_and_scale(self, X, y):
        """Split data and scale features"""
        print("\n" + "=" * 60)
        print("TRAIN-TEST SPLIT")
        print("=" * 60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} passengers")
        print(f"Test set: {len(X_test)} passengers")
        print(f"\nClass distribution in training:")
        print(y_train.value_counts())
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n✅ Features scaled")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train_scaled, y_train):
        """Train logistic regression model"""
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        print("✅ Model trained: Logistic Regression")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        return feature_importance
    
    def evaluate_model(self, X_train_scaled, X_test_scaled, y_train, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Probabilities
        y_test_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"\nACCURACY:")
        print(f"  Training: {train_acc*100:.2f}%")
        print(f"  Test: {test_acc*100:.2f}%")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"\nCONFUSION MATRIX:")
        print(cm)
        
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  True Negatives (correctly predicted died): {tn}")
        print(f"  False Positives (predicted survived, actually died): {fp}")
        print(f"  False Negatives (predicted died, actually survived): {fn}")
        print(f"  True Positives (correctly predicted survived): {tp}")
        
        # Precision, Recall, F1
        print(f"\nDETAILED METRICS:")
        print(classification_report(y_test, y_test_pred,
                                   target_names=['Died', 'Survived']))
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train,
                                     cv=5, scoring='accuracy')
        print(f"\nCross-Validation Accuracy:")
        print(f"  Mean: {cv_scores.mean():.4f}")
        print(f"  Std: {cv_scores.std():.4f}")
        
        return y_test_pred, y_test_pred_proba, cm, roc_auc
    
    def visualize_results(self, X_test_scaled, y_test, y_test_pred, 
                         y_test_pred_proba, cm, feature_importance):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Titanic Survival Prediction - Classification Analysis',
                     fontsize=20, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Died', 'Survived'],
                   yticklabels=['Died', 'Survived'],
                   annot_kws={'fontsize': 14})
        ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        
        ax2.plot(fpr, tpr, linewidth=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        # 3. Precision-Recall Curve
        ax3 = fig.add_subplot(gs[0, 2])
        precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
        
        ax3.plot(recall, precision, linewidth=3, color='green')
        ax3.set_xlabel('Recall', fontsize=12)
        ax3.set_ylabel('Precision', fontsize=12)
        ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. Feature Importance (Top 10)
        ax4 = fig.add_subplot(gs[1, :])
        top_features = feature_importance.head(10)
        colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
        ax4.barh(top_features['Feature'], top_features['Coefficient'], 
                color=colors, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Coefficient (Impact on Survival)', fontsize=12, fontweight='bold')
        ax4.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax4.grid(axis='x', alpha=0.3)
        
        # 5. Probability Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        died_probs = y_test_pred_proba[y_test == 0]
        survived_probs = y_test_pred_proba[y_test == 1]
        
        ax5.hist(died_probs, bins=20, alpha=0.6, label='Actually Died',
                color='red', edgecolor='black')
        ax5.hist(survived_probs, bins=20, alpha=0.6, label='Actually Survived',
                color='green', edgecolor='black')
        ax5.axvline(x=0.5, color='blue', linestyle='--', linewidth=2,
                   label='Threshold (0.5)')
        ax5.set_xlabel('Predicted Probability of Survival', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.set_title('Probability Distribution', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Prediction Accuracy by Probability
        ax6 = fig.add_subplot(gs[2, 1])
        prob_bins = np.linspace(0, 1, 11)
        bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
        bin_accuracy = []
        
        for i in range(len(prob_bins)-1):
            mask = (y_test_pred_proba >= prob_bins[i]) & (y_test_pred_proba < prob_bins[i+1])
            if mask.sum() > 0:
                acc = accuracy_score(y_test[mask], y_test_pred[mask])
                bin_accuracy.append(acc)
            else:
                bin_accuracy.append(0)
        
        ax6.bar(bin_centers, bin_accuracy, width=0.08, alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Predicted Probability', fontsize=12)
        ax6.set_ylabel('Accuracy', fontsize=12)
        ax6.set_title('Accuracy by Confidence Level', fontsize=14, fontweight='bold')
        ax6.set_ylim(0, 1)
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Model Performance Summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        summary_text = f"""
        MODEL PERFORMANCE SUMMARY
        
        Accuracy: {accuracy:.1%}
        Precision: {precision:.1%}
        Recall: {recall:.1%}
        F1-Score: {f1:.3f}
        ROC-AUC: {roc_auc_score(y_test, y_test_pred_proba):.3f}
        
        True Positives: {tp}
        True Negatives: {tn}
        False Positives: {fp}
        False Negatives: {fn}
        
        Test Set Size: {len(y_test)}
        """
        
        ax7.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.savefig('plots/30_titanic_classification_complete.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: plots/30_titanic_classification_complete.png")
    
    def predict_survival(self, passenger_data):
        """Predict survival for new passengers"""
        print("\n" + "=" * 60)
        print("PREDICTING NEW PASSENGERS")
        print("=" * 60)
        
        print("\nPassenger data:")
        print(passenger_data)
        
        # Scale features
        passenger_scaled = self.scaler.transform(passenger_data)
        
        # Predict
        predictions = self.model.predict(passenger_scaled)
        probabilities = self.model.predict_proba(passenger_scaled)[:, 1]
        
        print("\nPredictions:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = "SURVIVED" if pred == 1 else "DIED"
            print(f"  Passenger {i+1}: {result} (survival probability: {prob:.1%})")
        
        return predictions, probabilities
    
    def run_complete_project(self):
        """Execute complete Titanic classification pipeline"""
        # 1. Load and prepare data
        df = self.load_and_prepare_data()
        
        # 2. Select features
        X, y = self.select_features(df)
        
        # 3. Split and scale
        X_train_scaled, X_test_scaled, y_train, y_test = self.split_and_scale(X, y)
        
        # 4. Train model
        feature_importance = self.train_model(X_train_scaled, y_train)
        
        # 5. Evaluate model
        y_test_pred, y_test_pred_proba, cm, roc_auc = self.evaluate_model(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # 6. Visualize results
        self.visualize_results(X_test_scaled, y_test, y_test_pred,
                              y_test_pred_proba, cm, feature_importance)
        
        print("\n" + "=" * 60)
        print("🎉 TITANIC CLASSIFICATION COMPLETE! 🎉")
        print("=" * 60)
        print(f"\nFinal Test Accuracy: {accuracy_score(y_test, y_test_pred)*100:.2f}%")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print("\n✅ You successfully predicted Titanic survival!")


# Run the complete project
if __name__ == "__main__":
    predictor = TitanicSurvivalPredictor()
    predictor.run_complete_project()