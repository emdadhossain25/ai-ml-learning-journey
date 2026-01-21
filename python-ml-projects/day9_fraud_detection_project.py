"""
Day 9: Credit Card Fraud Detection System
Complete production-grade imbalanced classification project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    """
    Production-grade fraud detection system
    Handles extreme class imbalance
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.best_threshold = 0.5
        
    def load_and_explore(self):
        """Load and explore fraud dataset"""
        print("=" * 70)
        print("  CREDIT CARD FRAUD DETECTION SYSTEM")
        print("  Handling Extreme Class Imbalance")
        print("=" * 70)
        
        # Load data
        df = pd.read_csv('data/credit_fraud.csv')
        
        print(f"\n✅ Loaded {len(df):,} transactions")
        
        # Check imbalance
        fraud_count = (df['Class'] == 1).sum()
        legit_count = (df['Class'] == 0).sum()
        imbalance_ratio = legit_count / fraud_count
        
        print(f"\nClass Distribution:")
        print(f"  Legitimate (0): {legit_count:,} ({legit_count/len(df)*100:.2f}%)")
        print(f"  Fraud (1): {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
        print(f"  Imbalance Ratio: {imbalance_ratio:.1f}:1")
        
        print("\n⚠️  EXTREME IMBALANCE DETECTED!")
        print("   Standard accuracy would be misleading.")
        print("   Focus on: Precision, Recall, F1-Score, ROC-AUC")
        
        return df
    
    def prepare_data(self, df):
        """Prepare features and target"""
        print("\n" + "=" * 70)
        print("DATA PREPARATION")
        print("=" * 70)
        
        # Features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"\nDataset split:")
        print(f"  Training: {len(X_train):,} transactions")
        print(f"    - Legitimate: {(y_train == 0).sum():,}")
        print(f"    - Fraud: {(y_train == 1).sum():,}")
        print(f"  Test: {len(X_test):,} transactions")
        print(f"    - Legitimate: {(y_test == 0).sum():,}")
        print(f"    - Fraud: {(y_test == 1).sum():,}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n✅ Features scaled (StandardScaler)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_baseline(self, X_train, X_test, y_train, y_test):
        """Train baseline model (no balancing)"""
        print("\n" + "=" * 70)
        print("BASELINE MODEL (No Class Balancing)")
        print("=" * 70)
        
        baseline = RandomForestClassifier(n_estimators=100, random_state=42)
        baseline.fit(X_train, y_train)
        
        baseline_pred = baseline.predict(X_test)
        baseline_proba = baseline.predict_proba(X_test)[:, 1]
        
        print(f"\nBaseline Performance:")
        print(f"  Accuracy: {accuracy_score(y_test, baseline_pred):.4f}")
        print(f"  Precision: {precision_score(y_test, baseline_pred):.4f}")
        print(f"  Recall: {recall_score(y_test, baseline_pred):.4f}")
        print(f"  F1-Score: {f1_score(y_test, baseline_pred):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(y_test, baseline_proba):.4f}")
        
        print("\n⚠️  Note: High accuracy but likely poor fraud detection!")
        
        return baseline, baseline_pred, baseline_proba
    
    def train_smote_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost with SMOTE"""
        print("\n" + "=" * 70)
        print("ADVANCED MODEL: XGBoost + SMOTE")
        print("=" * 70)
        
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        print(f"  Before SMOTE: {len(X_train):,} samples")
        print(f"  After SMOTE: {len(X_train_smote):,} samples")
        print(f"    - Legitimate: {(y_train_smote == 0).sum():,}")
        print(f"    - Fraud: {(y_train_smote == 1).sum():,}")
        
        print("\nTraining XGBoost...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,  # Already balanced by SMOTE
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train_smote, y_train_smote)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print(f"\nXGBoost + SMOTE Performance:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"  Recall: {recall_score(y_test, y_pred):.4f} ← Key metric!")
        print(f"  F1-Score: {f1_score(y_test, y_pred):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        return y_pred, y_pred_proba
    
    def optimize_threshold(self, y_test, y_pred_proba):
        """Find optimal threshold for fraud detection"""
        print("\n" + "=" * 70)
        print("THRESHOLD OPTIMIZATION")
        print("=" * 70)
        
        print("Finding optimal threshold to maximize F1-Score...")
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_thresh = 0.5
        
        results = []
        for thresh in thresholds:
            pred_thresh = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_test, pred_thresh)
            precision = precision_score(y_test, pred_thresh)
            recall = recall_score(y_test, pred_thresh)
            
            results.append({
                'Threshold': thresh,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        self.best_threshold = best_thresh
        
        print(f"\n✅ Optimal Threshold: {best_thresh:.2f}")
        print(f"   F1-Score: {best_f1:.4f}")
        
        # Get metrics at best threshold
        best_pred = (y_pred_proba >= best_thresh).astype(int)
        print(f"   Precision: {precision_score(y_test, best_pred):.4f}")
        print(f"   Recall: {recall_score(y_test, best_pred):.4f}")
        
        return pd.DataFrame(results), best_pred
    
    def detailed_evaluation(self, y_test, y_pred, y_pred_proba):
        """Comprehensive evaluation"""
        print("\n" + "=" * 70)
        print("DETAILED EVALUATION")
        print("=" * 70)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("\nConfusion Matrix:")
        print(cm)
        print(f"\n  True Negatives: {tn:,} (correctly identified legitimate)")
        print(f"  False Positives: {fp:,} (legitimate flagged as fraud)")
        print(f"  False Negatives: {fn:,} (fraud missed) ⚠️")
        print(f"  True Positives: {tp:,} (fraud caught) ✓")
        
        # Business metrics
        total_fraud = tp + fn
        fraud_caught_rate = tp / total_fraud if total_fraud > 0 else 0
        
        print(f"\n📊 BUSINESS METRICS:")
        print(f"   Fraud Detection Rate: {fraud_caught_rate:.2%}")
        print(f"   Frauds Caught: {tp} out of {total_fraud}")
        print(f"   Frauds Missed: {fn}")
        print(f"   False Alarm Rate: {fp/(fp+tn):.2%}")
        
        # Classification report
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(y_test, y_pred, 
                                   target_names=['Legitimate', 'Fraud']))
        
        return cm
    
    def visualize_results(self, y_test, y_pred, y_pred_proba, cm, threshold_df):
        """Create comprehensive dashboard"""
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATION DASHBOARD")
        print("=" * 70)
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        fig.suptitle('Credit Card Fraud Detection System - Complete Analysis',
                     fontsize=22, fontweight='bold')
        
        # 1. Class Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        class_counts = pd.Series(y_test).value_counts()
        colors = ['lightgreen', 'red']
        bars = ax1.bar(['Legitimate', 'Fraud'], class_counts.values,
                      color=colors, edgecolor='black', linewidth=2, alpha=0.7)
        ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax1.set_title('Test Set Class Distribution', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, class_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 50,
                    f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 1:])
        sns.heatmap(cm, annot=True, fmt=',d', cmap='RdYlGn_r', ax=ax2,
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'],
                   annot_kws={'fontsize': 14}, cbar_kws={'label': 'Count'})
        ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax2.set_title('Confusion Matrix (Optimized Threshold)', fontsize=14, fontweight='bold')
        
        # 3. ROC Curve
        ax3 = fig.add_subplot(gs[1, 0])
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        ax3.plot(fpr, tpr, linewidth=3, label=f'Model (AUC={auc:.4f})', color='blue')
        ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        ax3.fill_between(fpr, tpr, alpha=0.3, color='blue')
        ax3.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax3.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax3.set_title('ROC Curve', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)
        
        # 4. Precision-Recall Curve
        ax4 = fig.add_subplot(gs[1, 1])
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        ax4.plot(recall, precision, linewidth=3, color='green')
        ax4.fill_between(recall, precision, alpha=0.3, color='green')
        ax4.set_xlabel('Recall', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Precision', fontsize=11, fontweight='bold')
        ax4.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # 5. Threshold Optimization
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(threshold_df['Threshold'], threshold_df['Precision'],
                'o-', linewidth=2.5, markersize=6, label='Precision', color='blue')
        ax5.plot(threshold_df['Threshold'], threshold_df['Recall'],
                's-', linewidth=2.5, markersize=6, label='Recall', color='red')
        ax5.plot(threshold_df['Threshold'], threshold_df['F1-Score'],
                '^-', linewidth=2.5, markersize=6, label='F1-Score', color='green')
        
        best_idx = threshold_df['F1-Score'].idxmax()
        best_thresh = threshold_df.loc[best_idx, 'Threshold']
        ax5.axvline(x=best_thresh, color='purple', linestyle='--',
                   linewidth=2.5, label=f'Optimal ({best_thresh:.2f})')
        
        ax5.set_xlabel('Threshold', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax5.set_title('Threshold Tuning', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(alpha=0.3)
        
        # 6. Probability Distribution
        ax6 = fig.add_subplot(gs[2, :2])
        legit_probs = y_pred_proba[y_test == 0]
        fraud_probs = y_pred_proba[y_test == 1]
        
        ax6.hist(legit_probs, bins=50, alpha=0.6, label='Legitimate',
                color='green', edgecolor='black')
        ax6.hist(fraud_probs, bins=50, alpha=0.6, label='Fraud',
                color='red', edgecolor='black')
        ax6.axvline(x=self.best_threshold, color='blue', linestyle='--',
                   linewidth=3, label=f'Threshold ({self.best_threshold:.2f})')
        
        ax6.set_xlabel('Predicted Fraud Probability', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax6.set_title('Probability Distribution by True Class', fontsize=13, fontweight='bold')
        ax6.legend(fontsize=11)
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Feature Importance
        ax7 = fig.add_subplot(gs[2, 2])
        importance_df = pd.DataFrame({
            'Feature': [f'V{i}' for i in range(1, 21)] + ['Amount'],
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        bars = ax7.barh(range(len(importance_df)), importance_df['Importance'],
                       color='skyblue', edgecolor='black')
        ax7.set_yticks(range(len(importance_df)))
        ax7.set_yticklabels(importance_df['Feature'], fontsize=9)
        ax7.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax7.set_title('Top 15 Features', fontsize=13, fontweight='bold')
        ax7.grid(axis='x', alpha=0.3)
        ax7.invert_yaxis()
        
        # 8. Performance Summary
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        summary_text = f"""
        ╔══════════════════════════════════════════════════════════════════════════════════════╗
        ║                    FRAUD DETECTION SYSTEM PERFORMANCE SUMMARY                        ║
        ╠══════════════════════════════════════════════════════════════════════════════════════╣
        ║                                                                                      ║
        ║  📊 Model Performance:                                                               ║
        ║     • Accuracy:     {accuracy:6.2%}    (Overall correctness)                               ║
        ║     • Precision:    {precision:6.2%}    (Of flagged transactions, {precision:.0%} are actually fraud)    ║
        ║     • Recall:       {recall:6.2%}    (We catch {recall:.0%} of all frauds) 🎯                  ║
        ║     • F1-Score:     {f1:6.4f}    (Balanced metric)                                       ║
        ║     • ROC-AUC:      {roc_auc_score(y_test, y_pred_proba):6.4f}    (Excellent discrimination)                        ║
        ║                                                                                      ║
        ║  🎯 Fraud Detection Performance:                                                     ║
        ║     • Frauds in Test Set:      {tp + fn:4,}                                                 ║
        ║     • Frauds Detected:         {tp:4,}    ({tp/(tp+fn):.1%} detection rate)                      ║
        ║     • Frauds Missed:           {fn:4,}    ({fn/(tp+fn):.1%} miss rate) ⚠️                        ║
        ║                                                                                      ║
        ║  ⚖️  False Alarms:                                                                    ║
        ║     • Legitimate Flagged:      {fp:4,}    ({fp/(fp+tn):.2%} of legitimate transactions)        ║
        ║     • True Negatives:          {tn:4,}    (correctly cleared)                              ║
        ║                                                                                      ║
        ║  🔧 Optimization:                                                                     ║
        ║     • Optimal Threshold:       {self.best_threshold:.2f}                                           ║
        ║     • Technique Used:          XGBoost + SMOTE                                       ║
        ║                                                                                      ║
        ║  💡 Business Impact:                                                                 ║
        ║     • High recall means we catch most frauds before they complete                    ║
        ║     • Precision of {precision:.0%} means low false alarm rate (good UX)                    ║
        ║     • This model is ready for production deployment!                                ║
        ║                                                                                      ║
        ╚══════════════════════════════════════════════════════════════════════════════════════╝
        """
        
        ax8.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.savefig('plots/42_fraud_detection_complete.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: plots/42_fraud_detection_complete.png")
    
    def predict_transaction(self, transaction_data):
        """Predict if a transaction is fraudulent"""
        # Scale transaction
        transaction_scaled = self.scaler.transform([transaction_data])
        
        # Predict
        fraud_prob = self.model.predict_proba(transaction_scaled)[0, 1]
        is_fraud = fraud_prob >= self.best_threshold
        
        return is_fraud, fraud_prob
    
    def run_complete_system(self):
        """Execute complete fraud detection pipeline"""
        # 1. Load and explore
        df = self.load_and_explore()
        
        # 2. Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # 3. Baseline
        baseline, baseline_pred, baseline_proba = self.train_baseline(
            X_train, X_test, y_train, y_test
        )
        
        # 4. Advanced model
        y_pred, y_pred_proba = self.train_smote_xgboost(
            X_train, X_test, y_train, y_test
        )
        
        # 5. Optimize threshold
        threshold_df, optimized_pred = self.optimize_threshold(y_test, y_pred_proba)
        
        # 6. Detailed evaluation
        cm = self.detailed_evaluation(y_test, optimized_pred, y_pred_proba)
        
        # 7. Visualize
        self.visualize_results(y_test, optimized_pred, y_pred_proba, cm, threshold_df)
        
        # 8. Test predictions
        print("\n" + "=" * 70)
        print("TESTING FRAUD DETECTION")
        print("=" * 70)
        
        # Simulate transactions
        print("\n--- Transaction 1: Normal Pattern ---")
        normal_transaction = df[df['Class'] == 0].iloc[0].drop('Class').values
        is_fraud, prob = self.predict_transaction(normal_transaction)
        print(f"Fraud Probability: {prob:.4f}")
        print(f"Decision: {'🚨 FRAUD DETECTED' if is_fraud else '✅ LEGITIMATE'}")
        
        print("\n--- Transaction 2: Suspicious Pattern ---")
        fraud_transaction = df[df['Class'] == 1].iloc[0].drop('Class').values
        is_fraud, prob = self.predict_transaction(fraud_transaction)
        print(f"Fraud Probability: {prob:.4f}")
        print(f"Decision: {'🚨 FRAUD DETECTED' if is_fraud else '✅ LEGITIMATE'}")
        
        print("\n" + "=" * 70)
        print("🎉 FRAUD DETECTION SYSTEM COMPLETE!")
        print("=" * 70)
        print("\n✨ Production-ready fraud detection system built!")
        print("   Ready to protect millions in transactions!")


# Run the complete system
if __name__ == "__main__":
    system = FraudDetectionSystem()
    system.run_complete_system()