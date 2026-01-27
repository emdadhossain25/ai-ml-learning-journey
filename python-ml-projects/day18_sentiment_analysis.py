"""
Day 18: Sentiment Analysis with NLP
Analyzing text sentiment - practical NLP introduction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DAY 18: SENTIMENT ANALYSIS - NLP BASICS")
print("=" * 60)

# Sample dataset (we'll create synthetic reviews)
print("\n1. CREATING SAMPLE REVIEW DATA")
print("-" * 60)

# Positive reviews
positive_reviews = [
    "This product is amazing! Highly recommended.",
    "Excellent quality and fast delivery. Very satisfied.",
    "Love it! Worth every penny. Will buy again.",
    "Outstanding service. The best purchase I've made.",
    "Fantastic product. Exceeded my expectations.",
    "Great quality! Very happy with this purchase.",
    "Absolutely perfect! Couldn't ask for more.",
    "Wonderful experience. Five stars all the way.",
    "Incredible value for money. Highly recommended.",
    "Best product ever! So glad I bought this.",
] * 10  # 100 positive reviews

# Negative reviews
negative_reviews = [
    "Terrible quality. Very disappointed with this purchase.",
    "Waste of money. Would not recommend to anyone.",
    "Poor service and bad product. Avoid at all costs.",
    "Horrible experience. Never buying from here again.",
    "Complete garbage. Broke after one day of use.",
    "Worst purchase ever. Total waste of money.",
    "Very unhappy. Quality is terrible and delivery was late.",
    "Disappointing product. Not worth the price at all.",
    "Bad quality and poor customer service. Avoid.",
    "Awful product. Requesting a refund immediately.",
] * 10  # 100 negative reviews

# Neutral reviews
neutral_reviews = [
    "It's okay. Nothing special but does the job.",
    "Average product. Met my basic expectations.",
    "Standard quality. Neither good nor bad.",
    "Acceptable. Would consider other options next time.",
    "Fair enough. Gets the work done.",
    "Reasonable product for the price.",
    "It works as described. Nothing more, nothing less.",
    "Decent product. Could be better, could be worse.",
    "Just fine. No complaints but not impressed either.",
    "Satisfactory. Does what it's supposed to do.",
] * 10  # 100 neutral reviews

# Combine all reviews
reviews = positive_reviews + negative_reviews + neutral_reviews
sentiments = ['positive'] * len(positive_reviews) + \
             ['negative'] * len(negative_reviews) + \
             ['neutral'] * len(neutral_reviews)

# Create DataFrame
df = pd.DataFrame({
    'review': reviews,
    'sentiment': sentiments
})

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Created {len(df)} reviews")
print(f"   Positive: {(df['sentiment'] == 'positive').sum()}")
print(f"   Negative: {(df['sentiment'] == 'negative').sum()}")
print(f"   Neutral: {(df['sentiment'] == 'neutral').sum()}")

print("\nSample reviews:")
print(df.head(10))

# ============================================
# TEXT PREPROCESSING & VECTORIZATION
# ============================================

print("\n" + "=" * 60)
print("2. TEXT PREPROCESSING")
print("=" * 60)

# TF-IDF Vectorization
print("Converting text to numerical features using TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=500,  # Top 500 words
    stop_words='english',  # Remove common words
    ngram_range=(1, 2)  # Unigrams and bigrams
)

X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

print(f"✅ Vectorized {len(df)} reviews")
print(f"   Feature matrix shape: {X.shape}")
print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")

# Top features
feature_names = vectorizer.get_feature_names_out()
print(f"\nTop 10 features: {list(feature_names[:10])}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Train-test split:")
print(f"   Training: {len(X_train)} reviews")
print(f"   Test: {len(X_test)} reviews")

# ============================================
# MODEL TRAINING
# ============================================

print("\n" + "=" * 60)
print("3. TRAINING MULTIPLE MODELS")
print("=" * 60)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy
    })
    
    print(f"  Accuracy: {accuracy:.4f}")

results_df = pd.DataFrame(results)
print(f"\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {results_df['Accuracy'].max():.4f}")

# ============================================
# DETAILED EVALUATION
# ============================================

print("\n" + "=" * 60)
print("4. DETAILED EVALUATION")
print("=" * 60)

y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative', 'neutral'])
print("\nConfusion Matrix:")
print(cm)

# ============================================
# REAL-TIME PREDICTION
# ============================================

print("\n" + "=" * 60)
print("5. TESTING WITH NEW REVIEWS")
print("=" * 60)

new_reviews = [
    "This is absolutely fantastic! Best purchase ever!",
    "Terrible product. Complete waste of money.",
    "It's okay. Nothing special.",
    "Amazing quality! Highly recommend to everyone!",
    "Very disappointed. Poor quality and late delivery.",
]

print("Predicting sentiment for new reviews:\n")

new_reviews_vectorized = vectorizer.transform(new_reviews)
predictions = best_model.predict(new_reviews_vectorized)

for review, sentiment in zip(new_reviews, predictions):
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment.upper()}")
    print()

# ============================================
# BUSINESS APPLICATION
# ============================================

print("=" * 60)
print("6. BUSINESS APPLICATIONS")
print("=" * 60)

print("""
SENTIMENT ANALYSIS USE CASES:

1. CUSTOMER FEEDBACK ANALYSIS
   • Analyze product reviews automatically
   • Identify unhappy customers for retention
   • Track sentiment trends over time

2. SOCIAL MEDIA MONITORING
   • Monitor brand mentions on Twitter/Facebook
   • Detect PR crises early (negative sentiment spike)
   • Measure campaign effectiveness

3. EMPLOYEE FEEDBACK
   • Analyze employee survey responses
   • Identify workplace issues
   • Measure employee satisfaction trends

4. CHATBOT RESPONSES
   • Detect customer frustration in real-time
   • Route angry customers to human agents
   • Improve response quality

5. MARKET RESEARCH
   • Analyze competitor reviews
   • Understand customer pain points
   • Guide product development
""")

# ============================================
# SAVE MODEL
# ============================================

print("\n" + "=" * 60)
print("7. SAVING MODEL")
print("=" * 60)

import joblib

joblib.dump(best_model, 'models/sentiment_analyzer.pkl')
joblib.dump(vectorizer, 'models/sentiment_vectorizer.pkl')

print("✅ Model saved: models/sentiment_analyzer.pkl")
print("✅ Vectorizer saved: models/sentiment_vectorizer.pkl")

# ============================================
# VISUALIZATION
# ============================================

print("\n" + "=" * 60)
print("8. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Sentiment Analysis - NLP Project', fontsize=18, fontweight='bold')

# Plot 1: Sentiment Distribution
ax1 = axes[0, 0]
sentiment_counts = df['sentiment'].value_counts()
ax1.bar(sentiment_counts.index, sentiment_counts.values, 
        color=['lightgreen', 'lightcoral', 'lightblue'], edgecolor='black')
ax1.set_xlabel('Sentiment', fontsize=11, fontweight='bold')
ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
ax1.set_title('Sentiment Distribution', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(sentiment_counts.values):
    ax1.text(i, v + 2, str(v), ha='center', fontweight='bold')

# Plot 2: Model Comparison
ax2 = axes[0, 1]
colors_models = ['lightblue', 'lightgreen', 'lightcoral']
bars = ax2.bar(results_df['Model'], results_df['Accuracy'], 
              color=colors_models, edgecolor='black', linewidth=2)
ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, results_df['Accuracy']):
    ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.02,
            f'{acc:.3f}', ha='center', fontweight='bold')

# Plot 3: Confusion Matrix Heatmap
ax3 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
           xticklabels=['Positive', 'Negative', 'Neutral'],
           yticklabels=['Positive', 'Negative', 'Neutral'],
           cbar_kws={'label': 'Count'})
ax3.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax3.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax3.set_title(f'Confusion Matrix - {best_model_name}', fontsize=13, fontweight='bold')

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
╔═══════════════════════════════════════════╗
║       SENTIMENT ANALYSIS PROJECT          ║
╠═══════════════════════════════════════════╣
║                                           ║
║  DATASET:                                 ║
║    Total Reviews: {len(df):,}                     ║
║    Positive: {(df['sentiment'] == 'positive').sum()}                           ║
║    Negative: {(df['sentiment'] == 'negative').sum()}                           ║
║    Neutral: {(df['sentiment'] == 'neutral').sum()}                            ║
║                                           ║
║  BEST MODEL: {best_model_name:20s}    ║
║    Accuracy: {results_df['Accuracy'].max():.2%}                       ║
║                                           ║
║  FEATURES:                                ║
║    TF-IDF Vectorization                   ║
║    Vocabulary Size: {len(vectorizer.vocabulary_):,}                  ║
║    Max Features: 500                      ║
║                                           ║
║  BUSINESS VALUE:                          ║
║    • Automated review analysis            ║
║    • Real-time sentiment detection        ║
║    • Customer satisfaction tracking       ║
║    • Brand reputation monitoring          ║
║                                           ║
║  NEXT STEPS:                              ║
║    • Deploy as REST API                   ║
║    • Integrate with chatbot               ║
║    • Scale to larger datasets             ║
║    • Add emotion detection (angry/happy)  ║
║                                           ║
╚═══════════════════════════════════════════╝
"""

ax4.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()
plt.savefig('plots/58_sentiment_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/58_sentiment_analysis.png")

print("\n" + "=" * 60)
print("DAY 18: SENTIMENT ANALYSIS COMPLETE!")
print("=" * 60)

print(f"""
PROJECT SUMMARY:
  • Created dataset: 300 reviews (100 each sentiment)
  • Trained 3 models: Naive Bayes, Logistic Regression, Random Forest
  • Best accuracy: {results_df['Accuracy'].max():.2%}
  • Model saved for deployment
  
SKILLS GAINED:
  • NLP fundamentals
  • Text preprocessing
  • TF-IDF vectorization
  • Multi-class classification
  • Real-world text analysis

BUSINESS APPLICATIONS:
  • Customer review analysis
  • Social media monitoring
  • Brand reputation management
  • Chatbot sentiment detection

NEXT: Deploy as API or integrate with real data!
""")