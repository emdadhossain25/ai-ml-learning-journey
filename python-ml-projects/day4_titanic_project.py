"""
Day 4: Mini Project - Titanic Survival Analysis
Complete data analysis from raw data to insights
"""

import pandas as pd
import numpy as np

class TitanicAnalyzer:
    """Complete Titanic dataset analysis"""
    
    def __init__(self, filepath):
        """Load and prepare data"""
        print("=" * 60)
        print("TITANIC SURVIVAL ANALYSIS PROJECT")
        print("=" * 60)
        self.df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(self.df)} passengers")
        
    def explore(self):
        """Initial exploration"""
        print("\n" + "=" * 60)
        print("1. DATA EXPLORATION")
        print("=" * 60)
        
        print("\nDataset shape:", self.df.shape)
        print("\nFirst few rows:")
        print(self.df.head())
        
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
    def clean(self):
        """Clean the data"""
        print("\n" + "=" * 60)
        print("2. DATA CLEANING")
        print("=" * 60)
        
        # Fill missing Age with median
        self.df['Age'].fillna(self.df['Age'].median(), inplace=True)
        
        # Fill missing Embarked with mode
        self.df['Embarked'].fillna(self.df['Embarked'].mode()[0], inplace=True)
        
        # Drop Cabin (too many missing)
        self.df.drop('Cabin', axis=1, inplace=True)
        
        print("✅ Missing Age filled with median")
        print("✅ Missing Embarked filled with mode")
        print("✅ Cabin column dropped")
        print(f"\nRemaining missing values: {self.df.isnull().sum().sum()}")
        
    def analyze_survival(self):
        """Analyze survival patterns"""
        print("\n" + "=" * 60)
        print("3. SURVIVAL ANALYSIS")
        print("=" * 60)
        
        overall_survival = self.df['Survived'].mean()
        print(f"Overall survival rate: {overall_survival:.2%}")
        
        print("\n--- By Gender ---")
        gender_survival = self.df.groupby('Sex')['Survived'].agg(['mean', 'count'])
        print(gender_survival)
        
        print("\n--- By Class ---")
        class_survival = self.df.groupby('Pclass')['Survived'].agg(['mean', 'count'])
        print(class_survival)
        
        print("\n--- By Age Group ---")
        self.df['AgeGroup'] = pd.cut(self.df['Age'], 
                                      bins=[0, 12, 18, 60, 100], 
                                      labels=['Child', 'Teen', 'Adult', 'Senior'])
        age_survival = self.df.groupby('AgeGroup')['Survived'].agg(['mean', 'count'])
        print(age_survival)
        
        print("\n--- By Family Size ---")
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        family_survival = self.df.groupby('FamilySize')['Survived'].mean()
        print(family_survival)
        
    def key_insights(self):
        """Extract key insights"""
        print("\n" + "=" * 60)
        print("4. KEY INSIGHTS")
        print("=" * 60)
        
        # Insight 1: Gender
        female_survival = self.df[self.df['Sex'] == 'female']['Survived'].mean()
        male_survival = self.df[self.df['Sex'] == 'male']['Survived'].mean()
        print(f"1. Women had {female_survival/male_survival:.1f}x higher survival rate than men")
        print(f"   Female: {female_survival:.2%} vs Male: {male_survival:.2%}")
        
        # Insight 2: Class
        first_class = self.df[self.df['Pclass'] == 1]['Survived'].mean()
        third_class = self.df[self.df['Pclass'] == 3]['Survived'].mean()
        print(f"\n2. First class passengers had {first_class/third_class:.1f}x higher survival")
        print(f"   1st Class: {first_class:.2%} vs 3rd Class: {third_class:.2%}")
        
        # Insight 3: Children
        children = self.df[self.df['Age'] < 18]['Survived'].mean()
        adults = self.df[self.df['Age'] >= 18]['Survived'].mean()
        print(f"\n3. Children survival rate: {children:.2%}")
        print(f"   Adults survival rate: {adults:.2%}")
        
        # Insight 4: Family size
        alone = self.df[self.df['FamilySize'] == 1]['Survived'].mean()
        with_family = self.df[self.df['FamilySize'] > 1]['Survived'].mean()
        print(f"\n4. Traveling with family: {with_family:.2%} survival")
        print(f"   Traveling alone: {alone:.2%} survival")
        
        # Insight 5: Fare
        expensive = self.df[self.df['Fare'] > self.df['Fare'].median()]['Survived'].mean()
        cheap = self.df[self.df['Fare'] <= self.df['Fare'].median()]['Survived'].mean()
        print(f"\n5. Higher fare (>${self.df['Fare'].median():.2f}): {expensive:.2%} survival")
        print(f"   Lower fare: {cheap:.2%} survival")
        
    def prepare_for_ml(self):
        """Prepare dataset for machine learning"""
        print("\n" + "=" * 60)
        print("5. PREPARING FOR MACHINE LEARNING")
        print("=" * 60)
        
        # Encode categorical variables
        df_ml = pd.get_dummies(self.df, columns=['Sex', 'Embarked'], drop_first=True)
        
        # Select features
        feature_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 
                          'Sex_male', 'Embarked_Q', 'Embarked_S', 'FamilySize']
        
        X = df_ml[feature_columns]
        y = df_ml['Survived']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"\nFeatures: {feature_columns}")
        
        # Save prepared data
        ml_data = pd.concat([X, y], axis=1)
        ml_data.to_csv('data/titanic_ml_ready.csv', index=False)
        print("\n✅ ML-ready data saved to 'data/titanic_ml_ready.csv'")
        
        return X, y
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        self.explore()
        self.clean()
        self.analyze_survival()
        self.key_insights()
        X, y = self.prepare_for_ml()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE! 🎉")
        print("=" * 60)
        print("\nYou just completed a full data science project!")
        print("Next step: Build a machine learning model to predict survival!")


# Run the analysis
if __name__ == "__main__":
    analyzer = TitanicAnalyzer('data/titanic.csv')
    analyzer.run_full_analysis()