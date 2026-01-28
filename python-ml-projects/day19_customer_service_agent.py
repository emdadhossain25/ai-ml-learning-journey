"""
Day 19: Smart Customer Service Agent
Uses NLP sentiment analysis + rule-based logic
"""

import joblib
import pandas as pd
from datetime import datetime
import re

print("=" * 60)
print("SMART CUSTOMER SERVICE AGENT")
print("=" * 60)

# ============================================
# LOAD SENTIMENT MODEL (from Day 18)
# ============================================

print("\n1. LOADING SENTIMENT ANALYZER")
print("-" * 60)

try:
    sentiment_model = joblib.load('models/sentiment_analyzer.pkl')
    sentiment_vectorizer = joblib.load('models/sentiment_vectorizer.pkl')
    print("✅ Sentiment model loaded successfully!")
except:
    print("⚠️  Sentiment model not found. Using rule-based fallback.")
    sentiment_model = None
    sentiment_vectorizer = None

# ============================================
# AGENT CLASS
# ============================================

class CustomerServiceAgent:
    """Intelligent customer service agent"""
    
    def __init__(self, sentiment_model=None, vectorizer=None):
        self.sentiment_model = sentiment_model
        self.vectorizer = vectorizer
        
        # Response templates
        self.responses = {
            'positive': [
                "Thank you for your kind words! We're glad you're satisfied.",
                "We appreciate your feedback! Is there anything else we can help with?",
                "Wonderful to hear! We're here if you need anything else."
            ],
            'negative': [
                "We sincerely apologize for the inconvenience. Let me connect you with a specialist.",
                "I understand your frustration. A senior agent will assist you shortly.",
                "We're sorry to hear that. Your concern is our priority."
            ],
            'neutral': [
                "Thank you for reaching out. How can I assist you today?",
                "I'm here to help. Could you provide more details?",
                "Thanks for contacting us. What can I do for you?"
            ]
        }
        
        # Urgency keywords
        self.urgency_keywords = [
            'urgent', 'immediately', 'asap', 'emergency', 'critical',
            'broken', 'not working', 'failed', 'error', 'help',
            'stuck', 'problem', 'issue', 'bug'
        ]
        
        # Product/service categories
        self.categories = {
            'billing': ['payment', 'charge', 'bill', 'invoice', 'refund', 'price'],
            'technical': ['error', 'bug', 'crash', 'not working', 'broken', 'slow'],
            'account': ['login', 'password', 'access', 'account', 'register'],
            'shipping': ['delivery', 'shipping', 'tracking', 'package', 'arrived'],
            'general': ['question', 'how to', 'help', 'support', 'information']
        }
    
    def analyze_sentiment(self, message):
        """Analyze message sentiment"""
        if self.sentiment_model and self.vectorizer:
            # Use ML model
            message_vectorized = self.vectorizer.transform([message])
            sentiment = self.sentiment_model.predict(message_vectorized)[0]
            confidence = self.sentiment_model.predict_proba(message_vectorized).max()
        else:
            # Fallback: Simple rule-based
            message_lower = message.lower()
            positive_words = ['great', 'excellent', 'love', 'perfect', 'amazing', 'thank']
            negative_words = ['terrible', 'awful', 'hate', 'worst', 'bad', 'disappointed']
            
            pos_count = sum(word in message_lower for word in positive_words)
            neg_count = sum(word in message_lower for word in negative_words)
            
            if pos_count > neg_count:
                sentiment = 'positive'
                confidence = 0.7
            elif neg_count > pos_count:
                sentiment = 'negative'
                confidence = 0.7
            else:
                sentiment = 'neutral'
                confidence = 0.6
        
        return sentiment, confidence
    
    def detect_urgency(self, message):
        """Detect if message is urgent"""
        message_lower = message.lower()
        urgency_score = sum(keyword in message_lower for keyword in self.urgency_keywords)
        
        # Check for capital letters (shouting = urgent)
        if message.isupper():
            urgency_score += 2
        
        # Check for exclamation marks
        urgency_score += message.count('!')
        
        is_urgent = urgency_score >= 2
        return is_urgent, urgency_score
    
    def categorize_message(self, message):
        """Categorize message by topic"""
        message_lower = message.lower()
        
        category_scores = {}
        for category, keywords in self.categories.items():
            score = sum(keyword in message_lower for keyword in keywords)
            category_scores[category] = score
        
        if max(category_scores.values()) == 0:
            return 'general'
        
        return max(category_scores, key=category_scores.get)
    
    def generate_response(self, sentiment):
        """Generate appropriate response"""
        import random
        return random.choice(self.responses[sentiment])
    
    def recommend_action(self, sentiment, is_urgent, category):
        """Recommend action for human agent"""
        if sentiment == 'negative' and is_urgent:
            return "🔴 ESCALATE TO SENIOR AGENT IMMEDIATELY"
        elif sentiment == 'negative':
            return "🟡 ESCALATE TO HUMAN AGENT"
        elif is_urgent:
            return "🟡 RESPOND WITHIN 1 HOUR"
        elif sentiment == 'positive':
            return "🟢 AUTO-RESPOND OK / LOG FOR ANALYTICS"
        else:
            return "🟢 STANDARD RESPONSE"
    
    def process_message(self, customer_id, message):
        """Process customer message - main method"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Analyze
        sentiment, confidence = self.analyze_sentiment(message)
        is_urgent, urgency_score = self.detect_urgency(message)
        category = self.categorize_message(message)
        
        # Generate response
        response = self.generate_response(sentiment)
        action = self.recommend_action(sentiment, is_urgent, category)
        
        # Compile result
        result = {
            'timestamp': timestamp,
            'customer_id': customer_id,
            'message': message,
            'sentiment': sentiment,
            'confidence': f"{confidence:.1%}",
            'is_urgent': is_urgent,
            'urgency_score': urgency_score,
            'category': category,
            'suggested_response': response,
            'recommended_action': action
        }
        
        return result

# ============================================
# TEST THE AGENT
# ============================================

print("\n" + "=" * 60)
print("2. TESTING CUSTOMER SERVICE AGENT")
print("=" * 60)

# Initialize agent
agent = CustomerServiceAgent(sentiment_model, sentiment_vectorizer)

# Test messages
test_messages = [
    {
        'customer_id': 'CUST001',
        'message': "Your product is amazing! Best purchase I've made this year. Thank you!"
    },
    {
        'customer_id': 'CUST002',
        'message': "URGENT! My payment failed and I need this resolved IMMEDIATELY!"
    },
    {
        'customer_id': 'CUST003',
        'message': "The app keeps crashing. This is the worst experience ever."
    },
    {
        'customer_id': 'CUST004',
        'message': "Hello, can you help me track my order? Order #12345"
    },
    {
        'customer_id': 'CUST005',
        'message': "How do I reset my password? I can't login to my account."
    },
    {
        'customer_id': 'CUST006',
        'message': "TERRIBLE service! I want a full refund NOW! This is unacceptable!"
    },
    {
        'customer_id': 'CUST007',
        'message': "Thanks for the quick delivery. Product works great!"
    },
    {
        'customer_id': 'CUST008',
        'message': "My bill shows incorrect charges. Can someone review this?"
    }
]

print("\nProcessing customer messages...\n")

results = []
for msg_data in test_messages:
    result = agent.process_message(msg_data['customer_id'], msg_data['message'])
    results.append(result)
    
    print("=" * 60)
    print(f"Customer ID: {result['customer_id']}")
    print(f"Time: {result['timestamp']}")
    print(f"\nMessage: {result['message']}")
    print(f"\n📊 ANALYSIS:")
    print(f"   Sentiment: {result['sentiment'].upper()} (Confidence: {result['confidence']})")
    print(f"   Urgent: {'YES' if result['is_urgent'] else 'NO'} (Score: {result['urgency_score']})")
    print(f"   Category: {result['category'].upper()}")
    print(f"\n💬 SUGGESTED RESPONSE:")
    print(f"   {result['suggested_response']}")
    print(f"\n🎯 RECOMMENDED ACTION:")
    print(f"   {result['recommended_action']}")
    print()

# ============================================
# ANALYTICS DASHBOARD
# ============================================

print("=" * 60)
print("3. ANALYTICS DASHBOARD")
print("=" * 60)

df = pd.DataFrame(results)

print("\nSENTIMENT DISTRIBUTION:")
sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   {sentiment.capitalize()}: {count} ({percentage:.1f}%)")

print("\nURGENT MESSAGES:")
urgent_count = df['is_urgent'].sum()
print(f"   Total: {urgent_count} out of {len(df)} ({(urgent_count/len(df))*100:.1f}%)")

print("\nCATEGORY BREAKDOWN:")
category_counts = df['category'].value_counts()
for category, count in category_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   {category.capitalize()}: {count} ({percentage:.1f}%)")

print("\nACTION PRIORITIES:")
print(f"   🔴 Escalate Immediately: {df['recommended_action'].str.contains('IMMEDIATELY').sum()}")
print(f"   🟡 Human Agent Needed: {df['recommended_action'].str.contains('HUMAN AGENT').sum()}")
print(f"   🟢 Auto-respond OK: {df['recommended_action'].str.contains('AUTO-RESPOND').sum()}")

# ============================================
# SAVE RESULTS
# ============================================

print("\n" + "=" * 60)
print("4. SAVING RESULTS")
print("=" * 60)

df.to_csv('customer_service_log.csv', index=False)
print("✅ Saved: customer_service_log.csv")

# ============================================
# BUSINESS VALUE
# ============================================

print("\n" + "=" * 60)
print("5. BUSINESS VALUE CALCULATION")
print("=" * 60)

print("""
AUTOMATION METRICS:

Scenario: E-commerce customer service
- Receives: 1,000 messages/day
- Human agents: 10 agents @ $500/month = $5,000/month
- Average handling: 5 minutes/message

WITH AI AGENT:
- Auto-responds to positive: 30% → 300 messages
- Flags urgent/negative: 25% → 250 messages (human priority)
- Standard responses: 45% → 450 messages (templated)

RESULTS:
- Human agents focus on 250 critical cases (not 1,000!)
- Response time: Instant (not 5-30 minutes)
- Customer satisfaction: ↑ 40% (faster resolution)
- Cost savings: Reduce from 10 to 4 agents = $3,000/month saved
- Annual savings: $36,000

ROI: Massive! Agent pays for itself in weeks.
""")

# ============================================
# DEPLOYMENT GUIDE
# ============================================

print("=" * 60)
print("6. DEPLOYMENT OPTIONS")
print("=" * 60)

print("""
OPTION 1: REST API (Flask)
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
agent = CustomerServiceAgent()

@app.route('/analyze', methods=['POST'])
def analyze_message():
    data = request.json
    result = agent.process_message(
        data['customer_id'], 
        data['message']
    )
    return jsonify(result)
```

OPTION 2: Integration with existing system
- Hook into Zendesk, Intercom, Freshdesk
- Analyze incoming tickets automatically
- Tag and route based on sentiment

OPTION 3: Chatbot widget
- Embed on website
- Real-time customer interaction
- Escalate to human when needed

OPTION 4: Email processing
- Monitor support@company.com
- Auto-categorize incoming emails
- Generate draft responses
""")

print("\n" + "=" * 60)
print("🤖 CUSTOMER SERVICE AGENT COMPLETE!")
print("=" * 60)

print("""
WHAT YOU BUILT:
✅ Sentiment analysis integration
✅ Urgency detection
✅ Message categorization
✅ Auto-response generation
✅ Action recommendations
✅ Analytics dashboard

SKILLS DEMONSTRATED:
✅ NLP application
✅ Rule-based logic
✅ ML model integration
✅ Business value analysis
✅ Production deployment thinking

NEXT STEPS:
1. Add more response templates
2. Connect to real messaging platform
3. Build REST API wrapper
4. Create dashboard for human agents
5. Add learning from human feedback

THIS is the kind of project that impresses employers! 🎯
""")
