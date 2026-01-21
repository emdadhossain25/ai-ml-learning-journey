"""
Day 10: Test Flask API
Test the deployed ML model API
"""

import requests
import json
import time

API_URL = 'http://localhost:5000'

print("=" * 60)
print("TESTING ML MODEL API")
print("=" * 60)

# Wait for server to start
print("\n⏳ Waiting for server to start...")
time.sleep(2)

# ============================================
# TEST 1: HOME ENDPOINT
# ============================================

print("\n" + "=" * 60)
print("TEST 1: Home Endpoint (GET /)")
print("=" * 60)

try:
    response = requests.get(f'{API_URL}/')
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Make sure the Flask server is running!")
    print("   Run: python3 day10_flask_api.py")
    exit(1)

# ============================================
# TEST 2: HEALTH CHECK
# ============================================

print("\n" + "=" * 60)
print("TEST 2: Health Check (GET /health)")
print("=" * 60)

response = requests.get(f'{API_URL}/health')
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")

if response.json()['status'] != 'healthy':
    print("❌ API is not healthy!")
    exit(1)

print("✅ API is healthy!")

# ============================================
# TEST 3: SINGLE PREDICTION
# ============================================

print("\n" + "=" * 60)
print("TEST 3: Single Prediction (POST /predict)")
print("=" * 60)

# Test passenger 1: First class woman
passenger1 = {
    'Pclass': 1,
    'Name': 'Miss. Jane Smith',
    'Sex': 'female',
    'Age': 25,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 100,
    'Embarked': 'C',
    'Cabin': 'C85',
    'Ticket': '12345'
}

print("\nPassenger 1 (First class woman, 25 years old):")
print(json.dumps(passenger1, indent=2))

response = requests.post(f'{API_URL}/predict', json=passenger1)
print(f"\nStatus Code: {response.status_code}")
print(f"Prediction:")
print(json.dumps(response.json(), indent=2))

# Test passenger 2: Third class man
passenger2 = {
    'Pclass': 3,
    'Name': 'Mr. John Doe',
    'Sex': 'male',
    'Age': 30,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 8,
    'Embarked': 'S',
    'Cabin': None,
    'Ticket': '67890'
}

print("\n" + "-" * 60)
print("Passenger 2 (Third class man, 30 years old):")
print(json.dumps(passenger2, indent=2))

response = requests.post(f'{API_URL}/predict', json=passenger2)
print(f"\nStatus Code: {response.status_code}")
print(f"Prediction:")
print(json.dumps(response.json(), indent=2))

# ============================================
# TEST 4: BATCH PREDICTIONS
# ============================================

print("\n" + "=" * 60)
print("TEST 4: Batch Predictions (POST /predict_batch)")
print("=" * 60)

passengers = [
    {
        'Pclass': 1, 'Name': 'Mrs. Rich Lady', 'Sex': 'female', 
        'Age': 35, 'SibSp': 1, 'Parch': 0, 'Fare': 150,
        'Embarked': 'C', 'Cabin': 'B22', 'Ticket': '11111'
    },
    {
        'Pclass': 2, 'Name': 'Master. Little Boy', 'Sex': 'male',
        'Age': 5, 'SibSp': 0, 'Parch': 2, 'Fare': 30,
        'Embarked': 'S', 'Cabin': None, 'Ticket': '22222'
    },
    {
        'Pclass': 3, 'Name': 'Mr. Poor Man', 'Sex': 'male',
        'Age': 40, 'SibSp': 0, 'Parch': 0, 'Fare': 7,
        'Embarked': 'Q', 'Cabin': None, 'Ticket': '33333'
    }
]

print(f"\nTesting {len(passengers)} passengers...")

response = requests.post(f'{API_URL}/predict_batch', json=passengers)
print(f"\nStatus Code: {response.status_code}")
print(f"Response:")
print(json.dumps(response.json(), indent=2))

# ============================================
# PERFORMANCE TEST
# ============================================

print("\n" + "=" * 60)
print("TEST 5: Performance Test")
print("=" * 60)

n_requests = 100
print(f"\nSending {n_requests} requests...")

start_time = time.time()

for i in range(n_requests):
    requests.post(f'{API_URL}/predict', json=passenger1)

end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / n_requests

print(f"\n✅ Performance Results:")
print(f"   Total time: {total_time:.2f} seconds")
print(f"   Average time per request: {avg_time*1000:.2f} ms")
print(f"   Requests per second: {n_requests/total_time:.2f}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("🎉 API TESTING COMPLETE!")
print("=" * 60)

print("""
API SUCCESSFULLY DEPLOYED! 🚀

Your ML model is now available as a REST API that can:
  ✓ Accept HTTP requests
  ✓ Make real-time predictions
  ✓ Handle single or batch predictions
  ✓ Return probabilities and confidence levels

NEXT STEPS:
  1. Deploy to cloud (AWS, GCP, Azure)
  2. Add authentication
  3. Add logging and monitoring
  4. Scale with load balancer
  5. Add caching for performance
""")