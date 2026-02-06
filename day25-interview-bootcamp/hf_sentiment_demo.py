from transformers import pipeline
import time

print("=" * 60)
print("HUGGING FACE SENTIMENT ANALYSIS DEMO")
print("=" * 60)

print("\nLoading pretrained model...")
start = time.time()
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print(f"Model loaded in {time.time() - start:.2f} seconds")

test_reviews = [
    "This product is absolutely amazing! Best purchase ever!",
    "Terrible quality. Waste of money. Very disappointed.",
    "It's okay. Nothing special but does the job.",
]

print("\n" + "=" * 60)
print("PREDICTIONS")
print("=" * 60)

for review in test_reviews:
    result = classifier(review)[0]
    label = result['label']
    score = result['score']
    emoji = "😊" if label == "POSITIVE" else "😞"
    print(f"\nReview: {review}")
    print(f"Prediction: {emoji} {label} ({score*100:.1f}%)")

print("\n✅ You now have hands-on Hugging Face experience!")
