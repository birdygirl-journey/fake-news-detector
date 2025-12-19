import pickle

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def detect_fake_news(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return prediction[0]

# Test examples
news_english = "Breaking news: aliens landed today"
news_urdu = "یہ خبر مکمل طور پر جھوٹی ہے"

print("English News:", detect_fake_news(news_english))
print("Urdu News:", detect_fake_news(news_urdu))
