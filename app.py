import pickle

# Load model
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

print("💬 Emotion Detection from Text")
print("Type 'exit' to stop")

while True:
    text = input("\nEnter text: ")
    if text.lower() == "exit":
        break

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)

    print("🔍 Detected Emotion:", prediction[0])
