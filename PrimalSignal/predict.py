import pickle


# load model
model = pickle.load(open("models/toxic_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


print("==============================")
print("TOXIC CHAT DETECTOR")
print("==============================")


while True:

    userMessage = input("\nEnter a message: ")

    if userMessage.lower() == "exit":
        break

    transformedText = vectorizer.transform([userMessage])

    prediction = model.predict(transformedText)[0]


    if prediction == 1:
        print("\nPrediction: TOXIC MESSAGE")

    else:
        print("\nPrediction: NON-TOXIC MESSAGE")