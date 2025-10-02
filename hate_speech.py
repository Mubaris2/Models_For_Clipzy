from transformers import pipeline

# Label ID to text label mapping
label_map = {
    'LABEL_0': 'acceptable',
    'LABEL_1': 'inappropriate',
    'LABEL_2': 'offensive',
    'LABEL_3': 'violent'
}

# Load model with PyTorch backend
pipe = pipeline("text-classification", model="IMSyPP/hate_speech_en", framework="pt")

# Ask for user input
text = input("Enter text to classify for hate speech: ")

# Get prediction
output = pipe(text)[0]
predicted_label = label_map.get(output['label'], output['label'])

# Print only the predicted class
print(f"Prediction: {predicted_label}")
