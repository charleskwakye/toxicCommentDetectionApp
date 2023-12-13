import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_path = "/Users/charlesnanakwakye/BigData/StreamlitToxic/comment_finetuned"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

st.title("Toxic Comment Classification")

# Use st.text_area for user input
comment = st.text_area("Enter your text here:", "")

# Use st.button for prediction button
if st.button("Predict"):
    # Tokenize the input text
    tok_test = tokenizer(comment, truncation=True,
                         max_length=128, return_tensors="pt")

    # Make the prediction
    with torch.no_grad():
        logits = model(**tok_test).logits

    predicted_class_id = logits.argmax().item()

    # Make sure to check if the adjusted index is within the range of your labels
    if predicted_class_id in model.config.id2label:
        predicted_label = model.config.id2label[predicted_class_id]
        st.markdown(
            f"<center><b><h1>Prediction: {predicted_label}</h1></b></center>", unsafe_allow_html=True)
    else:
        st.write("Invalid label index:", predicted_class_id)
