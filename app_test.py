import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Define the model and tokenizer for English to Hindi translation
model_name = "Helsinki-NLP/opus-mt-en-hi"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Function to perform translation
def translate(text):
    # Tokenize the input text
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    # Decode the output to text
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# Streamlit App Layout
st.title("English to Hindi Translator")
st.write("Enter English text below to translate to Hindi:")

# Input text box
english_text = st.text_area("English Text")

# Translate button
if st.button("Translate"):
    if english_text:
        # Perform translation
        hindi_translation = translate(english_text)
        st.write("Translated Text (Hindi):")
        st.success(hindi_translation[0])
    else:
        st.warning("Please enter text to translate.")
