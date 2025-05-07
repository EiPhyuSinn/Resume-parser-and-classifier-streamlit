

import streamlit as st
import joblib
import docx
import PyPDF2
import re

# Load model
svc_model = joblib.load('pretrained/svc.pkl')

# Clean resume text
def clean_resume_text(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"[^a-zA-Z0-9.,;!?()&%-]", ' ', text)
    text = re.sub(r'\b\d{10,}\b', '', text)
    return text.strip()

# Extract text from file
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        reader = PyPDF2.PdfReader(file)
        return ' '.join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == 'docx':
        doc = docx.Document(file)
        return '\n'.join(p.text for p in doc.paragraphs)
    elif ext == 'txt':
        try:
            return file.read().decode('utf-8')
        except:
            return file.read().decode('latin-1')
    else:
        raise ValueError("Unsupported file type")

# Predict category
def predict_resume(text):
    cleaned = clean_resume_text(text)
    return svc_model.predict([cleaned])[0]

# Streamlit app
def main():
    st.title("📄 Resume Category Predictor")
    file = st.file_uploader("Upload resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
    manual_text = st.text_area("Or paste resume text here", height=200)

    resume_text = None
    if file:
        try:
            resume_text = extract_text(file)
            st.success("Text extracted from file.")
        except Exception as e:
            st.error(f"File error: {e}")
    elif manual_text.strip():
        resume_text = manual_text.strip()

    if resume_text:
        if st.checkbox("Show resume text"):
            st.text_area("Resume Text", resume_text, height=250)

        category = predict_resume(resume_text)
        st.success(f"Predicted Category: **{category}**")

if __name__ == "__main__":
    main()
