import streamlit as st
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fpdf import FPDF
import torch

torch.set_num_threads(1)

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(page_title="EduNote AI", layout="wide")
st.title("🎓 EduNote AI")
st.subheader("Lecture Text → Smart Notes Generator")

# -----------------------------------------
# Load Models (Cached)
# -----------------------------------------
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = load_models()

# -----------------------------------------
# Text Cleaning
# -----------------------------------------
def clean_text(text):
    text = re.sub(r'\b(um|ah|okay|so|like)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------------------
# Summarization
# -----------------------------------------
def summarize_text(text, max_len=200):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    summary_ids = model.generate(
        inputs,
        max_length=max_len,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# -----------------------------------------
# Key Points Generator
# -----------------------------------------
def generate_keypoints(text):
    prompt = "summarize and extract 5 key points: " + text
    return summarize_text(prompt, max_len=150)

# -----------------------------------------
# Quiz Generator
# -----------------------------------------
def generate_quiz(text):
    prompt = "generate 3 short quiz questions from this text: " + text
    return summarize_text(prompt, max_len=150)

# -----------------------------------------
# PDF Generator
# -----------------------------------------
def generate_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, content)
    file_path = "Lecture_Notes.pdf"
    pdf.output(file_path)
    return file_path

# -----------------------------------------
# User Input Section
# -----------------------------------------
text_input = st.text_area(
    "📚 Paste Your Lecture Text Here",
    height=250
)

if text_input:
    cleaned_text = clean_text(text_input)

    st.subheader("📄 Cleaned Text")
    st.write(cleaned_text)

    # Summary
    st.info("Generating Summary...")
    summary_text = summarize_text(cleaned_text)
    st.subheader("📝 Summary")
    st.write(summary_text)

    # Key Points
    st.info("Generating Key Points...")
    keypoints = generate_keypoints(cleaned_text)
    st.subheader("📌 Key Points")
    st.write(keypoints)

    # Quiz
    st.info("Generating Quiz Questions...")
    quiz = generate_quiz(cleaned_text)
    st.subheader("❓ Quiz Questions")
    st.write(quiz)

    # PDF Download
    if st.button("Download Notes as PDF"):
        full_content = f"""
SUMMARY:
{summary_text}

KEY POINTS:
{keypoints}

QUIZ:
{quiz}
"""
        pdf_file = generate_pdf(full_content)

        with open(pdf_file, "rb") as f:
            st.download_button(
                "Click Here to Download",
                f,
                file_name="Lecture_Notes.pdf"
            )
