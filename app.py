import streamlit as st
import whisper
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fpdf import FPDF
import tempfile

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(page_title="EduNote AI", layout="wide")
st.title("🎓 EduNote AI")
st.subheader("Lecture Voice → Smart Notes Generator")

# -----------------------------------------
# Load Models (Cached)
# -----------------------------------------
@st.cache_resource
def load_models():
    # Whisper Speech Recognition
    speech_model = whisper.load_model("Tiny")

    # Hugging Face Summarization Model
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

    return speech_model, tokenizer, model

speech_model, tokenizer, model = load_models()

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
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_len, min_length=60, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# -----------------------------------------
# Key Points Generator
# -----------------------------------------
def generate_keypoints(text):
    prompt = f"Extract 5 key points from this lecture:\n{text}"
    return summarize_text(prompt, max_len=150)

# -----------------------------------------
# Quiz Generator
# -----------------------------------------
def generate_quiz(text):
    prompt = f"Generate 3 short quiz questions from this lecture:\n{text}"
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
# Upload Audio
# -----------------------------------------
uploaded_file = st.file_uploader("Upload Lecture Audio File", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Step 1: Speech-to-Text
    st.info("Transcribing audio...")
    result = speech_model.transcribe(temp_path)
    raw_text = result["text"]
    st.success("Transcription Completed!")

    cleaned_text = clean_text(raw_text)
    st.subheader("📄 Transcribed Text")
    st.write(cleaned_text)

    # Step 2: Summary
    st.info("Generating Summary...")
    summary_text = summarize_text(cleaned_text)
    st.subheader("📝 Summary")
    st.write(summary_text)

    # Step 3: Key Points
    st.info("Generating Key Points...")
    keypoints = generate_keypoints(cleaned_text)
    st.subheader("📌 Key Points")
    st.write(keypoints)

    # Step 4: Quiz
    st.info("Generating Quiz Questions...")
    quiz = generate_quiz(cleaned_text)
    st.subheader("❓ Quiz Questions")
    st.write(quiz)

    # Step 5: Download PDF
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

            st.download_button("Click Here to Download", f, file_name="Lecture_Notes.pdf")
