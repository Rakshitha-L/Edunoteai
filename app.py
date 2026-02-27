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
# Load Model (Cached)
# -----------------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------------------
# Text Cleaning
# -----------------------------------------
def clean_text(text):
    text = re.sub(r'\b(um|ah|okay|so|like)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------------------
# Generate Output (Core Function)
# -----------------------------------------
def generate_output(prompt, max_len=200):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    output_ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=40,
        num_beams=2,
        do_sample=False
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# -----------------------------------------
# PDF Generator
# -----------------------------------------
def generate_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, content)
    return pdf.output(dest="S").encode("latin-1")

# -----------------------------------------
# User Input
# -----------------------------------------
lecture_text = st.text_area("📚 Paste Your Lecture Text Here", height=250)

if st.button("Generate Smart Notes"):

    if lecture_text.strip() == "":
        st.warning("Please paste lecture text.")
    else:
        cleaned_text = clean_text(lecture_text)

        st.subheader("📄 Cleaned Text")
        st.write(cleaned_text)

        # ---------------- Summary ----------------
        st.info("Generating Summary...")
        summary_prompt = "Summarize this lecture:\n" + cleaned_text
        summary = generate_output(summary_prompt, 180)

        st.subheader("📝 Summary")
        st.write(summary)

        # ---------------- Key Points ----------------
        st.info("Extracting Key Points...")
        key_prompt = "Extract 5 important bullet points from this lecture:\n" + cleaned_text
        keypoints = generate_output(key_prompt, 150)

        st.subheader("📌 Key Points")
        st.write(keypoints)

        # ---------------- Quiz ----------------
        st.info("Generating Quiz Questions...")
        quiz_prompt = "Generate 3 short quiz questions from this lecture:\n" + cleaned_text
        quiz = generate_output(quiz_prompt, 150)

        st.subheader("❓ Quiz Questions")
        st.write(quiz)

        # ---------------- PDF Download ----------------
        if st.button("Download Notes as PDF"):
            full_content = f"""
SUMMARY:
{summary}

KEY POINTS:
{keypoints}

QUIZ QUESTIONS:
{quiz}
"""
            pdf_file = generate_pdf(full_content)

            st.download_button(
                "Click Here to Download",
                pdf_file,
                file_name="Lecture_Notes.pdf",
                mime="application/pdf"
            )
