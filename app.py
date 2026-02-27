import streamlit as st
from fpdf import FPDF
import re

st.set_page_config(page_title="EduNote AI", layout="wide")

st.title("🎓 EduNote AI")
st.subheader("Lecture Text → Smart Notes Generator")

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.replace('"', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------
# Smart Sentence Split
# -----------------------------
def split_sentences(text):
    # Avoid breaking decimal numbers like 3.13
    text = re.sub(r'(\d)\.(\d)', r'\1<dot>\2', text)
    sentences = re.split(r'\.\s+', text)
    sentences = [s.replace('<dot>', '.') for s in sentences]
    return [s.strip() for s in sentences if s.strip()]

# -----------------------------
# Summary
# -----------------------------
def generate_summary(text):
    sentences = split_sentences(text)
    return ". ".join(sentences[:3]) + "."

# -----------------------------
# Key Points
# -----------------------------
def generate_keypoints(text):
    sentences = split_sentences(text)
    key_points = sentences[:5]
    return "\n".join([f"• {point}" for point in key_points])

# -----------------------------
# Quiz Generator
# -----------------------------
def generate_quiz(text):
    sentences = split_sentences(text)
    questions = []

    for i, sentence in enumerate(sentences[:3]):
        words = sentence.split()
        if len(words) > 4:
            keyword = words[0]
            question = f"{i+1}. Explain the concept of '{keyword}' in the lecture."
            questions.append(question)

    return "\n".join(questions)

# -----------------------------
# PDF Generator
# -----------------------------
def generate_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, content)
    file_path = "Lecture_Notes.pdf"
    pdf.output(file_path)
    return file_path

# -----------------------------
# UI
# -----------------------------
lecture_text = st.text_area("📚 Paste Your Lecture Text Here", height=250)

if st.button("Generate Smart Notes"):

    if lecture_text.strip() == "":
        st.warning("Please paste lecture text.")
    else:
        cleaned = clean_text(lecture_text)

        st.subheader("📄 Cleaned Text")
        st.write(cleaned)

        st.subheader("📝 Summary")
        summary = generate_summary(cleaned)
        st.write(summary)

        st.subheader("📌 Key Points")
        keypoints = generate_keypoints(cleaned)
        st.write(keypoints)

        st.subheader("❓ Quiz Questions")
        quiz = generate_quiz(cleaned)
        st.write(quiz)

        full_content = f"""
SUMMARY:
{summary}

KEY POINTS:
{keypoints}

QUIZ:
{quiz}
"""

        if st.button("Download Notes as PDF"):
            pdf_file = generate_pdf(full_content)
            with open(pdf_file, "rb") as f:
                st.download_button("Click Here to Download", f, file_name="Lecture_Notes.pdf")
