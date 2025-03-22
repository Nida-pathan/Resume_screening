import os
import pandas as pd
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  # Fixing missing assignment
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit app UI
st.title("AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("Rank Resumes"):
    if job_description and uploaded_files:
        resumes_text = [extract_text_from_pdf(file) for file in uploaded_files]
        scores = rank_resumes(job_description, resumes_text)
        
        # Create a DataFrame to display rankings
        results_df = pd.DataFrame({
            "Candidate": [file.name for file in uploaded_files],
            "Similarity Score": scores
        })
        
        results_df = results_df.sort_values(by="Similarity Score", ascending=False)
        st.write("### Ranked Candidates:")
        st.dataframe(results_df)
    else:
        st.warning("Please provide both a job description and at least one resume file.")
