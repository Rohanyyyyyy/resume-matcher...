
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import re
import nltk
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Safe download of NLTK data
def safe_nltk_download():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

safe_nltk_download()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def extract_resume_text(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_jobs_from_pdfs(pdf_files):
    job_data = []
    for pdf_file in pdf_files:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            job_text = ""
            for page in doc:
                job_text += page.get_text()
        job_data.append({
            "Job Title": pdf_file.name.replace(".pdf", ""),
            "Job Description": job_text
        })
    return pd.DataFrame(job_data)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def find_best_match(resume_text, job_df):
    job_texts = job_df['Job Description'].fillna('').apply(clean_text).tolist()
    resume_cleaned = clean_text(resume_text)

    all_texts = [resume_cleaned] + job_texts

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_index = similarity_scores.argmax()
    best_score = similarity_scores[best_index]

    return job_df.iloc[best_index], best_score

st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.title("🤖 AI-Based Resume Classifier and Job Matching System")

st.write("Upload your resume and job descriptions (CSV or PDF), and this app will find the best match using AI!")

mode = st.sidebar.radio("Choose job source", ["Upload Files", "Scrape LinkedIn"])
job_df = None

if mode == "Scrape LinkedIn":
    keyword = st.sidebar.text_input("Job keyword", "Data Scientist")
    location = st.sidebar.text_input("Location", "India")
    if st.sidebar.button("♻️ Scrape Jobs"):
        from linkedin_scraper import scrape_linkedin_jobs
        with st.spinner("Scraping…"):
            job_df = scrape_linkedin_jobs(keyword, location, limit=30)
            st.success(f"Scraped {len(job_df)} jobs.")
else:
    job_file_type = st.radio("📁 Job Description File Type", ["CSV", "PDF (1 or more)"])

    if job_file_type == "CSV":
        job_file = st.file_uploader("📑 Upload Job Descriptions (CSV)", type="csv")
        if job_file:
            job_df = pd.read_csv(job_file)

    elif job_file_type == "PDF (1 or more)":
        job_files = st.file_uploader("📑 Upload Job Description PDFs", type="pdf", accept_multiple_files=True)
        if job_files:
            job_df = extract_jobs_from_pdfs(job_files)

resume_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")

if st.button("🔍 Find Best Job Match"):
    if resume_file and job_df is not None:
        with st.spinner("Reading and matching..."):
            resume_text = extract_resume_text(resume_file)
            job_df.dropna(inplace=True)

            if 'Job Description' not in job_df.columns or 'Job Title' not in job_df.columns:
                st.error("❌ Job data must contain 'Job Title' and 'Job Description' columns.")
            else:
                best_job, score = find_best_match(resume_text, job_df)
                st.success("✅ Best Match Found!")
                st.markdown(f"**Job Title:** {best_job['Job Title']}")
                st.markdown(f"**Matching Score:** {round(score * 100, 2)}%")
                st.markdown("**Job Description:**")
                st.markdown(f"> {best_job['Job Description']}")
    else:
        st.warning("⚠️ Please upload both the resume and job data.")
