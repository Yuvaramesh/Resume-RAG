import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import PyPDF2

# Force torch to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Streamlit app title
st.title("🔍 Ask Questions About Your Resume (Local LLM)")

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text()
    return full_text

# Cache the model loading to avoid reloads
@st.cache_resource
def load_local_llm():
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return qa_pipeline

# Generate the prompt and get response
def generate_answer(qa_pipeline, context, question):
    prompt = f"""Answer the question based only on the context below.

Context:
{context}

Question: {question}

Answer:"""
    result = qa_pipeline(prompt, max_new_tokens=150)[0]["generated_text"]
    return result.strip()

# Upload section
uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type="pdf")

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Resume uploaded and processed.")

    # Load model
    qa_pipeline = load_local_llm()

    # Ask a question
    question = st.text_input("❓ Ask a question about your resume")

    if question:
        with st.spinner("Generating answer..."):
            answer = generate_answer(qa_pipeline, resume_text, question)
            st.markdown("### 🧠 Answer:")
            st.write(answer)
