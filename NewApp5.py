import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="📚 Employee Training Platform", layout="wide")

# Initialize Session State
def init_session_state():
    defaults = {
        'training_content': None,
        'content_generated': False,
        'generation_active': False,
        'answered_questions': set(),
        'question_count': 0,
        'pdf_texts': [],
        'training_queries': [],
        'session_uuid': str(uuid.uuid4()),
        'uploaded_pdfs': [],
        'pdf_filenames': [],
        'feedback_log': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Sidebar Configuration
st.sidebar.title("🎓 Employee Training System")

# Reset Application
if st.sidebar.button("🔄 Reset System"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
    st.rerun()

# Simulated Secure API Key Handling (Replace with secrets.toml in production)
openai_api_key = st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password")
if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        client.models.list()  # Test API key validity
        st.session_state['api_key_valid'] = True
    except Exception:
        st.sidebar.error("Invalid API key. Please check and try again.")
        st.session_state['api_key_valid'] = False
else:
    st.session_state['api_key_valid'] = False

# PDF File Uploader with Validation
def validate_pdf(file):
    if file.size > 10 * 1024 * 1024:  # 10MB limit
        st.error(f"File {file.name} exceeds 10MB limit.")
        return False
    if not file.name.lower().endswith('.pdf'):
        st.error(f"File {file.name} is not a PDF.")
        return False
    return True

# Define extract_pdf_content
def extract_pdf_content(pdf_file):
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text
    except Exception as e:
        st.error(f"Error processing {pdf_file.name}: {e}")
        return ""

# Define chunk_text Before It's Called (Fix for NameError)
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# PDF Uploader Block (Calls extract_pdf_content and chunk_text)
uploaded_pdfs = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)
if uploaded_pdfs and st.session_state['api_key_valid']:
    current_filenames = [pdf.name for pdf in uploaded_pdfs]
    if current_filenames != st.session_state['pdf_filenames']:
        st.session_state['pdf_texts'] = []
        st.session_state['uploaded_pdfs'] = []
        st.session_state['pdf_filenames'] = current_filenames
        with st.spinner("Processing PDFs..."):
            for pdf in uploaded_pdfs:
                if validate_pdf(pdf):
                    text = extract_pdf_content(pdf)
                    if text:
                        st.session_state['pdf_texts'].append({
                            'filename': pdf.name,
                            'text': text,
                            'chunks': chunk_text(text)  # Line 181: Now chunk_text is defined above
                        })
                        st.session_state['uploaded_pdfs'].append(pdf)
        if st.session_state['pdf_texts']:
            st.sidebar.success(f"✅ {len(st.session_state['pdf_texts'])} PDFs processed!")
else:
    st.info("📥 Enter a valid OpenAI API key and upload PDFs to start.")

# Initialize Vector Search
@st.cache_resource
def init_vector_search():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.IndexFlatL2(384)  # Dimension of MiniLM embeddings
    return model, index

if st.session_state['pdf_texts']:
    embed_model, faiss_index = init_vector_search()
    for doc in st.session_state['pdf_texts']:
        if 'embeddings' not in doc:
            embeddings = embed_model.encode(doc['chunks'])
            doc['embeddings'] = embeddings
            faiss_index.add(np.array(embeddings, dtype='float32'))

# Model and Role Selection
model_choices = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Select AI Model", model_choices, index=0)
roles = ["Manager", "Executive", "Developer", "Designer", "Marketer", "HR", "Other", "Fresher"]
user_role = st.sidebar.selectbox("Your Role", roles)
focus_areas = ["Leadership", "Technical Skills", "Communication", "Project Management", "Innovation", "Team Building", "Finance"]
learning_goals = st.sidebar.multiselect("Learning Goals", focus_areas)

# Display Uploaded Files
if st.session_state['pdf_filenames']:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Uploaded Documents")
    for i, fname in enumerate(st.session_state['pdf_filenames'], 1):
        st.sidebar.text(f"{i}. {fname}")

# RAG with Vector Search
def rag_answer_query(query, docs, course_data=None, top_k=3):
    if not st.session_state['api_key_valid']:
        return "Valid API key required."
    if not docs:
        return "No documents available. Upload PDFs first."
    
    try:
        # Vector search for relevant chunks
        query_embedding = embed_model.encode([query])[0]
        distances, indices = faiss_index.search(np.array([query_embedding], dtype='float32'), top_k)
        context = ""
        for idx in indices[0]:
            doc_idx = idx // 100  # Approximate document index
            chunk_idx = idx % 100
            if doc_idx < len(docs) and chunk_idx < len(docs[doc_idx]['chunks']):
                context += f"\nDocument {docs[doc_idx]['filename']}:\n{docs[doc_idx]['chunks'][chunk_idx]}\n"
        
        # Course context
        course_context = ""
        if course_data:
            course_context = f"""
            Course: {course_data.get('course_title', '')}
            Description: {course_data.get('course_description', '')}
            Modules: {', '.join(m.get('title', '') for m in course_data.get('modules', []))}
            """
        
        prompt = f"""
        You are an AI trainer for an employee training platform. Answer the query below using the provided document excerpts and course details. Be precise, professional, and reference specific documents where relevant.
        
        Query: {query}
        Documents: {context[:4000]}
        Course Info: {course_context}
        
        If the query cannot be answered, state so clearly. Provide a detailed, actionable response.
        """
        
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        answer = response.choices[0].message.content
        
        # Log feedback
        with st.container():
            st.write(answer)
            feedback = st.radio("Rate this answer:", ["👍 Good", "👎 Needs Improvement"], key=f"feedback_{query}_{st.session_state['session_uuid']}")
            if feedback:
                log_feedback(query, answer, feedback)
        
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Log Feedback
def log_feedback(query, answer, rating):
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer': answer[:100],
        'rating': rating,
        'session': st.session_state['session_uuid']
    }
    st.session_state['feedback_log'].append(feedback_entry)
    with open('feedback_log.json', 'w') as f:
        json.dump(st.session_state['feedback_log'], f, indent=2)

# Check Quiz Answer
def verify_answer(qid, user_choice, correct_choice):
    if user_choice == correct_choice:
        st.success("🎉 Correct!")
        st.session_state['answered_questions'].add(qid)
        return True
    else:
        st.error(f"Incorrect. Correct answer: {correct_choice}")
        return False

# Generate Training Course
def start_course_creation():
    st.session_state['generation_active'] = True
    st.session_state['content_generated'] = False
    st.rerun()

def create_training_course():
    try:
        doc_content = ""
        for i, doc in enumerate(st.session_state['pdf_texts']):
            doc_content += f"\n--- Document {i+1}: {doc['filename']} ---\n{doc['text'][:2000]}\n"
        
        role_context = f"Role: {user_role}, Goals: {', '.join(learning_goals)}"
        summary = rag_answer_query("Summarize key concepts and applications from these documents.", st.session_state['pdf_texts'])
        
        prompt = f"""
        Create a professional training course based on multiple documents.
        Context: {role_context}
        Document Summary: {summary}
        Documents: {doc_content[:4000]}
        
        Design a course with:
        - A compelling title reflecting integrated document insights
        - A 300+ word description synthesizing all documents
        - 5-8 modules in logical sequence
        - 4-6 learning objectives per module with practical examples
        - Detailed module content as 10-15 bullet points with actionable insights
        - 3-5 quiz questions per module testing key concepts
        
        Return JSON:
        {{
            "course_title": "",
            "course_description": "",
            "modules": [
                {{
                    "title": "",
                    "learning_objectives": [],
                    "content": "",
                    "quiz": {{
                        "questions": [
                            {{
                                "question": "",
                                "options": [],
                                "correct_answer": ""
                            }}
                        ]
                    }}
                }}
            ]
        }}
        """
        
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        st.session_state['training_content'] = json.loads(response.choices[0].message.content)
        st.session_state['content_generated'] = True
        
        # Update question count
        total = sum(len(m.get('quiz', {}).get('questions', [])) for m in st.session_state['training_content'].get('modules', []))
        st.session_state['question_count'] = total
    
    except Exception as e:
        st.error(f"Course creation failed: {e}")
    
    st.session_state['generation_active'] = False

# Generate Progress Report
def generate_progress_report():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Training Progress Report")
    c.drawString(100, 730, f"User: {user_role}")
    c.drawString(100, 710, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    completed = len(st.session_state['answered_questions'])
    total = st.session_state['question_count']
    c.drawString(100, 690, f"Progress: {completed}/{total} questions completed ({completed/total*100:.1f}%)")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Main UI with Tabs
tab1, tab2, tab3 = st.tabs(["📚 Training Content", "❓ Training Queries", "📑 Document Library"])

if st.session_state['generation_active']:
    with st.spinner("Creating your training course..."):
        st.session_state['answered_questions'] = set()
        create_training_course()
        st.success("✅ Course Ready!")
        st.rerun()

with tab1:
    if st.session_state['content_generated'] and st.session_state['training_content']:
        course = st.session_state['training_content']
        st.title(f"🌟 {course.get('course_title', 'Training Course')}")
        st.markdown(f"*Designed for {user_role}s focusing on {', '.join(learning_goals)}*")
        st.write(course.get('course_description', ''))
        
        # Progress Dashboard
        completed = len(st.session_state['answered_questions'])
        total = st.session_state['question_count']
        progress = (completed / total * 100) if total > 0 else 0
        st.progress(progress / 100)
        st.write(f"**Progress:** {completed}/{total} ({progress:.1f}%)")
        st.download_button("📥 Download Report", generate_progress_report(), "progress_report.pdf")
        
        st.markdown("---")
        st.subheader("📋 Course Outline")
        for i, module in enumerate(course.get('modules', []), 1):
            st.write(f"**Module {i}:** {module.get('title', f'Module {i}')}")
        
        st.markdown("---")
        for i, module in enumerate(course.get('modules', []), 1):
            with st.expander(f"📚 Module {i}: {module.get('title', f'Module {i}')}"):
                st.markdown("### 🎯 Objectives:")
                for obj in module.get('learning_objectives', []):
                    st.markdown(f"- {obj}")
                
                st.markdown("### 📖 Content:")
                for line in module.get('content', '').split('\n'):
                    if line.strip():
                        st.markdown(f"• {line}" if not line.startswith(('-', '*', '•')) else line)
                
                st.markdown("### 📝 Quiz:")
                for q_idx, q in enumerate(module.get('quiz', {}).get('questions', []), 1):
                    qid = f"mod_{i}_q_{q_idx}"
                    st.markdown(f"**Question {q_idx}:** {q.get('question', '')}")
                    options = q.get('options', [])
                    if options:
                        choice = st.radio("Answer:", options, key=f"quiz_{qid}")
                        if qid in st.session_state['answered_questions']:
                            st.success("✓ Completed")
                        elif st.button("Submit", key=f"submit_{qid}"):
                            verify_answer(qid, choice, q.get('correct_answer', ''))
                    st.markdown("---")
    else:
        st.title("Employee Training Platform")
        st.markdown("""
        ### AI-Powered Training System
        Upload PDFs, select your role and goals, and generate a custom training course!
        1. Enter API key
        2. Upload training PDFs
        3. Choose role and focus
        4. Generate course
        """)
        if st.session_state['pdf_texts'] and st.session_state['api_key_valid'] and not st.session_state['generation_active']:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Generate Course", use_container_width=True):
                    start_course_creation()

with tab2:
    st.title("💬 Training Queries")
    st.markdown("Submit questions about the course or documents to get AI-generated answers.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Submit Query")
    new_query = st.sidebar.text_area("Your Question:", height=100)
    if st.sidebar.button("Submit"):
        if new_query:
            answer = rag_answer_query(new_query, st.session_state['pdf_texts'], st.session_state['training_content'])
            st.session_state['training_queries'].append({
                'query': new_query,
                'response': answer,
                'answered': bool(answer)
            })
            st.sidebar.success("Query submitted!")
            st.rerun()
    
    if not st.session_state['training_queries']:
        st.info("No queries submitted yet.")
    else:
        for i, query in enumerate(st.session_state['training_queries']):
            with st.expander(f"Query {i+1}: {query['query'][:50]}..."):
                st.write(f"**Query:** {query['query']}")
                st.write(f"**Response:** {query['response']}")

with tab3:
    st.title("📑 Document Library")
    if not st.session_state['pdf_texts']:
        st.info("No documents uploaded.")
    else:
        st.write(f"**{len(st.session_state['pdf_texts'])} documents:**")
        for i, doc in enumerate(st.session_state['pdf_texts']):
            with st.expander(f"Document {i+1}: {doc['filename']}"):
                preview = doc['text'][:1000] + ("..." if len(doc['text']) > 1000 else "")
                st.text_area("Preview:", preview, height=300, disabled=True)
                if st.button(f"Summarize {doc['filename']}", key=f"sum_{i}"):
                    summary = rag_answer_query("Summarize this document.", [doc])
                    st.markdown("### Summary:")
                    st.write(summary)
