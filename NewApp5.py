import streamlit as st
import pdfplumber
import numpy as np
import json
import uuid
import os
import tempfile
import io
import requests
from typing import List, Dict, Any
from collections import defaultdict
import re
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.documents import Document

# Page Configuration
st.set_page_config(page_title="📖 Advanced Learning Hub", layout="wide")

# Initialize Session State
def initialize_session():
    defaults = {
        'course_data': None,
        'course_ready': False,
        'generating': False,
        'answered_questions': set(),
        'question_count': 0,
        'doc_chunks': [],
        'vector_store': None,
        'queries': [],
        'session_id': str(uuid.uuid4()),
        'uploaded_docs': [],
        'doc_names': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session()

# Sidebar Setup
st.sidebar.title("🎓 Learning Management System")

# Reset Application
if st.sidebar.button("🔄 Restart Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session()
    st.rerun()

# OpenAI API Key Input
api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

# PDF Uploader
uploaded_docs = st.sidebar.file_uploader("📄 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)

# Function to Extract Text with pdfplumber
def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name
        with pdfplumber.open(temp_file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        os.unlink(temp_file_path)
        return text
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return ""

# Chunking Engine
def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    chunks = []
    paragraphs = re.split(r'\n{2,}', text.strip())
    current_chunk = ""
    current_size = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_size = len(para)
        if current_size + para_size <= max_chunk_size:
            current_chunk += para + "\n\n"
            current_size += para_size
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
            current_size = para_size

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Generate Embeddings Using OpenAI API with requests
def generate_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-ada-002",
        "input": texts
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        embeddings = [embedding['embedding'] for embedding in data['data']]
        return embeddings
    except Exception as e:
        st.error(f"Embedding generation failed: {e}")
        return []

# Process Uploaded PDFs and Store in Chroma
if uploaded_docs and api_key:
    current_doc_names = [doc.name for doc in uploaded_docs]
    if current_doc_names != st.session_state.doc_names:
        st.session_state.doc_chunks = []
        st.session_state.vector_store = None
        st.session_state.uploaded_docs = uploaded_docs
        st.session_state.doc_names = current_doc_names

        with st.spinner("Processing documents..."):
            all_chunks = []
            chunk_metadata = []
            for doc in uploaded_docs:
                text = extract_text_from_pdf(doc)
                if text:
                    chunks = chunk_text(text)
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        chunk_metadata.append({"filename": doc.name, "chunk_index": i})

            if all_chunks:
                embeddings = generate_embeddings(all_chunks, api_key)
                if embeddings:
                    # Prepare documents for Chroma
                    documents = [
                        Document(page_content=chunk, metadata=meta)
                        for chunk, meta in zip(all_chunks, chunk_metadata)
                    ]
                    # Initialize Chroma with LangChain's OpenAI embeddings
                    embedding_function = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
                    vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=embedding_function,
                        collection_name=f"session_{st.session_state.session_id}"
                    )
                    st.session_state.doc_chunks = [
                        {"text": chunk, "metadata": meta}
                        for chunk, meta in zip(all_chunks, chunk_metadata)
                    ]
                    st.session_state.vector_store = vector_store
                    st.sidebar.success(f"✅ Processed {len(uploaded_docs)} PDFs!")
                else:
                    st.error("Failed to generate embeddings.")
            else:
                st.error("No valid text extracted from PDFs.")
else:
    st.info("📤 Please provide an API key and upload PDFs to proceed.")

# Model and Role Selection
models = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Choose AI Model", models, index=0)

roles = ["Manager", "Executive", "Engineer", "Designer", "Marketer", "HR", "Custom", "Entry-Level"]
selected_role = st.sidebar.selectbox("Your Role", roles)

focus_areas = ["Leadership", "Tech Skills", "Communication", "Project Management", "Innovation", "Teamwork", "Finance"]
selected_focus = st.sidebar.multiselect("Learning Goals", focus_areas)

# Display Uploaded Files
if st.session_state.doc_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📜 Uploaded Documents")
    for i, name in enumerate(st.session_state.doc_names, 1):
        st.sidebar.text(f"{i}. {name}")

# RAG Answer Generation with LangChain
def generate_answer(query: str, vector_store: Chroma, api_key: str) -> str:
    if not api_key:
        return "API key required."
    if not vector_store:
        return "No documents available to search."

    # Set up LangChain RetrievalQA
    llm = OpenAI(
        api_key=api_key,
        model=selected_model,
        temperature=0.5
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a learning assistant. Answer the following question based EXCLUSIVELY on the provided document context. Do NOT use external knowledge. If the information is insufficient, return: "Insufficient content to generate a detailed summary."

        Context: {context}

        Question: {question}

        Answer:
        """
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    try:
        result = qa_chain({"query": query})
        answer = result['result']
        return answer
    except Exception as e:
        return f"Failed to generate summary due to error: {str(e)}"

# Employer Queries
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Queries")
query_input = st.sidebar.text_area("Submit a Question:", height=100)
if st.sidebar.button("Add Query"):
    if query_input:
        answer = ""
        if st.session_state.vector_store:
            with st.spinner("Retrieving answer..."):
                answer = generate_answer(query_input, st.session_state.vector_store, api_key)
        else:
            answer = "Please upload documents to enable query answering."

        st.session_state.queries.append({
            "query": query_input,
            "response": answer,
            "answered": bool(answer)
        })
        st.sidebar.success("Query added!")
        st.rerun()

# Answer Verification
def verify_answer(q_id: str, user_response: str, correct_response: str, options: List[str]) -> bool:
    option_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    if correct_response in option_mapping:
        correct_option_index = option_mapping[correct_response]
        correct_option_text = options[correct_option_index]
    else:
        correct_option_text = correct_response

    if user_response == correct_option_text:
        st.session_state.answered_questions.add(q_id)
        st.session_state[f"correct_{q_id}"] = True
        st.success("✅ Correct!")
        return True
    else:
        st.session_state[f"correct_{q_id}"] = False
        st.error(f"Incorrect. Correct answer: {correct_option_text}")
        return False

# Course Generation Trigger
def initiate_course_creation():
    st.session_state.generating = True
    st.session_state.course_ready = False
    st.session_state.course_data = None

# Course Content Generation with LangChain
def create_course_content(vector_store: Chroma, api_key: str) -> Dict[str, Any]:
    # Group chunks by PDF filename
    doc_chunks_by_pdf = defaultdict(list)
    for chunk in st.session_state.doc_chunks:
        filename = chunk['metadata']['filename']
        doc_chunks_by_pdf[filename].append(chunk)

    # Build document context for each PDF
    doc_context = ""
    pdf_summaries = []
    for i, (filename, chunks) in enumerate(doc_chunks_by_pdf.items(), 1):
        pdf_content = "\n".join(chunk['text'][:2000] for chunk in chunks)
        doc_context += f"\n--- Document {i}: {filename} ---\n{pdf_content}\n"

        # Generate summary for this PDF using LangChain
        summary_query = f"Summarize the key concepts, theories, and applications from the document '{filename}'."
        summary_answer = generate_answer(summary_query, vector_store, api_key)
        pdf_summaries.append(f"Summary of {filename}: {summary_answer}")

    doc_summary = "\n".join(pdf_summaries)
    role_context = f"Role: {selected_role}, Focus: {', '.join(selected_focus)}"

    # Truncate to avoid API limits
    doc_context = doc_context[:4000]
    doc_summary = doc_summary[:1500]

    # Use LangChain to generate course content
    llm = OpenAI(
        api_key=api_key,
        model=selected_model,
        temperature=0.7
    )
    prompt_template = PromptTemplate(
        input_variables=["context", "doc_summary", "role_context"],
        template="""
        Create a professional learning course for an employee training system based on multiple documents.

        Context: {role_context}
        Document Summaries: {doc_summary}
        Documents: {context}

        Design a course by:
        1. Crafting an inspiring course title.
        2. Writing a 300-word course description.
        3. Developing exactly 2 modules per PDF, focusing on each PDF's content. For {len(doc_chunks_by_pdf)} PDFs, this will result in {len(doc_chunks_by_pdf) * 2} modules.
        4. Defining 4-6 learning objectives per module.
        5. Summarizing module content in 6-10 detailed bullet points (15-25 words each) that help trainees answer quiz questions.
        6. Including 3-5 quiz questions per module, directly related to the module content.

        Return JSON:
        {{
            "course_title": "Title",
            "course_description": "Description",
            "modules": [
                {{
                    "title": "Module Title",
                    "source_pdf": "Filename of the PDF",
                    "learning_objectives": ["Obj1", "Obj2"],
                    "content": "Bullet-point content",
                    "quiz": {{
                        "questions": [
                            {{
                                "question": "Text",
                                "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                                "correct_answer": "Option 2"
                            }}
                        ]
                    }}
                }}
            ]
        }}
        """
    )

    # Since we're not retrieving documents for course generation, use a simple chain
    from langchain.chains import LLMChain
    chain = LLMChain(llm=llm, prompt=prompt_template)
    try:
        response = chain.run(
            context=doc_context,
            doc_summary=doc_summary,
            role_context=role_context
        )
        course_data = json.loads(response)
        return course_data
    except Exception as e:
        st.error(f"Course creation failed: {str(e)}")
        return None

# Handle Course Generation
if 'create_course_clicked' not in st.session_state:
    st.session_state.create_course_clicked = False

if st.session_state.doc_chunks and api_key and not st.session_state.generating and not st.session_state.course_ready:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Create Course", use_container_width=True):
            st.session_state.create_course_clicked = True
            initiate_course_creation()
            with st.spinner("Crafting your course..."):
                course_data = create_course_content(st.session_state.vector_store, api_key)
                if course_data:
                    st.session_state.course_data = course_data
                    st.session_state.course_ready = True
                    total_questions = sum(
                        len(module.get('quiz', {}).get('questions', []))
                        for module in course_data.get('modules', [])
                    )
                    st.session_state.question_count = total_questions
                    st.success("✅ Course created successfully!")
            st.session_state.generating = False
            st.session_state.create_course_clicked = False

# Main UI with Tabs
tab_course, tab_queries, tab_docs = st.tabs(["📚 Course", "❓ Queries", "📑 Sources"])

with tab_course:
    if st.session_state.course_ready and st.session_state.course_data:
        course = st.session_state.course_data
        if not course.get('course_title') or not course.get('course_description') or not course.get('modules'):
            st.error("Course data is incomplete. Please try generating the course again.")
        else:
            st.title(f"🌟 {course.get('course_title', 'Learning Course')}")
            st.markdown(f"*Tailored for {selected_role}s focusing on {', '.join(selected_focus)}*")
            st.write(course.get('course_description', ''))

            completed = len(st.session_state.answered_questions)
            total = st.session_state.question_count
            progress = (completed / total * 100) if total > 0 else 0
            st.progress(progress / 100)
            st.write(f"**Progress:** {completed}/{total} questions ({progress:.1f}%)")

            st.markdown("---")
            st.subheader("📋 Course Outline")
            modules = course.get('modules', [])
            if not modules:
                st.warning("No modules found in the course data.")
            for i, module in enumerate(modules, 1):
                source_pdf = module.get('source_pdf', 'Unknown PDF')
                st.write(f"**Module {i}:** {module.get('title', f'Module {i}')} (Source: {source_pdf})")

            st.markdown("---")
            for i, module in enumerate(modules, 1):
                source_pdf = module.get('source_pdf', 'Unknown PDF')
                with st.expander(f"📚 Module {i}: {module.get('title', f'Module {i}')} (Source: {source_pdf})"):
                    st.markdown("### 🎯 Objectives")
                    objectives = module.get('learning_objectives', [])
                    if not objectives:
                        st.warning("No learning objectives defined for this module.")
                    for obj in objectives:
                        st.markdown(f"- {obj}")

                    st.markdown("### 📖 Content")
                    content_value = module.get('content', '')
                    if isinstance(content_value, list):
                        content = content_value
                    else:
                        content = content_value.split('\n') if isinstance(content_value, str) else []
                    if not content:
                        st.warning("No content available for this module.")
                    for line in content:
                        if line.strip():
                            st.markdown(f"• {line}" if not line.startswith(('- ', '* ', '• ')) else line)

                    st.markdown("### 💡 Takeaways")
                    st.info("Apply these skills in your professional role for immediate impact.")

                    st.markdown("### 📝 Quiz")
                    quiz_questions = module.get('quiz', {}).get('questions', [])
                    if not quiz_questions:
                        st.warning("No quiz questions available for this module.")
                    for q_idx, q in enumerate(quiz_questions, 1):
                        q_id = f"mod_{i}_q_{q_idx}"
                        st.markdown(f"**Question {q_idx}:** {q.get('question', '')}")
                        options = q.get('options', [])
                        correct_response = q.get('correct_answer', '')
                        if options:
                            option_key = f"quiz_{i}_{q_idx}"
                            user_answer = st.radio("Choose:", options, key=option_key)
                            submit_key = f"submit_{i}_{q_idx}"
                            if q_id in st.session_state.answered_questions:
                                if st.session_state.get(f"correct_{q_id}", False):
                                    st.success("✓ Correct!")
                                else:
                                    option_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
                                    correct_option_text = options[option_mapping.get(correct_response, 0)] if correct_response in option_mapping else correct_response
                                    st.error(f"Incorrect. Correct answer: {correct_option_text}")
                            elif st.button("Check", key=submit_key):
                                verify_answer(q_id, user_answer, correct_response, options)
                        st.markdown("---")
    elif st.session_state.generating:
        st.title("Generating Your Course...")
        st.info("Please wait while the course is being created. This may take a few moments.")
    else:
        st.title("Advanced Learning Hub")
        st.markdown("""
        ## Elevate Your Skills with AI-Driven Learning

        Upload training PDFs, and I'll craft a tailored course integrating all materials.

        ### Steps:
        1. Input your OpenAI API key
        2. Select your role and focus areas
        3. Upload PDFs
        4. Generate a custom course

        Start your learning journey now!
        """)

with tab_queries:
    st.title("💬 Queries")
    st.markdown("Submit questions to get AI-generated answers based on uploaded documents.")
    if not st.session_state.queries:
        st.info("No queries submitted. Use the sidebar to add one.")
    else:
        for i, query in enumerate(st.session_state.queries, 1):
            with st.expander(f"Query {i}: {query['query'][:50]}..." if len(query['query']) > 50 else f"Query {i}: {query['query']}"):
                st.markdown(f"**Question:** {query['query']}")
                st.markdown(f"**Response:** {query['response']}")

with tab_docs:
    st.title("📑 Uploaded Documents")
    if not st.session_state.doc_chunks:
        st.info("No documents uploaded yet.")
    else:
        for doc_name in st.session_state.doc_names:
            with st.expander(f"📜 {doc_name}"):
                chunks = [chunk for chunk in st.session_state.doc_chunks if chunk['metadata']['filename'] == doc_name]
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(chunk['text'])
                    st.markdown("---")
