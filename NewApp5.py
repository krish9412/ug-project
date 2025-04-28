import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import numpy as np
import faiss
import json
import uuid
from openai import OpenAI
import re
import asyncio

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
        'doc_embeddings': None,
        'faiss_index': None,
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

# Function to Extract Text with PyMuPDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_file.seek(0)
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return ""

# Chunking Engine
def chunk_text(text, max_chunk_size=500):
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

# Embedding Function
def generate_embeddings(texts, api_key):
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        embeddings = [np.array(data.embedding, dtype=np.float32) for data in response.data]
        return embeddings
    except Exception as e:
        st.error(f"Embedding generation failed: {e}")
        return []

# FAISS Index Creation
def create_faiss_index(embeddings):
    if not embeddings:
        return None
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# Process Uploaded PDFs
if uploaded_docs and api_key:
    current_doc_names = [doc.name for doc in uploaded_docs]
    if current_doc_names != st.session_state.doc_names:
        st.session_state.doc_chunks = []
        st.session_state.doc_embeddings = None
        st.session_state.faiss_index = None
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
                    st.session_state.doc_chunks = [
                        {"text": chunk, "metadata": meta}
                        for chunk, meta in zip(all_chunks, chunk_metadata)
                    ]
                    st.session_state.doc_embeddings = embeddings
                    st.session_state.faiss_index = create_faiss_index(embeddings)
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

# Semantic Retrieval
def retrieve_relevant_chunks(query, index, embeddings, chunks, k=3):
    if not index or not embeddings:
        return []
    query_embedding = generate_embeddings([query], api_key)
    if not query_embedding:
        return []
    distances, indices = index.search(np.array([query_embedding[0]]), k)
    return [chunks[i] for i in indices[0]]

# RAG Answer Generation
def generate_answer(query, chunks, course_data=None):
    if not api_key:
        return "API key required."
    if not chunks:
        return "No relevant document chunks found."
    
    context = ""
    for i, chunk in enumerate(chunks[:3], 1):
        context += f"Document {i} ({chunk['metadata']['filename']}):\n{chunk['text'][:2000]}\n\n"
    
    course_context = ""
    if course_data:
        course_context = f"""
        Course: {course_data.get('course_title', '')}
        Description: {course_data.get('course_description', '')}
        Modules:
        """
        for i, module in enumerate(course_data.get('modules', []), 1):
            course_context += f"""
            Module {i}: {module.get('title', '')}
            Objectives: {', '.join(module.get('learning_objectives', []))}
            Content: {module.get('content', '')[:200]}...
            """
    
    prompt = f"""
    As a learning assistant, provide a detailed answer to the following question based on the provided context and course information. Be precise and reference specific documents.

    Question: {query}

    Document Context: {context}

    Course Context: {course_context}

    Answer comprehensively, citing documents where applicable. If the information is insufficient, state so clearly.
    """
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Answer generation error: {str(e)}"

# Employer Queries
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Queries")
query_input = st.sidebar.text_area("Submit a Question:", height=100)
if st.sidebar.button("Add Query"):
    if query_input:
        answer = ""
        if st.session_state.doc_chunks and st.session_state.faiss_index:
            with st.spinner("Retrieving answer..."):
                relevant_chunks = retrieve_relevant_chunks(
                    query_input,
                    st.session_state.faiss_index,
                    st.session_state.doc_embeddings,
                    st.session_state.doc_chunks
                )
                answer = generate_answer(
                    query_input,
                    relevant_chunks,
                    st.session_state.course_data if st.session_state.course_ready else None
                )
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
def verify_answer(q_id, user_response, correct_response):
    if user_response == correct_response:
        st.session_state.answered_questions.add(q_id)
        st.success("✅ Correct!")
        return True
    else:
        st.error(f"Incorrect. Correct answer: {correct_response}")
        return False

# Course Generation Trigger
def initiate_course_creation():
    st.session_state.generating = True
    st.session_state.course_ready = False
    st.rerun()

# Course Content Generation
async def create_course_content():
    try:
        doc_context = ""
        for i, chunk in enumerate(st.session_state.doc_chunks, 1):
            doc_context += f"\n--- Document {i} ({chunk['metadata']['filename']}) ---\n{chunk['text'][:3000]}\n"

        role_context = f"Role: {selected_role}, Focus: {', '.join(selected_focus)}"
        summary_query = "Summarize the key concepts, theories, and applications from these documents."
        summary_chunks = retrieve_relevant_chunks(
            summary_query,
            st.session_state.faiss_index,
            st.session_state.doc_embeddings,
            st.session_state.doc_chunks
        )
        doc_summary = generate_answer(summary_query, summary_chunks)

        prompt = f"""
        Create a professional learning course based on multiple documents.
        Context: {role_context}
        Document Summary: {doc_summary}
        Documents: {doc_context[:5000]}

        Design a course by:
        1. Analyzing documents for themes and insights
        2. Crafting an inspiring course title
        3. Writing a 300-word course description
        4. Developing 5-8 modules in logical sequence
        5. Defining 4-6 learning objectives per module
        6. Creating detailed module content as bullet points (10-15 points, practical and quiz-relevant)
        7. Including 3-5 quiz questions per module

        Return JSON:
        {{
            "course_title": "Title",
            "course_description": "Description",
            "modules": [
                {{
                    "title": "Module Title",
                    "learning_objectives": ["Obj1", "Obj2"],
                    "content": "Bullet-point content",
                    "quiz": {{
                        "questions": [
                            {{
                                "question": "Text",
                                "options": ["A", "B", "C", "D"],
                                "correct_answer": "A"
                            }}
                        ]
                    }}
                }}
            ]
        }}
        """
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        course_data = json.loads(response.choices[0].message.content)
        st.session_state.course_data = course_data
        st.session_state.course_ready = True

        total_questions = sum(
            len(module.get('quiz', {}).get('questions', []))
            for module in course_data.get('modules', [])
        )
        st.session_state.question_count = total_questions

    except Exception as e:
        st.error(f"Course creation failed: {e}")
    finally:
        st.session_state.generating = False

# Main UI with Tabs
tab_course, tab_queries, tab_docs = st.tabs(["📚 Course", "❓ Queries", "📑 Sources"])

if st.session_state.generating:
    with st.spinner("Crafting your course..."):
        st.session_state.answered_questions = set()
        asyncio.run(create_course_content())
        st.success("✅ Course created!")
        st.rerun()

with tab_course:
    if st.session_state.course_ready and st.session_state.course_data:
        course = st.session_state.course_data
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
        for i, module in enumerate(modules, 1):
            st.write(f"**Module {i}:** {module.get('title', f'Module {i}')}")

        st.markdown("---")
        for i, module in enumerate(modules, 1):
            with st.expander(f"📚 Module {i}: {module.get('title', f'Module {i}')}"):
                st.markdown("### 🎯 Objectives")
                for obj in module.get('learning_objectives', []):
                    st.markdown(f"- {obj}")

                st.markdown("### 📖 Content")
                content_value = module.get('content', '')
                content = content_value.split('\n') if isinstance(content_value, str) else []
                for line in content:
                    if line.strip():
                        st.markdown(f"• {line}" if not line.startswith(('- ', '* ', '• ')) else line)

                st.markdown("### 💡 Takeaways")
                st.info("Apply these skills in your professional role for immediate impact.")

                st.markdown("### 📝 Quiz")
                for q_idx, q in enumerate(module.get('quiz', {}).get('questions', []), 1):
                    q_id = f"mod_{i}_q_{q_idx}"
                    st.markdown(f"**Question {q_idx}:** {q.get('question', '')}")
                    options = q.get('options', [])
                    if options:
                        option_key = f"quiz_{i}_{q_idx}"
                        user_answer = st.radio("Choose:", options, key=option_key)
                        submit_key = f"submit_{i}_{q_idx}"
                        if q_id in st.session_state.answered_questions:
                            st.success("✓ Completed")
                        elif st.button("Check", key=submit_key):
                            verify_answer(q_id, user_answer, q.get('correct_answer', ''))
                    st.markdown("---")
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
        if st.session_state.doc_chunks and api_key and not st.session_state.generating:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Create Course", use_container_width=True):
                    initiate_course_creation()

with tab_queries:
    st.title("💬 Queries")
    st.markdown("Submit questions to get AI-generated answers based on uploaded documents.")
    if not st.session_state.queries:
        st.info("No queries submitted. Use the sidebar to add one.")
    else:
        for i, query in enumerate(st.session_state.queries, 1):
            with st.expander(f"Query {i}: {query['query'][:50]}..." if len(query['query']) > 50 else f"Query {i}: {query['query']}"):
                st.write(f"**Query:** {query['query']}")
                if query['answered']:
                    st.write(f"**Response:** {query['response']}")
                else:
                    st.info("Generating response...")
                    if st.session_state.doc_chunks:
                        relevant_chunks = retrieve_relevant_chunks(
                            query['query'],
                            st.session_state.faiss_index,
                            st.session_state.doc_embeddings,
                            st.session_state.doc_chunks
                        )
                        answer = generate_answer(
                            query['query'],
                            relevant_chunks,
                            st.session_state.course_data if st.session_state.course_ready else None
                        )
                        st.session_state.queries[i-1]['response'] = answer
                        st.session_state.queries[i-1]['answered'] = True
                        st.rerun()
                    else:
                        st.warning("Upload documents to generate responses.")

with tab_docs:
    st.title("📑 Document Sources")
    if not st.session_state.doc_chunks:
        st.info("No documents uploaded. Add PDFs in the sidebar.")
    else:
        st.write(f"**{len(st.session_state.doc_names)} documents processed:**")
        unique_docs = set(chunk['metadata']['filename'] for chunk in st.session_state.doc_chunks)
        for i, filename in enumerate(unique_docs, 1):
            with st.expander(f"Document {i}: {filename}"):
                doc_text = "\n".join(
                    chunk['text'][:1000]
                    for chunk in st.session_state.doc_chunks
                    if chunk['metadata']['filename'] == filename
                )
                st.text_area("Preview:", value=doc_text[:1000] + "...", height=300, disabled=True)
                if st.button(f"Summarize {filename}", key=f"sum_{i}"):
                    with st.spinner("Summarizing..."):
                        summary_chunks = [
                            chunk for chunk in st.session_state.doc_chunks
                            if chunk['metadata']['filename'] == filename
                        ]
                        summary = generate_answer(
                            "Summarize this document's key concepts and applications.",
                            summary_chunks
                        )
                        st.markdown("### Summary:")
                        st.write(summary)
