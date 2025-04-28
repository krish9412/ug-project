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
        return "No relevant document chunks found to summarize."
    
    # Build context from retrieved chunks
    context = ""
    for i, chunk in enumerate(chunks[:3], 1):
        context += f"Document {i} ({chunk['metadata']['filename']}):\n{chunk['text'][:2000]}\n\n"
    
    # Check if the retrieved chunks are relevant to the query
    query_lower = query.lower()
    context_lower = context.lower()
    relevant = any(word in context_lower for word in query_lower.split() if len(word) > 3)  # Basic relevance check
    
    if not relevant:
        return "Unable to generate a summary due to lack of relevant content in the provided documents."
    
    # Build course context if available
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
    
    # Updated prompt to enforce strict context usage
    prompt = f"""
    You are a learning assistant. Your task is to provide a detailed answer to the following question based EXCLUSIVELY on the provided document context and course information. Do NOT use any external knowledge or assumptions beyond what is explicitly stated in the context. If the information is not available in the context, return a placeholder summary stating that the content is insufficient.

    Question: {query}

    Document Context: {context}

    Course Context: {course_context}

    Answer strictly based on the provided context, citing specific documents where applicable. If the information is insufficient, return: "Insufficient content to generate a detailed summary."
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
        return f"Failed to generate summary due to API error: {str(e)}"

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
def verify_answer(q_id, user_response, correct_response, options):
    # Map the correct_response (e.g., "B") to the corresponding option text
    option_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    if correct_response in option_mapping:
        correct_option_index = option_mapping[correct_response]
        correct_option_text = options[correct_option_index]
    else:
        correct_option_text = correct_response  # Fallback in case correct_response is already the full text

    if user_response == correct_option_text:
        st.session_state.answered_questions.add(q_id)
        st.session_state[f"correct_{q_id}"] = True  # Track correct answers
        st.success("✅ Correct!")
        return True
    else:
        st.session_state[f"correct_{q_id}"] = False  # Track incorrect answers
        st.error(f"Incorrect. Correct answer: {correct_option_text}")
        return False

# Course Generation Trigger
def initiate_course_creation():
    st.session_state.generating = True
    st.session_state.course_ready = False

# Course Content Generation
async def create_course_content():
    try:
        # Debugging: Log start of the process
        st.info("Starting course generation process...")

        # Group chunks by PDF filename
        st.info("Grouping document chunks by PDF...")
        doc_chunks_by_pdf = {}
        for chunk in st.session_state.doc_chunks:
            filename = chunk['metadata']['filename']
            if filename not in doc_chunks_by_pdf:
                doc_chunks_by_pdf[filename] = []
            doc_chunks_by_pdf[filename].append(chunk)

        # Build context and summary for each PDF
        st.info("Building context and summaries for each PDF...")
        doc_context = ""
        pdf_summaries = []
        for i, (filename, chunks) in enumerate(doc_chunks_by_pdf.items(), 1):
            pdf_content = "\n".join(chunk['text'][:2000] for chunk in chunks)  # Limit to 2000 chars per PDF
            doc_context += f"\n--- Document {i}: {filename} ---\n{pdf_content}\n"

            # Generate a summary for this specific PDF
            pdf_chunks = [chunk for chunk in chunks]
            summary_query = f"Summarize the key concepts, theories, and applications from the document '{filename}'."
            summary_chunks = pdf_chunks  # Use only chunks from this PDF
            pdf_summary = generate_answer(summary_query, summary_chunks)
            # Ensure pdf_summary is a string
            if not isinstance(pdf_summary, str):
                pdf_summary = "Unable to generate summary due to unexpected response format."
            pdf_summaries.append(f"Summary of {filename}: {pdf_summary}")

        # Combine summaries for the prompt
        doc_summary = "\n".join(pdf_summaries)

        role_context = f"Role: {selected_role}, Focus: {', '.join(selected_focus)}"

        # Truncate doc_context and doc_summary to avoid exceeding API limits
        doc_context = doc_context[:4000]  # Total limit
        doc_summary = doc_summary[:1500]  # Total summary limit

        prompt = f"""
        Create a professional learning course based on multiple documents. The documents are provided below, with each document representing a separate PDF file.

        Context: {role_context}
        Document Summaries: {doc_summary}
        Documents: {doc_context}

        Design a course by:
        1. Analyzing each document (PDF) separately to identify its themes and insights.
        2. Crafting an inspiring course title that reflects the combined focus of all documents.
        3. Writing a 300-word course description that summarizes the overall learning objectives.
        4. Developing exactly 2 modules for each PDF, ensuring that each module focuses exclusively on the content of its respective PDF. Do not mix content between PDFs. For {len(doc_chunks_by_pdf)} PDFs, this will result in {len(doc_chunks_by_pdf) * 2} total modules.
        5. Defining 4-6 learning objectives per module, specific to the PDF's content.
        6. Summarizing the module content in 5-8 concise, digestible bullet points that directly help trainees answer the quiz questions. Each bullet point must be a complete sentence (10-20 words) and a clear, focused takeaway that connects to the quiz content and the specific PDF. For example, if a quiz question is "What is a key strategy for remote team engagement?", a bullet point should be: "Regular virtual check-ins are a key strategy for remote team engagement."
        7. Including 3-5 quiz questions per module, with each question directly related to the content of the module and the specific PDF.

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
                                "correct_answer": "Option 2"  # Use the full text of the correct option
                            }}
                        ]
                    }}
                }}
            ]
        }}
        """
        
        # Debugging: Log before API call
        st.info("Calling OpenAI API to generate course content...")

        # Add timeout for the API call
        client = OpenAI(api_key=api_key)
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                ),
                timeout=300  # 5-minute timeout
            )
        except asyncio.TimeoutError:
            raise Exception("OpenAI API call timed out after 5 minutes. Please try again or use a smaller PDF.")

        # Debugging: Log after API call
        st.info("Received response from OpenAI API. Processing...")

        # Validate the response structure
        if not hasattr(response, 'choices') or not response.choices:
            raise Exception("Invalid API response: 'choices' attribute missing or empty.")

        course_data = json.loads(response.choices[0].message.content)
        
        # Validate that course_data is a dictionary
        if not isinstance(course_data, dict):
            raise Exception("Invalid course data format: Expected a dictionary, got: " + str(type(course_data)))

        # Validate that each PDF has modules
        modules = course_data.get('modules', [])
        pdfs_with_modules = set(module.get('source_pdf', '') for module in modules)
        if len(pdfs_with_modules) != len(doc_chunks_by_pdf):
            st.warning("The generated course did not include modules for all PDFs. Adjusting the course structure...")
            # Fallback: If a PDF is missing modules, add placeholder modules (optional)

        # Validate module content
        for module in modules:
            content = module.get('content', '')
            bullet_points = [line.strip() for line in content.split('\n') if line.strip()]
            if len(bullet_points) < 5:
                st.warning(f"Module '{module.get('title', 'Unknown')}' has fewer than 5 bullet points. Expected 5-8.")
            for bp in bullet_points:
                if len(bp.split()) < 10:
                    st.warning(f"Bullet point in '{module.get('title', 'Unknown')}' is too short: '{bp}'. Expected 10-20 words.")

        st.session_state.course_data = course_data
        st.session_state.course_ready = True

        total_questions = sum(
            len(module.get('quiz', {}).get('questions', []))
            for module in course_data.get('modules', [])
        )
        st.session_state.question_count = total_questions

        st.success("✅ Course created successfully!")

    except Exception as e:
        st.error(f"Course creation failed: {str(e)}")
        # Ensure the generating flag is reset even on error
        st.session_state.generating = False
        st.session_state.course_ready = False
    finally:
        st.session_state.generating = False

# Main UI with Tabs
tab_course, tab_queries, tab_docs = st.tabs(["📚 Course", "❓ Queries", "📑 Sources"])

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
            source_pdf = module.get('source_pdf', 'Unknown PDF')
            st.write(f"**Module {i}:** {module.get('title', f'Module {i}')} (Source: {source_pdf})")

        st.markdown("---")
        for i, module in enumerate(modules, 1):
            source_pdf = module.get('source_pdf', 'Unknown PDF')
            with st.expander(f"📚 Module {i}: {module.get('title', f'Module {i}')} (Source: {source_pdf})"):
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
                    correct_response = q.get('correct_answer', '')
                    if options:
                        option_key = f"quiz_{i}_{q_idx}"
                        user_answer = st.radio("Choose:", options, key=option_key)
                        submit_key = f"submit_{i}_{q_idx}"
                        if q_id in st.session_state.answered_questions:
                            if st.session_state.get(f"correct_{q_id}", False):
                                st.success("✓ Correct!")
                            else:
                                # Map the correct_response to the full text for display
                                option_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
                                correct_option_text = options[option_mapping.get(correct_response, 0)] if correct_response in option_mapping else correct_response
                                st.error(f"Incorrect. Correct answer: {correct_option_text}")
                        elif st.button("Check", key=submit_key):
                            verify_answer(q_id, user_answer, correct_response, options)
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
                    st.session_state.answered_questions = set()
                    with st.spinner("Crafting your course..."):
                        asyncio.run(create_course_content())
                    # No st.rerun() here; let create_course_content handle the state

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
