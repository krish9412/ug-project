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
import pinecone
import time
from typing import List, Dict, Any, Tuple

# Page Configuration
st.set_page_config(page_title="📚 Advanced Professional Learning Platform", layout="wide")

# Initializing sessions state variables
if 'course_content' not in st.session_state:
    st.session_state.course_content = None
if 'course_generated' not in st.session_state:
    st.session_state.course_generated = False
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'completed_questions' not in st.session_state:
    st.session_state.completed_questions = set()
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = []
if 'employer_queries' not in st.session_state:
    st.session_state.employer_queries = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []
if 'pinecone_initialized' not in st.session_state:
    st.session_state.pinecone_initialized = False
if 'index_name' not in st.session_state:
    st.session_state.index_name = "employee-training-docs"
if 'namespace' not in st.session_state:
    st.session_state.namespace = str(uuid.uuid4())

# Sidebars Appearance
st.sidebar.title("🎓 Professional Learning System")

# Clear Sessions Button & Session Management
if st.sidebar.button("🔄 Reset Application"):
    # Delete vectors from Pinecone if initialized
    if st.session_state.pinecone_initialized:
        try:
            index = pinecone.Index(st.session_state.index_name)
            index.delete(delete_all=True, namespace=st.session_state.namespace)
        except Exception as e:
            st.sidebar.error(f"Error clearing Pinecone vectors: {e}")
    
    # Reset all session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Initialize new session
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.namespace = str(uuid.uuid4())
    st.session_state.extracted_texts = []
    st.session_state.uploaded_files = []
    st.session_state.uploaded_file_names = []
    st.session_state.pinecone_initialized = False
    st.rerun()

# 🔐 API Keys Input Section
with st.sidebar.expander("🔑 API Configurations", expanded=False):
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")
    pinecone_environment = st.text_input("Pinecone Environment", value="gcp-starter")

# 📄 Multi-File Uploader for PDFs
uploaded_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)

# Function to chunk text for embeddings
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for processing."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

# Function to generate embeddings using OpenAI
def generate_embeddings(texts: List[str], client: OpenAI) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI API."""
    embeddings = []
    for text in texts:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            # Add a zero vector as a placeholder for failed embeddings
            embeddings.append([0.0] * 1536)  # OpenAI embeddings dimension
    return embeddings

# Initialize Pinecone index
def initialize_pinecone(api_key: str, environment: str, index_name: str):
    """Initialize Pinecone client and ensure index exists."""
    try:
        pinecone.init(api_key=api_key, environment=environment)
        
        # Check if index exists
        existing_indexes = pinecone.list_indexes()
        
        if index_name not in existing_indexes:
            # Create index if it doesn't exist
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine"
            )
            # Wait for index to be ready
            while not index_name in pinecone.list_indexes():
                time.sleep(1)
        
        return pinecone.Index(index_name)
    
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        return None

# Function to upload document chunks to Pinecone
def upload_to_pinecone(index, chunks: List[str], embeddings: List[List[float]], 
                      metadata: Dict[str, Any], namespace: str):
    """Upload document chunks and their embeddings to Pinecone."""
    try:
        vectors_to_upsert = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create a unique ID for each chunk
            vector_id = f"{metadata['doc_id']}_{i}"
            
            # Create metadata for the chunk
            chunk_metadata = {
                "text": chunk,
                "filename": metadata["filename"],
                "chunk_id": i,
                "doc_id": metadata["doc_id"]
            }
            
            # Prepare vector for upserting
            vectors_to_upsert.append((vector_id, embedding, chunk_metadata))
            
            # Upsert in batches of 100 to avoid rate limits
            if len(vectors_to_upsert) >= 100:
                index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                vectors_to_upsert = []
        
        # Upsert any remaining vectors
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert, namespace=namespace)
            
        return True
    
    except Exception as e:
        st.error(f"Error uploading to Pinecone: {e}")
        return False

# Process uploaded files and add to session state
if uploaded_files and openai_api_key and pinecone_api_key:
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # Initialize Pinecone
    if not st.session_state.pinecone_initialized:
        with st.sidebar.spinner("Initializing Pinecone..."):
            index = initialize_pinecone(
                pinecone_api_key, 
                pinecone_environment,
                st.session_state.index_name
            )
            
            if index:
                st.session_state.pinecone_initialized = True
                st.sidebar.success("✅ Pinecone initialized successfully!")
            else:
                st.sidebar.error("Failed to initialize Pinecone. Please check your API key and environment.")
    
    # Clear previous uploads if list has changed
    current_filenames = [file.name for file in uploaded_files]
    if current_filenames != st.session_state.uploaded_file_names:
        # Reset Pinecone namespace for new uploads
        if st.session_state.pinecone_initialized:
            try:
                index = pinecone.Index(st.session_state.index_name)
                index.delete(delete_all=True, namespace=st.session_state.namespace)
                st.session_state.namespace = str(uuid.uuid4())  # Create new namespace
            except Exception as e:
                st.sidebar.error(f"Error clearing previous vectors: {e}")
        
        st.session_state.extracted_texts = []
        st.session_state.uploaded_files = []
        st.session_state.uploaded_file_names = current_filenames
        
        # Extract text from each PDF and store in session state
        with st.spinner("Processing and embedding PDF files..."):
            for i, pdf_file in enumerate(uploaded_files):
                extracted_text = extract_pdf_text(pdf_file)
                
                if extracted_text:
                    doc_id = f"doc_{i}_{uuid.uuid4().hex[:8]}"
                    
                    # Store in session state
                    st.session_state.extracted_texts.append({
                        "filename": pdf_file.name,
                        "text": extracted_text,
                        "doc_id": doc_id
                    })
                    st.session_state.uploaded_files.append(pdf_file)
                    
                    # Process for Pinecone if initialized
                    if st.session_state.pinecone_initialized:
                        # Chunk the text
                        chunks = chunk_text(extracted_text)
                        
                        # Generate embeddings
                        embeddings = generate_embeddings(chunks, client)
                        
                        # Upload to Pinecone
                        index = pinecone.Index(st.session_state.index_name)
                        upload_success = upload_to_pinecone(
                            index,
                            chunks,
                            embeddings,
                            {"filename": pdf_file.name, "doc_id": doc_id},
                            st.session_state.namespace
                        )
                        
                        if upload_success:
                            st.sidebar.success(f"✅ {pdf_file.name} processed and embedded")
                        else:
                            st.sidebar.error(f"❌ Failed to embed {pdf_file.name}")
                
        if st.session_state.extracted_texts:
            st.sidebar.success(f"✅ {len(st.session_state.extracted_texts)} PDF files processed successfully!")
else:
    st.info("📥 Please enter your API keys and upload PDF files to begin.")

# 🎯 GPT Model and Role selection
model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Select OpenAI Model", model_options, index=0)

role_options = ["Manager", "Executive", "Developer", "Designer", "Marketer", "Human Resources", "Other", "Fresher"]
role = st.sidebar.selectbox("Select Your Role", role_options)

learning_focus_options = ["Leadership", "Technical Skills", "Communication", "Project Management", "Innovation", "Team Building", "Finance"]
learning_focus = st.sidebar.multiselect("Select Learning Focus", learning_focus_options)

# Display uploaded files in sidebar
if st.session_state.uploaded_file_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Uploaded Files")
    for i, filename in enumerate(st.session_state.uploaded_file_names):
        st.sidebar.text(f"{i+1}. {filename}")

# Enhanced RAG function using Pinecone vector search
def generate_rag_answer(question, documents, course_content=None):
    try:
        if not openai_api_key:
            return "API key is required to generate answers."
        
        if not documents:
            return "Document text is not available. Please process documents first."
        
        client = OpenAI(api_key=openai_api_key)
        
        # Generate embedding for the question
        question_embedding = generate_embeddings([question], client)[0]
        
        # Get relevant context from Pinecone if initialized
        context_chunks = []
        if st.session_state.pinecone_initialized:
            try:
                index = pinecone.Index(st.session_state.index_name)
                query_results = index.query(
                    vector=question_embedding,
                    top_k=5,
                    namespace=st.session_state.namespace,
                    include_metadata=True
                )
                
                # Extract the relevant chunks and their metadata
                for match in query_results.matches:
                    context_chunks.append({
                        "text": match.metadata["text"],
                        "filename": match.metadata["filename"],
                        "score": match.score
                    })
            except Exception as e:
                st.error(f"Error querying Pinecone: {e}")
                # Fallback to direct document search
                for doc in documents[:3]:
                    context_chunks.append({
                        "text": doc["text"][:2000],
                        "filename": doc["filename"],
                        "score": 0.5  # Default score
                    })
        else:
            # Fallback to direct document search
            for doc in documents[:3]:
                context_chunks.append({
                    "text": doc["text"][:2000],
                    "filename": doc["filename"],
                    "score": 0.5  # Default score
                })
        
        # Format the context for the prompt
        combined_context = ""
        for i, chunk in enumerate(context_chunks):
            combined_context += f"\nDocument: {chunk['filename']} (Relevance: {chunk['score']:.2f})\n{chunk['text']}\n"
        
        # Include course content for additional context if available
        course_context = ""
        if course_content:
            course_context = f"""
            Course Title: {course_content.get('course_title', '')}
            Course Description: {course_content.get('course_description', '')}
            
            Module Information:
            """
            for i, module in enumerate(course_content.get('modules', []), 1):
                course_context += f"""
                Module {i}: {module.get('title', '')}
                Learning Objectives: {', '.join(module.get('learning_objectives', []))}
                Content Summary: {module.get('content', '')[:200]}...
                """
        
        prompt = f"""
        You are an AI assistant for a professional learning platform. Answer the following question 
        based on the provided document content. Be specific, accurate, and helpful.
        
        Question: {question}
        
        Retrieved Document Content: {combined_context}
        
        Course Information: {course_context}
        
        Provide a comprehensive answer using information from the documents and course contents.
        If the question cannot be answered based on the provided information, say so politely.
        Reference specific documents when appropriate in your answer.
        """
        
        # Generate response
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        # Return generated answers
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Employee Queries Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Employer Queries")

new_query = st.sidebar.text_area("Add a new question:", height=100)
if st.sidebar.button("Submit Question"):
    if new_query:
        # Generate proper answers automatically if documents are available
        answer = ""
        if st.session_state.extracted_texts:
            with st.spinner("Generating answer..."):
                answer = generate_rag_answer(
                    new_query, 
                    st.session_state.extracted_texts,
                    st.session_state.course_content if st.session_state.course_generated else None
                )
        else:
            answer = "Please upload and process documents first to enable question answering."
        
        st.session_state.employer_queries.append({
            "question": new_query,
            "answer": answer,
            "answered": bool(answer)
        })
        st.sidebar.success("Question submitted and answered!")
        st.rerun()

# Functions to check answer and update progress
def check_answer(question_id, user_answer, correct_answer):
    if user_answer == correct_answer:
        st.success("🎉 Correct! Well done!")
        # Add to completed questions set if not already there
        st.session_state.completed_questions.add(question_id)
        return True
    else:
        st.error(f"Not quite. The correct answer is: {correct_answer}")
        return False

# Course Generation function
def generate_course():
    # Set generation flag to True when starting
    st.session_state.is_generating = True
    st.session_state.course_generated = False
    st.rerun()  # Trigger rerun to show loading state

# Function to actually generate the course content
def perform_course_generation():
    try:
        # Check if we have the necessary API keys and documents
        if not openai_api_key:
            st.error("OpenAI API key is required to generate a course.")
            st.session_state.is_generating = False
            return
            
        if not st.session_state.extracted_texts:
            st.error("Please upload and process documents first.")
            st.session_state.is_generating = False
            return
            
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Get relevant context from Pinecone for course generation
        all_relevant_chunks = []
        if st.session_state.pinecone_initialized:
            try:
                # Create a query embedding for the course focus
                query_text = f"Educational material about {', '.join(learning_focus)} for {role}s"
                query_embedding = generate_embeddings([query_text], client)[0]
                
                index = pinecone.Index(st.session_state.index_name)
                query_results = index.query(
                    vector=query_embedding,
                    top_k=20,  # Get more chunks for course generation
                    namespace=st.session_state.namespace,
                    include_metadata=True
                )
                
                # Extract relevant chunks
                for match in query_results.matches:
                    all_relevant_chunks.append({
                        "text": match.metadata["text"],
                        "filename": match.metadata["filename"]
                    })
            except Exception as e:
                st.error(f"Error retrieving context from Pinecone: {e}")
                # Fallback to using all documents
                all_relevant_chunks = []
                for doc in st.session_state.extracted_texts:
                    chunks = chunk_text(doc["text"])
                    for chunk in chunks[:5]:  # Limit chunks per document
                        all_relevant_chunks.append({
                            "text": chunk,
                            "filename": doc["filename"]
                        })
        else:
            # Fallback to using all documents
            all_relevant_chunks = []
            for doc in st.session_state.extracted_texts:
                chunks = chunk_text(doc["text"])
                for chunk in chunks[:5]:  # Limit chunks per document
                    all_relevant_chunks.append({
                        "text": chunk,
                        "filename": doc["filename"]
                    })
        
        # Combine context for the course generation prompt
        combined_context = ""
        for i, chunk in enumerate(all_relevant_chunks[:15]):  # Limit to 15 chunks
            combined_context += f"\n--- FROM DOCUMENT: {chunk['filename']} ---\n{chunk['text']}\n"
        
        # Get a document summary first
        summary_query = "Create a comprehensive summary of these documents highlighting key concepts, theories, and practical applications across all materials."
        document_summary = generate_rag_answer(summary_query, st.session_state.extracted_texts)
        
        professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
        
        prompt = f"""
        Design a comprehensive professional learning course based on the multiple documents provided.
        Context: {professional_context}
        Document Summary: {document_summary}
        
        Document Contents: {combined_context}
        
        Create an engaging, thorough and well-structured course by:
        1. Analyzing all provided documents and identifying common themes, complementary concepts, and unique insights from each source
        2. Creating an inspiring course title that reflects the integrated knowledge from all documents
        3. Writing a detailed course description (at least 300 words) that explains how the course synthesizes information from multiple sources
        4. Developing 5-8 comprehensive modules that build upon each other in a logical sequence
        5. Providing 4-6 clear learning objectives for each module with specific examples and practical applications
        6. Creating detailed, well-explained content for each module (at least 500 words per module) including:
           - Real-world examples and case studies
           - Practical applications of concepts
           - Visual explanations where appropriate
           - Step-by-step guides for complex procedures
           - Comparative analysis when sources present different perspectives
        7. Including a quiz with 3-5 thought-provoking questions per module for better understanding
        
        Return the response in the following JSON format:
        {{
            "course_title": "Your Course Title",
            "course_description": "Detailed description of the course",
            "modules": [
                {{
                    "title": "Module 1 Title",
                    "learning_objectives": ["Objective 1", "Objective 2", "Objective 3"],
                    "content": "Module content text with detailed explanations, examples, and practical applications",
                    "quiz": {{
                        "questions": [
                            {{
                                "question": "Question text?",
                                "options": ["Option A", "Option B", "Option C", "Option D"],
                                "correct_answer": "Option A"
                            }}
                        ]
                    }}
                }}
            ]
        }}
        
        Make the content exceptionally practical, actionable, and tailored to the professional context.
        Provide detailed explanations, real-world examples, and practical applications in each module content.
        Where document sources provide different perspectives or approaches to the same topic, compare and contrast them.
        """
        
        try:
            # Generate course content
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            # Accessing the response content
            response_content = response.choices[0].message.content
            
            try:
                st.session_state.course_content = json.loads(response_content)
                st.session_state.course_generated = True
                
                # Count total questions for progress tracking
                total_questions = 0
                for module in st.session_state.course_content.get("modules", []):
                    quiz = module.get("quiz", {})
                    total_questions += len(quiz.get("questions", []))
                st.session_state.total_questions = total_questions
                
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON response: {e}")
                st.text(response_content)
        
        except Exception as e:
            st.error(f"OpenAI API Error: {e}")
            st.error("Please check your API key and model selection.")
            
    except Exception as e:
        st.error(f"Error: {e}")
    
    # Always reset the generation flag when done
    st.session_state.is_generating = False

# Main contents area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📊 Analytics", "📑 Document Sources"])

# Check if we're in the middle of generating a course and need to continue
if st.session_state.is_generating:
    with st.spinner("Generating your personalized course from multiple documents..."):
        # Reset completed questions when generating a new course
        st.session_state.completed_questions = set()
        perform_course_generation()
        st.success("✅ Your Comprehensive Course is Ready!")
        st.rerun()  # Refresh the UI after completion

with tab1:
    # Display Course Content
    if st.session_state.course_generated and st.session_state.course_content:
        course = st.session_state.course_content
        
        # Course Header with appreciation
        st.title(f"🌟 {course.get('course_title', 'Professional Course')}")
        st.markdown(f"*Specially designed for {role}s focusing on {', '.join(learning_focus)}*")
        st.write(course.get('course_description', 'A structured course to enhance your skills.'))
        
        # Tracking the Progress
        completed = len(st.session_state.completed_questions)
        total = st.session_state.total_questions
        progress_percentage = (completed / total * 100) if total > 0 else 0
        
        st.progress(progress_percentage / 100)
        st.write(f"**Progress:** {completed}/{total} questions completed ({progress_percentage:.1f}%)")
        
        st.markdown("---")
        st.subheader("📋 Course Overview")
        
        # Safely access module titles
        modules = course.get("modules", [])
        if modules:
            modules_list = [module.get('title', f'Module {i+1}') for i, module in enumerate(modules)]
            for i, module_title in enumerate(modules_list, 1):
                st.write(f"**Module {i}:** {module_title}")
        else:
            st.warning("No modules were found in the course content.")
        
        st.markdown("---")
        
        # Detailed Module Contents with improved formatting
        for i, module in enumerate(modules, 1):
            module_title = module.get('title', f'Module {i}')
            with st.expander(f"📚 Module {i}: {module_title}"):
                # Module Learning Objectives
                st.markdown("### 🎯 Learning Objectives:")
                objectives = module.get('learning_objectives', [])
                if objectives:
                    for obj in objectives:
                        st.markdown(f"- {obj}")
                else:
                    st.write("No learning objectives specified.")
                
                # Module Content with better readability
                st.markdown("### 📖 Module Content:")
                module_content = module.get('content', 'No content available for this module.')
                
                # Split the content into paragraphs and add proper formatting
                paragraphs = module_content.split('\n\n')
                for para in paragraphs:
                    if para.strip().startswith('#'):
                        # Handle markdown headers
                        st.markdown(para)
                    elif para.strip().startswith('*') and para.strip().endswith('*'):
                        # Handle emphasized text
                        st.markdown(para)
                    elif para.strip().startswith('1.') or para.strip().startswith('- '):
                        # Handle lists
                        st.markdown(para)
                    else:
                        # Regular paragraphs
                        st.write(para)
                        st.write("")  # Add spacing between paragraphs
                
                # Key Takeaways section
                st.markdown("### 💡 Key Takeaways:")
                st.info("The content in this module will help you develop practical skills that you can apply immediately in your professional context.")
                
                # Module Quiz with improved UI
                st.markdown("### 📝 Module Quiz:")
                quiz = module.get('quiz', {})
                questions = quiz.get('questions', [])
                
                if questions:
                    for q_idx, q in enumerate(questions, 1):
                        question_id = f"module_{i}_question_{q_idx}"
                        question_text = q.get('question', f'Question {q_idx}')
                        
                        # Create quiz question container
                        quiz_container = st.container()
                        with quiz_container:
                            st.markdown(f"**Question {q_idx}:** {question_text}")
                            
                            options = q.get('options', [])
                            if options:
                                # Create a unique key for each radio button
                                option_key = f"quiz_{i}_{q_idx}"
                                user_answer = st.radio("Select your answer:", options, key=option_key)
                                
                                # Create a unique key for each submit button
                                submit_key = f"submit_{i}_{q_idx}"
                                
                                # Show completion status for this question
                                if question_id in st.session_state.completed_questions:
                                    st.success("✓ Question completed")
                                else:
                                    if st.button(f"Check Answer", key=submit_key):
                                        correct_answer = q.get('correct_answer', '')
                                        check_answer(question_id, user_answer, correct_answer)
                            else:
                                st.write("No options available for this question.")
                        
                        st.markdown("---")
                else:
                    st.write("No quiz questions available for this module.")

    else:
        # Welcome screen when no course is generated yet
        st.title("Welcome to Professional Learning Platform")
        st.markdown("""
        ## Transform your professional development with AI-powered learning system
        
        Upload multiple PDF documents, and I'll create a comprehensive, integrated learning course just for you!
        
        ### How it works:
        1. Enter your OpenAI API key and Pinecone API key in the sidebar
        2. Select your professional role and learning focus
        3. Upload multiple PDF documents related to your area of interest
        4. Click "Generate Course" to create your personalized learning journey that combines insights from all documents
        
        Get ready to enhance your skills and accelerate your professional growth!
        """)
        
        # Generate Course Button - only if not currently generating
        if st.session_state.extracted_texts and openai_api_key and not st.session_state.is_generating:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Generate My Course", use_container_width=True):
                    generate_course()
        elif st.session_state.is_generating:
            st.info("Generating your personalized course... Please wait.")

with tab2:
    st.title("💬 Employer Queries")
    st.markdown("""
    This section allows employers to ask questions and get AI-generated answers about the course content or related topics.
    Submit your questions in the sidebar, and our AI will automatically generate answers based on the uploaded documents.
    """)
    
    if not st.session_state.employer_queries:
        st.info("No questions have been submitted yet. Add a question in the sidebar to get started.")
    else:
        for i, query in enumerate(st.session_state.employer_
                                  # Continuing from the previous code...
    
        for i, query in enumerate(st.session_state.employer_queries):
            with st.expander(f"Question {i+1}: {query['question'][:50]}..." if len(query['question']) > 50 else f"Question {i+1}: {query['question']}"):
                st.write(f"**Question:** {query['question']}")
                
                if query['answered']:
                    st.write(f"**Answer:** {query['answer']}")
                else:
                    st.info("Generating answer...")
                    # Generate answer on-demand if not already answered
                    if st.session_state.extracted_texts:
                        try:
                            answer = generate_rag_answer(
                                query['question'], 
                                st.session_state.extracted_texts,
                                st.session_state.course_content if st.session_state.course_generated else None
                            )
                            st.session_state.employer_queries[i]['answer'] = answer
                            st.session_state.employer_queries[i]['answered'] = True
                            st.rerun()
                        except Exception as e:
                            error_msg = f"Error generating answer: {str(e)}. Please try resetting the application."
                            st.error(error_msg)
                            st.session_state.employer_queries[i]['answer'] = error_msg
                            st.session_state.employer_queries[i]['answered'] = True
                    else:
                        st.warning("No documents uploaded yet. Please upload documents to generate answers.")

with tab3:
    st.title("📊 Analytics Dashboard")
    
    # Course Progress Analytics
    st.subheader("Course Progress Analytics")
    if st.session_state.course_generated and st.session_state.course_content:
        # Calculate progress metrics
        completed = len(st.session_state.completed_questions)
        total = st.session_state.total_questions
        progress_percentage = (completed / total * 100) if total > 0 else 0
        
        # Display progress metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Completed Questions", completed)
        with col2:
            st.metric("Total Questions", total)
        with col3:
            st.metric("Completion Rate", f"{progress_percentage:.1f}%")
        
        # Module completion visualization
        st.subheader("Module Completion")
        modules = st.session_state.course_content.get("modules", [])
        
        if modules:
            # Calculate completion per module
            module_stats = []
            for i, module in enumerate(modules, 1):
                module_questions = [f"module_{i}_question_{q_idx}" 
                                   for q_idx in range(1, len(module.get("quiz", {}).get("questions", [])) + 1)]
                module_completed = sum(1 for q in module_questions if q in st.session_state.completed_questions)
                module_total = len(module_questions)
                module_percentage = (module_completed / module_total * 100) if module_total > 0 else 0
                
                module_stats.append({
                    "module": f"Module {i}: {module.get('title', '')}",
                    "completed": module_completed,
                    "total": module_total,
                    "percentage": module_percentage
                })
            
            # Display module completion chart
            for mod in module_stats:
                st.write(f"**{mod['module']}**")
                st.progress(mod['percentage'] / 100)
                st.write(f"{mod['completed']}/{mod['total']} questions completed ({mod['percentage']:.1f}%)")
        else:
            st.info("No modules found to analyze.")
    else:
        st.info("Generate a course first to view analytics.")
    
    # Document Usage Analytics
    st.subheader("Document Usage Analytics")
    if st.session_state.pinecone_initialized and st.session_state.extracted_texts:
        st.write("Top documents referenced in answers:")
        
        # This would require tracking document references in actual implementation
        # Here we'll simulate it with mock data based on the actual documents
        doc_stats = []
        for i, doc in enumerate(st.session_state.extracted_texts):
            doc_stats.append({
                "filename": doc["filename"],
                "references": len(st.session_state.employer_queries) - i,  # Mock data
                "relevance_score": min(0.95, 0.7 + (i * 0.05))  # Mock data
            })
        
        # Display as a table
        st.table(pd.DataFrame(doc_stats))
    else:
        st.info("Upload documents and initialize Pinecone to view document analytics.")

with tab4:
    st.title("📑 Document Sources")
    
    if not st.session_state.extracted_texts:
        st.info("No documents have been uploaded yet. Please upload PDF files in the sidebar to see their content here.")
    else:
        st.write(f"**{len(st.session_state.extracted_texts)} documents uploaded:**")
        
        for i, doc in enumerate(st.session_state.extracted_texts):
            with st.expander(f"Document {i+1}: {doc['filename']}"):
                # Display document preview (first 1000 characters)
                preview_text = doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text']
                st.markdown("### Document Preview:")
                st.text_area("Content Preview:", value=preview_text, height=300, disabled=True)
                
                # Add document summary using AI
                if st.button(f"Generate Summary for {doc['filename']}", key=f"sum_{i}"):
                    with st.spinner("Generating document summary..."):
                        summary_query = f"Create a comprehensive summary of this document highlighting key concepts, theories, and practical applications:"
                        summary = generate_rag_answer(summary_query, [doc])
                        st.markdown("### AI-Generated Summary:")
                        st.write(summary)
                
                # Show document statistics
                word_count = len(doc['text'].split())
                st.write(f"**Document Statistics:** {word_count} words")
                
                # Document search
                st.markdown("### Search Within Document:")
                search_term = st.text_input("Enter search term:", key=f"search_{i}")
                if search_term:
                    # Simple text search implementation
                    search_results = []
                    text_chunks = doc['text'].split('\n')
                    for chunk_idx, chunk in enumerate(text_chunks):
                        if search_term.lower() in chunk.lower():
                            search_results.append({
                                "chunk_idx": chunk_idx,
                                "text": chunk
                            })
                    
                    if search_results:
                        st.success(f"Found {len(search_results)} matches")
                        for result in search_results[:5]:  # Limit to 5 results
                            st.markdown(f"**Match:**")
                            # Highlight the search term in the text
                            highlighted_text = result["text"].replace(
                                search_term, 
                                f"**{search_term}**"
                            )
                            st.markdown(highlighted_text)
                            st.markdown("---")
                    else:
                        st.warning(f"No matches found for '{search_term}'")

# Add the missing pandas import at the top of the file
import pandas as pd

# Add export functionality - this would go right above the tabs section
st.sidebar.markdown("---")
st.sidebar.subheader("📤 Export Options")

if st.session_state.course_generated and st.session_state.course_content:
    if st.sidebar.button("📄 Export Course as PDF"):
        st.sidebar.info("PDF export functionality would be implemented here")
    
    if st.sidebar.button("📊 Export Analytics Report"):
        st.sidebar.info("Analytics export functionality would be implemented here")

# Add settings section to sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")

# Add system settings
with st.sidebar.expander("System Settings"):
    chunk_size = st.slider("Document Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    
    # Add a checkbox to enable/disable advanced RAG features
    use_advanced_rag = st.checkbox("Use Advanced RAG Features", value=True)
    
    # Add a checkbox to enable/disable analytics tracking
    enable_analytics = st.checkbox("Enable Analytics Tracking", value=True)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Professional Learning Platform")
st.sidebar.markdown("*Powered by OpenAI and Pinecone*")
