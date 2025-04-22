import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
import chromadb
from openai import OpenAI
import numpy as np

# Page Configuration
st.set_page_config(page_title="📚 Professional Learning Platform", layout="wide")

# Initializing sessions state variables
if 'course_content' not in st.session_state:
    st.session_state.course_content = None
if 'course_generated' not in st.session_state:
    st.session_state.course_generated = False
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False  # Added flag to track generation state
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
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'embeddings_created' not in st.session_state:
    st.session_state.embeddings_created = False

# Sidebars Appearance
st.sidebar.title("🎓 Professional Learning System")

# Clear Sessions Button & Session Management
if st.sidebar.button("🔄 Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.extracted_texts = []
    st.session_state.uploaded_files = []
    st.session_state.uploaded_file_names = []
    st.session_state.embeddings_created = False
    # Initialize ChromaDB client again after reset
    st.session_state.chroma_client = chromadb.Client()
    st.rerun()

# 🔐 OpenAI API Key Inputs
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")

# 📄 Multi-File Uploader for PDFs
uploaded_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)

# Initialize ChromaDB client
if 'chroma_client' not in st.session_state or st.session_state.chroma_client is None:
    try:
        st.session_state.chroma_client = chromadb.Client()
        st.session_state.collection = st.session_state.chroma_client.create_collection(
            name=f"learning_docs_{st.session_state.session_id}",
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity for embeddings
        )
    except Exception as e:
        st.sidebar.error(f"Error initializing ChromaDB: {e}")

# Function to create embeddings using OpenAI
def create_embedding(text, model="text-embedding-3-small"):
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating embedding: {e}")
        return None

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length and end - start == chunk_size:
            # Find the last period or newline to make chunks more coherent
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            breakpoint = max(last_period, last_newline)
            if breakpoint > start + chunk_size // 2:  # Ensure chunk is not too small
                end = breakpoint + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end < text_length else text_length
        
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

# Process uploaded files and add to session state
if uploaded_files and openai_api_key:
    # Clear previous uploads if list has changed
    current_filenames = [file.name for file in uploaded_files]
    if current_filenames != st.session_state.uploaded_file_names or not st.session_state.embeddings_created:
        st.session_state.extracted_texts = []
        st.session_state.uploaded_files = []
        st.session_state.uploaded_file_names = current_filenames
        st.session_state.embeddings_created = False
        
        # Clear existing collection and create a new one
        try:
            if st.session_state.collection is not None:
                st.session_state.chroma_client.delete_collection(name=f"learning_docs_{st.session_state.session_id}")
            st.session_state.collection = st.session_state.chroma_client.create_collection(
                name=f"learning_docs_{st.session_state.session_id}",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            st.sidebar.error(f"Error resetting ChromaDB collection: {e}")
        
        # Extract text from each PDF and store in session state
        with st.spinner("Processing PDF files and creating embeddings..."):
            for pdf_file in uploaded_files:
                extracted_text = extract_pdf_text(pdf_file)
                if extracted_text:
                    st.session_state.extracted_texts.append({
                        "filename": pdf_file.name,
                        "text": extracted_text
                    })
                    st.session_state.uploaded_files.append(pdf_file)
                    
                    # Create chunks and embeddings for each document
                    chunks = chunk_text(extracted_text)
                    
                    for i, chunk in enumerate(chunks):
                        # Create embeddings for each chunk
                        embedding = create_embedding(chunk)
                        if embedding:
                            # Add document chunk with embedding to ChromaDB
                            try:
                                st.session_state.collection.add(
                                    embeddings=[embedding],
                                    documents=[chunk],
                                    metadatas=[{"source": pdf_file.name, "chunk_id": i}],
                                    ids=[f"{pdf_file.name}_chunk_{i}"]
                                )
                            except Exception as e:
                                st.error(f"Error adding to ChromaDB: {e}")
            
            st.session_state.embeddings_created = True
                    
        if st.session_state.extracted_texts:
            st.sidebar.success(f"✅ {len(st.session_state.extracted_texts)} PDF files processed successfully with vector embeddings!")
else:
    st.info("📥 Please enter your OpenAI API key and upload PDF files to begin.")

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

# Enhanced RAG function using vector search with ChromaDB
def generate_rag_answer(question, documents, course_content=None):
    try:
        if not openai_api_key:
            return "API key is required to generate answers."
        
        if not documents or not st.session_state.embeddings_created:
            return "Document embeddings are not available. Please process documents first."
        
        # Create embedding for the question
        question_embedding = create_embedding(question)
        if not question_embedding:
            return "Failed to create embedding for question."
        
        # Query ChromaDB for relevant chunks using the question embedding
        query_results = st.session_state.collection.query(
            query_embeddings=[question_embedding],
            n_results=5  # Get top 5 relevant chunks
        )
        
        # Extract relevant chunks from query results
        relevant_chunks = []
        if query_results and 'documents' in query_results and query_results['documents']:
            for doc, metadata in zip(query_results['documents'][0], query_results['metadatas'][0]):
                relevant_chunks.append({
                    "text": doc,
                    "source": metadata['source']
                })
        
        # Create a context from the most relevant chunks
        combined_context = ""
        for i, chunk in enumerate(relevant_chunks):
            combined_context += f"\nRelevant Document {i+1} ({chunk['source']}):\n{chunk['text']}\n"
        
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
        
        Relevant Document Chunks: {combined_context}
        
        Course Information: {course_context}
        
        Provide a comprehensive answer using information from the documents and course contents.
        If the question cannot be answered based on the provided information, say so politely.
        Reference specific documents when appropriate in your answer.
        """
        
        # Create OpenAI client correctly
        client = OpenAI(api_key=openai_api_key)
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
        if st.session_state.extracted_texts and st.session_state.embeddings_created:
            with st.spinner("Generating answer using vector search..."):
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
        # For course generation, use vector search to find the most relevant content
        if st.session_state.embeddings_created:
            # Create a query to summarize the entire document set
            query = f"Create a comprehensive course for {role}s focusing on {', '.join(learning_focus)}"
            query_embedding = create_embedding(query)
            
            # Get most relevant chunks for course creation
            query_results = st.session_state.collection.query(
                query_embeddings=[query_embedding],
                n_results=15  # Get more chunks for comprehensive course creation
            )
            
            # Combine relevant chunks for course generation
            combined_docs = ""
            if query_results and 'documents' in query_results and query_results['documents']:
                for doc, metadata in zip(query_results['documents'][0], query_results['metadatas'][0]):
                    combined_docs += f"\n--- FROM DOCUMENT: {metadata['source']} ---\n"
                    combined_docs += doc + "\n\n"
        else:
            # Fallback to using regular documents if embeddings aren't available
            combined_docs = ""
            for i, doc in enumerate(st.session_state.extracted_texts):
                doc_summary = f"\n--- DOCUMENT {i+1}: {doc['filename']} ---\n"
                doc_summary += doc['text'][:3000]  # Limit each doc to avoid token limits
                combined_docs += doc_summary + "\n\n"
        
        professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
        
        # Get a document summary first
        summary_query = "Create a comprehensive summary of these documents highlighting key concepts, theories, and practical applications across all materials."
        document_summary = generate_rag_answer(summary_query, st.session_state.extracted_texts)
        
        prompt = f"""
        Design a comprehensive professional learning course based on the multiple documents provided.
        Context: {professional_context}
        Document Summary: {document_summary}
        
        Document Contents: {combined_docs[:5000]}
        
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
            # Create OpenAI client correctly
            client = OpenAI(api_key=openai_api_key)
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
tab1, tab2, tab3, tab4 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources", "🔍 Vector Search"])

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
        1. Enter your OpenAI API key in the sidebar
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
        for i, query in enumerate(st.session_state.employer_queries):
            with st.expander(f"Question {i+1}: {query['question'][:50]}..." if len(query['question']) > 50 else f"Question {i+1}: {query['question']}"):
                st.write(f"**Question:** {query['question']}")
                
                if query['answered']:
                    st.write(f"**Answer:** {query['answer']}")
                else:
                    st.info("Generating answer...")
                    # Generate answer on-demand if not already answered
                    if st.session_state.extracted_texts and st.session_state.embeddings_created:
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
                        st.warning("No document embeddings created yet. Please upload documents to generate answers.")

with tab3:
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

with tab4:
    st.title("🔍 Vector Search")
    st.markdown("""
    This section allows you to search through your documents using vector embeddings 
    to find the most relevant information for your queries.
    """)
    
    # Only enable if embeddings are created
    if st.session_state.embeddings_created:
        search_query = st.text_input("Search Query:", placeholder="Enter your search query...")
        num_results = st.slider("Number of results to show:", min_value=1, max_value=10, value=3)
        
        if st.button("Search Documents") and search_query:
            with st.spinner("Searching documents with vector embeddings..."):
                # Create embedding for search query
                query_embedding = create_embedding(search_query)
                
                if query_embedding:
                    # Search ChromaDB for relevant chunks
                    query_results = st.session_state.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=num_results
                    )
                    
                    # Display search results
                    st.subheader("Search Results:")
                    
                    if query_results and 'documents' in query_results and query_results['documents']:
                        for i, (doc, metadata, distance) in enumerate(zip(
                            query_results['documents'][0], 
                            query_results['metadatas'][0],
                            query_results['distances'][0]
                        )):
                            relevancy_score = 1 - distance  # Convert distance to similarity score
                            with st.expander(f"Result {i+1} from {metadata['source']} (Relevance: {relevancy_score:.2f})"):
                                st.markdown(f"**Source:** {metadata['source']}")
                                st.markdown(f"**Chunk ID:** {metadata['chunk_id']}")
                                st.markdown(f"**Relevance Score:** {relevancy_score:.2f}")
                                st.markdown("**Content:**")
                                st.write(doc)
                    else:
                        st.info("No relevant results found in the documents.")
                else:
                    st.error("Failed to create embedding for search query.")
    else:
        st.with tab4:
    st.title("🔍 Vector Search")
    st.markdown("""
    This section allows you to search through your documents using vector embeddings 
    to find the most relevant information for your queries.
    """)
    
    # Only enable if embeddings are created
    if st.session_state.embeddings_created:
        search_query = st.text_input("Search Query:", placeholder="Enter your search query...")
        num_results = st.slider("Number of results to show:", min_value=1, max_value=10, value=3)
        
        if st.button("Search Documents") and search_query:
            with st.spinner("Searching documents with vector embeddings..."):
                # Create embedding for search query
                query_embedding = create_embedding(search_query)
                
                if query_embedding:
                    # Search ChromaDB for relevant chunks
                    query_results = st.session_state.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=num_results
                    )
                    
                    # Display search results
                    st.subheader("Search Results:")
                    
                    if query_results and 'documents' in query_results and query_results['documents']:
                        for i, (doc, metadata, distance) in enumerate(zip(
                            query_results['documents'][0], 
                            query_results['metadatas'][0],
                            query_results['distances'][0]
                        )):
                            relevancy_score = 1 - distance  # Convert distance to similarity score
                            with st.expander(f"Result {i+1} from {metadata['source']} (Relevance: {relevancy_score:.2f})"):
                                st.markdown(f"**Source:** {metadata['source']}")
                                st.markdown(f"**Chunk ID:** {metadata['chunk_id']}")
                                st.markdown(f"**Relevance Score:** {relevancy_score:.2f}")
                                st.markdown("**Content:**")
                                st.write(doc)
                    else:
                        st.info("No relevant results found in the documents.")
                else:
                    st.error("Failed to create embedding for search query.")
    else:
        st.warning("No document embeddings created yet. Please upload PDF files and process them first.")

    # Add a visual explanation of vector search
    with st.expander("ℹ️ How Vector Search Works"):
        st.markdown("""
        ### Understanding Vector Embeddings & Semantic Search

        **What are Vector Embeddings?**
        Vector embeddings convert text into numerical representations (vectors) that capture semantic meaning.
        Similar concepts have similar vector representations, even if they use different words.

        **How the Search Works:**
        1. **Document Processing:** Each document is split into chunks and converted to vector embeddings
        2. **Query Processing:** Your search query is converted to the same vector space
        3. **Similarity Matching:** The system finds chunks with vectors most similar to your query vector
        4. **Results Ranking:** Results are ranked by similarity score (cosine similarity)

        This approach enables finding conceptually related content even when exact keywords don't match.
        """)
        
        # Add a simple visualization of vector search
        st.markdown("""
        ```
        Document Space:                  Vector Space:                 Search Results:
        +-------------------+            +-------------------+         +-------------------+
        | PDF Documents     |    →       | Vector Embeddings |    →    | Ranked by         |
        | - Document 1      |            | [0.2, 0.8, ...]   |         | Relevance:        |
        | - Document 2      |            | [0.5, 0.3, ...]   |         | 1. Chunk from     |
        | - Document 3      |            | [0.1, 0.7, ...]   |         |    Document 2     |
        +-------------------+            +-------------------+         | 2. Chunk from     |
                                                ↑                      |    Document 1     |
                                                |                      | 3. Chunk from     |
                                         +-------------+               |    Document 3     |
                                         | Your Query  |               +-------------------+
                                         | [0.4, 0.3,..]|
                                         +-------------+
        ```
        """)

# Enhanced document similarity visualization
def show_document_similarity():
    st.title("📊 Document Similarity Analysis")
    st.markdown("""
    This section provides a visualization of how similar your uploaded documents are to each other.
    Documents that are plotted closer together have more similar content.
    """)
    
    if st.session_state.embeddings_created and len(st.session_state.uploaded_file_names) >= 2:
        with st.spinner("Analyzing document similarity..."):
            try:
                # Generate document-level embeddings for each full document
                document_embeddings = []
                document_names = []
                
                for doc in st.session_state.extracted_texts:
                    # Get embedding for the first 1000 characters of each document to represent it
                    doc_summary = doc['text'][:1000]
                    doc_embedding = create_embedding(doc_summary)
                    if doc_embedding:
                        document_embeddings.append(doc_embedding)
                        document_names.append(doc['filename'])
                
                if document_embeddings:
                    # Create a visualization using a scatter plot and PCA
                    from sklearn.decomposition import PCA
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.use('Agg')  # Use non-interactive backend
                    
                    # Perform PCA to reduce embedding dimensions to 2D
                    pca = PCA(n_components=2)
                    reduced_embeddings = pca.fit_transform(document_embeddings)
                    
                    # Create a scatter plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=100)
                    
                    # Add document names as labels
                    for i, name in enumerate(document_names):
                        # Truncate long filenames
                        display_name = name[:20] + '...' if len(name) > 20 else name
                        ax.annotate(display_name, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
                    
                    ax.set_title('Document Similarity Visualization')
                    ax.set_xlabel('Principal Component 1')
                    ax.set_ylabel('Principal Component 2')
                    ax.grid(True)
                    
                    # Display the plot in Streamlit
                    st.pyplot(fig)
                    
                    st.markdown("""
                    **How to interpret:** 
                    - Documents that appear closer together in this plot have more similar content
                    - Distance between points represents semantic difference between documents
                    - This visualization helps identify which documents cover similar topics
                    """)
                else:
                    st.warning("Could not generate document embeddings for visualization.")
            except Exception as e:
                st.error(f"Error in similarity visualization: {e}")
    else:
        st.info("Upload at least 2 documents and process them to see similarity analysis.")

# Add a fifth tab for document similarity analysis
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources", "🔍 Vector Search", "📊 Document Similarity"])

# In the fifth tab, show document similarity visualization
with tab5:
    show_document_similarity()

# Run the app
if __name__ == "__main__":
    pass  # Streamlit already executes all code
