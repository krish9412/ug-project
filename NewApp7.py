import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from openai import OpenAI
from langchain.chat_models import ChatOpenAI

# App configuration
st.set_page_config(page_title="🏢 Employee Training System", layout="wide")

# Initialize session state variables
if 'learning_materials' not in st.session_state:
    st.session_state.learning_materials = None
if 'materials_created' not in st.session_state:
    st.session_state.materials_created = False
if 'creation_in_progress' not in st.session_state:
    st.session_state.creation_in_progress = False
if 'answered_quiz_items' not in st.session_state:
    st.session_state.answered_quiz_items = set()
if 'quiz_item_count' not in st.session_state:
    st.session_state.quiz_item_count = 0
if 'document_contents' not in st.session_state:
    st.session_state.document_contents = []
if 'stakeholder_questions' not in st.session_state:
    st.session_state.stakeholder_questions = []
if 'user_session_id' not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'processed_document_names' not in st.session_state:
    st.session_state.processed_document_names = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []

# Sidebar setup
st.sidebar.title("🎓 Employee Training System")

# Application reset functionality
if st.sidebar.button("🔄 Clear All Data"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.user_session_id = str(uuid.uuid4())
    st.session_state.document_contents = []
    st.session_state.processed_documents = []
    st.session_state.processed_document_names = []
    st.session_state.chat_history = []
    st.session_state.chunks = []
    st.rerun()

# API credentials input
openai_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key if openai_api_key else ""

# Document uploader
uploaded_files = st.sidebar.file_uploader("📝 Upload Training Materials (PDF)", type=['pdf'], accept_multiple_files=True)

# Helper functions
def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the provided tokenizer"""
    return len(tokenizer.encode(text))

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using pdfplumber"""
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            full_text = ""
            for page in pdf.pages:
                page_content = page.extract_text() or ""
                full_text += page_content + "\n"
        return full_text
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return ""

def process_documents(files):
    """Process documents and create chunks for RAG"""
    document_contents = []
    processed_documents = []
    processed_document_names = [file.name for file in files]
    chunks = []
    
    # Initialize tokenizer for token counting
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except:
        st.warning("Failed to load GPT-2 tokenizer. Using approximate token count.")
        tokenizer = None
    
    # Create text splitter with token counting if available
    if tokenizer:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=24,
            length_function=lambda text: count_tokens(text, tokenizer),
        )
    else:
        # Fallback to character-based splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
        )
    
    # Process each file
    for pdf_file in files:
        # Extract text from PDF
        text_content = extract_text_from_pdf(pdf_file)
        
        if text_content:
            # Save document content
            document_contents.append({
                "filename": pdf_file.name,
                "text": text_content
            })
            processed_documents.append(pdf_file)
            
            # Create document chunks for embedding
            doc_chunks = text_splitter.create_documents([text_content], metadatas=[{"source": pdf_file.name}])
            chunks.extend(doc_chunks)
    
    return document_contents, processed_documents, processed_document_names, chunks

def create_vectorstore(chunks):
    """Create a vector store from document chunks using OpenAI embeddings"""
    if not openai_api_key or not chunks:
        return None
    
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def create_qa_chain(vectorstore):
    """Create a question-answering chain with conversation history"""
    if not openai_api_key or not vectorstore:
        return None
    
    try:
        # Create OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Use ChatOpenAI instead of OpenAI
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, client=client)
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

# Process uploaded PDF files and create RAG components
if uploaded_files and openai_api_key:
    # Check if file list has changed
    current_files = [file.name for file in uploaded_files]
    if current_files != st.session_state.processed_document_names:
        with st.spinner("Processing documents and building knowledge base..."):
            # Process documents
            document_contents, processed_documents, processed_document_names, chunks = process_documents(uploaded_files)
            
            # Update session state
            st.session_state.document_contents = document_contents
            st.session_state.processed_documents = processed_documents
            st.session_state.processed_document_names = processed_document_names
            st.session_state.chunks = chunks
            
            # Create vector store
            vectorstore = create_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore
            
            # Create QA chain
            if vectorstore:
                qa_chain = create_qa_chain(vectorstore)
                st.session_state.qa_chain = qa_chain
                
        if st.session_state.document_contents and st.session_state.vectorstore:
            st.sidebar.success(f"✅ {len(st.session_state.document_contents)} PDFs processed and indexed!")
else:
    st.info("📥 Enter your OpenAI API key and upload PDF files to begin.")

# AI model and user context configuration
model_selection = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
selected_model = st.sidebar.selectbox("Choose AI Model", model_selection, index=0)

job_roles = ["Manager", "Executive", "Developer", "Designer", "Marketer", "Human Resources", "Other", "Fresher"]
selected_role = st.sidebar.selectbox("Employee Role", job_roles)

learning_areas = ["Leadership", "Technical Skills", "Communication", "Project Management", "Innovation", "Team Building", "Finance"]
selected_areas = st.sidebar.multiselect("Training Priorities", learning_areas)

# Display document list in sidebar
if st.session_state.processed_document_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Processed Documents")
    for i, filename in enumerate(st.session_state.processed_document_names):
        st.sidebar.text(f"{i+1}. {filename}")

# RAG query function
def perform_rag_query(question, history=None):
    """Perform a RAG query using the QA chain"""
    if not openai_api_key:
        return "API key required to generate responses.", []
    
    if not st.session_state.qa_chain:
        return "Please upload and process documents first to build the knowledge base.", []
    
    try:
        # Use conversation history if available
        chat_history = history if history is not None else []
        
        # Call the QA chain
        result = st.session_state.qa_chain({"question": question, "chat_history": chat_history})
        
        # Extract answer and source documents
        answer = result['answer']
        source_docs = result.get('source_documents', [])
        
        return answer, source_docs
    except Exception as e:
        return f"Error generating answer: {str(e)}", []

# Create course content function with RAG
def create_course_content():
    """Generate course content using RAG and OpenAI"""
    try:
        if not openai_api_key or not st.session_state.document_contents:
            return
        
        # Generate document summary first using RAG
        summary_question = "Create a comprehensive summary of all the training documents highlighting key concepts, theories, and practical applications."
        materials_summary, _ = perform_rag_query(summary_question)
        
        # Learner context
        learner_context = f"Role: {selected_role}, Focus: {', '.join(selected_areas)}"
        
        # Course prompt for OpenAI
        course_prompt = f"""
        Design a comprehensive employee training course based on the provided document summary.
        Context: {learner_context}
        Document Summary: {materials_summary}
        
        Create an engaging, thorough and well-structured course by:
        1. Creating an inspiring course title that reflects the integrated knowledge from all documents
        2. Writing a detailed course description (at least 300 words) that explains how the course benefits employees
        3. Developing 3-5 comprehensive modules that build upon each other in a logical sequence
        4. Providing 3-4 clear learning objectives for each module with specific examples and practical applications
        5. Creating detailed content for each module (at least the key points) including:
           - Real-world examples and case studies
           - Practical applications of concepts
           - Step-by-step guides for complex procedures
        6. Including a quiz with 2-3 thought-provoking questions per module

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
        """
        
        try:
            # Use OpenAI to generate course content
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": course_prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            # Process the response
            response_content = response.choices[0].message.content
            
            try:
                # Parse JSON response
                st.session_state.learning_materials = json.loads(response_content)
                st.session_state.materials_created = True
                
                # Count total quiz questions
                question_count = 0
                for module in st.session_state.learning_materials.get("modules", []):
                    quiz = module.get("quiz", {})
                    question_count += len(quiz.get("questions", []))
                st.session_state.quiz_item_count = question_count
                
            except json.JSONDecodeError as e:
                st.error(f"JSON parsing error: {e}")
                st.text(response_content)
        
        except Exception as e:
            st.error(f"OpenAI API Error: {e}")
            st.error("Please verify your API key and model selection.")
            
    except Exception as e:
        st.error(f"Error: {e}")
    
    # Reset generation flag
    st.session_state.creation_in_progress = False

# Course generation trigger function
def initiate_course_generation():
    # Set generation flag
    st.session_state.creation_in_progress = True
    st.session_state.materials_created = False
    st.rerun()  # Trigger UI update to show loading state

# Quiz answer verification function
def verify_answer(question_id, student_answer, solution):
    if student_answer == solution:
        st.success("🎉 Correct! Great job!")
        # Add to completed questions
        st.session_state.answered_quiz_items.add(question_id)
        return True
    else:
        st.error(f"Not quite right. The correct answer is: {solution}")
        return False

# Main content area with tabbed interface
tab1, tab2, tab3, tab4 = st.tabs(["📚 Training Modules", "💬 RAG Chat", "❓ FAQ Portal", "📑 Source Materials"])

# Check if course generation is in progress
if st.session_state.creation_in_progress:
    with st.spinner("Building your custom training path from source materials..."):
        # Reset progress tracking
        st.session_state.answered_quiz_items = set()
        create_course_content()
        st.success("✅ Your Training Course is Ready!")
        st.rerun()  # Refresh UI

with tab1:
    # Display course content if available
    if st.session_state.materials_created and st.session_state.learning_materials:
        course = st.session_state.learning_materials
        
        # Course header
        st.title(f"🌟 {course.get('course_title', 'Employee Training Course')}")
        st.markdown(f"*Customized for {selected_role}s focusing on {', '.join(selected_areas)}*")
        st.write(course.get('course_description', 'A structured learning path to enhance your professional capabilities.'))
        
        # Progress tracking
        completed = len(st.session_state.answered_quiz_items)
        total = st.session_state.quiz_item_count
        progress_percent = (completed / total * 100) if total > 0 else 0
        
        st.progress(progress_percent / 100)
        st.write(f"**Progress:** {completed}/{total} questions answered correctly ({progress_percent:.1f}%)")
        
        st.markdown("---")
        st.subheader("📋 Course Structure")
        
        # Display module list
        modules = course.get("modules", [])
        if modules:
            module_titles = [module.get('title', f'Module {i+1}') for i, module in enumerate(modules)]
            for i, title in enumerate(module_titles, 1):
                st.write(f"**Module {i}:** {title}")
        else:
            st.warning("No modules found in the course content.")
        
        st.markdown("---")
        
        # Display detailed module content
        for i, module in enumerate(modules, 1):
            title = module.get('title', f'Module {i}')
            with st.expander(f"📚 Module {i}: {title}"):
                # Learning objectives
                st.markdown("### 🎯 Learning Objectives:")
                objectives = module.get('learning_objectives', [])
                if objectives:
                    for obj in objectives:
                        st.markdown(f"- {obj}")
                else:
                    st.write("No learning objectives specified.")
                
                # Module content with formatting
                st.markdown("### 📖 Module Content:")
                content = module.get('content', 'No content available for this module.')
                
                # Format paragraphs properly
                paragraphs = content.split('\n\n')
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
                        st.write("")  # Add spacing
                
                # Key insights
                st.markdown("### 💡 Key Takeaways:")
                st.info("This module provides practical skills directly applicable to your work context.")
                
                # Quiz section
                st.markdown("### 📝 Knowledge Check:")
                quiz = module.get('quiz', {})
                questions = quiz.get('questions', [])
                
                if questions:
                    for q_idx, q in enumerate(questions, 1):
                        question_id = f"module_{i}_question_{q_idx}"
                        question_text = q.get('question', f'Question {q_idx}')
                        
                        # Create quiz container
                        quiz_box = st.container()
                        with quiz_box:
                            st.markdown(f"**Question {q_idx}:** {question_text}")
                            
                            options = q.get('options', [])
                            if options:
                                # Create radio button for answers
                                option_id = f"quiz_{i}_{q_idx}"
                                user_answer = st.radio("Select your answer:", options, key=option_id)
                                
                                # Create submit button
                                submit_id = f"submit_{i}_{q_idx}"
                                
                                # Show completion status
                                if question_id in st.session_state.answered_quiz_items:
                                    st.success("✓ Question completed")
                                else:
                                    if st.button(f"Check Answer", key=submit_id):
                                        correct_answer = q.get('correct_answer', '')
                                        verify_answer(question_id, user_answer, correct_answer)
                            else:
                                st.write("No options available for this question.")
                        
                        st.markdown("---")
                else:
                    st.write("No quiz questions available for this module.")

    else:
        # Welcome screen
        st.title("Welcome to Employee Training System")
        st.markdown("""
        ## Transform employee development with AI-powered training
        
        Upload PDF training materials, and our system will create a comprehensive, integrated training course with RAG capabilities!
        
        ### How it works:
        1. Enter your OpenAI API key in the sidebar
        2. Select employee role and training focus areas
        3. Upload PDF documents related to the training topics
        4. Click "Generate Training Course" to create personalized learning journeys
        
        Enhance employee skills and accelerate professional growth!
        """)
        
        # Course generation button
        if st.session_state.document_contents and openai_api_key and not st.session_state.creation_in_progress:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Create Training Course", use_container_width=True):
                    initiate_course_generation()
        elif st.session_state.creation_in_progress:
            st.info("Building personalized training course... Please wait.")

# RAG Chat tab - Interactive RAG chat interface
with tab2:
    st.title("💬 RAG Interactive Training Assistant")
    st.markdown("""
    Ask questions about the training materials, and get answers powered by RAG (Retrieval-Augmented Generation).
    The system will retrieve relevant information from the uploaded documents and generate accurate responses.
    """)
    
    # Initialize chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: #e0e5f2; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><b>Assistant:</b> {message['content']}</div>", unsafe_allow_html=True)
                if 'sources' in message and message['sources']:
                    with st.expander("View Sources"):
                        for i, source in enumerate(message['sources']):
                            st.markdown(f"**Source {i+1}:** {source}")
    
    # User input
    user_question = st.text_input("Ask a question about the training materials:", key="rag_question")
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Send", key="send_rag")
    with col2:
        clear_chat = st.button("Clear Chat History", key="clear_rag_chat")
    
    # Process user input
    if send_button and user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Get context-relevant chat history
        recent_history = [(q["content"], a["content"]) for q, a in zip(
            st.session_state.chat_history[::2], 
            st.session_state.chat_history[1::2]
        )] if len(st.session_state.chat_history) > 1 else []
        
        # Get answer using RAG
        with st.spinner("Generating answer..."):
            answer, sources = perform_rag_query(user_question, recent_history)
            
            source_texts = []
            for doc in sources:
                source_texts.append(f"{doc.metadata.get('source', 'Unknown')}: {doc.page_content[:150]}...")
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "sources": source_texts
            })
        
        # Rerun to update UI
        st.rerun()
    
    # Clear chat history if requested
    if clear_chat:
        st.session_state.chat_history = []
        st.rerun()
    
    # Show help text if no chat history
    if not st.session_state.chat_history:
        st.info("Start by asking a question about the training materials. The system will use RAG to provide accurate answers based on the uploaded documents.")

# FAQ Portal tab
with tab3:
    st.title("❓ Training FAQ Portal")
    st.markdown("""
    This section allows you to submit questions and get AI-generated answers about the training content or related topics.
    Submit your questions below, and our AI will generate answers based on the uploaded documents.
    """)
    
    # Question submission form
    new_question = st.text_area("Submit a new question:", height=100, key="faq_question")
    if st.button("Submit Question", key="submit_faq"):
        if new_question:
            # Generate answer if documents are available
            response = ""
            sources = []
            if st.session_state.document_contents and st.session_state.qa_chain:
                with st.spinner("Generating answer..."):
                    response, source_docs = perform_rag_query(new_question)
                    sources = [f"{doc.metadata.get('source', 'Unknown')}" for doc in source_docs]
            else:
                response = "Please upload and process documents first to enable question answering."
            
            st.session_state.stakeholder_questions.append({
                "question": new_question,
                "answer": response,
                "sources": sources,
                "answered": bool(response)
            })
            st.success("Question submitted and answered!")
            st.rerun()
    
    # Display FAQ questions
    if not st.session_state.stakeholder_questions:
        st.info("No questions submitted yet. Add a question to begin building the FAQ.")
    else:
        for i, query in enumerate(st.session_state.stakeholder_questions):
            with st.expander(f"Question {i+1}: {query['question'][:50]}..." if len(query['question']) > 50 else f"Question {i+1}: {query['question']}"):
                st.write(f"**Question:** {query['question']}")
                
                if query['answered']:
                    st.write(f"**Answer:** {query['answer']}")
                    if query.get('sources'):
                        st.markdown("**Sources:**")
                        for src in query['sources']:
                            st.markdown(f"- {src}")
                else:
                    st.info("Generating answer...")
                    # Generate answer on-demand
                    if st.session_state.document_contents and st.session_state.qa_chain:
                        try:
                            answer, source_docs = perform_rag_query(query['question'])
                            sources = [f"{doc.metadata.get('source', 'Unknown')}" for doc in source_docs]
                            
                            st.session_state.stakeholder_questions[i]['answer'] = answer
                            st.session_state.stakeholder_questions[i]['sources'] = sources
                            st.session_state.stakeholder_questions[i]['answered'] = True
                            st.rerun()
                        except Exception as e:
                            error_msg = f"Error generating answer: {str(e)}. Please try resetting the application."
                            st.error(error_msg)
                            st.session_state.stakeholder_questions[i]['answer'] = error_msg
                            st.session_state.stakeholder_questions[i]['answered'] = True
                    else:
                        st.warning("No documents available. Please upload documents to generate answers.")

# Source Materials tab
with tab4:
    st.title("📑 Training Source Materials")
    
    if not st.session_state.document_contents:
        st.info("No documents uploaded yet. Please upload PDF files in the sidebar to view content.")
    else:
        st.write(f"**{len(st.session_state.document_contents)} documents processed:**")
        
        for i, doc in enumerate(st.session_state.document_contents):
            with st.expander(f"Document {i+1}: {doc['filename']}"):
                # Show document preview
                preview = doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text']
                st.markdown("### Document Preview:")
                st.text_area("Content:", value=preview, height=300, disabled=True)
                
                # Generate document summary
                if st.button(f"Summarize {doc['filename']}", key=f"sum_{i}"):
                    with st.spinner("Creating document summary..."):
                        summary_question = f"Create a comprehensive summary of this document: {doc['filename']}"
                        summary, _ = perform_rag_query(summary_question)
                        st.markdown("### AI-Generated Summary:")
                        st.write(summary)
                        
                # Document insights
                if st.button(f"Key Insights from {doc['filename']}", key=f"insights_{i}"):
                    with st.spinner("Extracting key insights..."):
                        insights_question = f"What are the 5 most important insights or takeaways from this document: {doc['filename']}"
                        insights, _ = perform_rag_query(insights_question)
                        st.markdown("### Key Insights:")
                        st.write(insights)

# Footer
st.markdown("---")
st.markdown("*Employee Training System with RAG (Retrieval-Augmented Generation) | Built with Streamlit & LangChain*")
