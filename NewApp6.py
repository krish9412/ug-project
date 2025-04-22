import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
from openai import OpenAI 

# App configuration
st.set_page_config(page_title="📚 Learning Development Hub", layout="wide")

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

# Sidebar setup
st.sidebar.title("🎓 Professional Development System")

# Application reset functionality
if st.sidebar.button("🔄 Clear All Data"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.user_session_id = str(uuid.uuid4())
    st.session_state.document_contents = []
    st.session_state.processed_documents = []
    st.session_state.processed_document_names = []
    st.rerun()

# API credentials input
openai_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

# Document uploader
uploaded_files = st.sidebar.file_uploader("📝 Upload Learning Materials (PDF)", type=['pdf'], accept_multiple_files=True)

# PDF text extraction function
def extract_text_from_pdf(pdf_file):
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

# Process uploaded PDF files
if uploaded_files and openai_api_key:
    # Check if file list has changed
    current_files = [file.name for file in uploaded_files]
    if current_files != st.session_state.processed_document_names:
        st.session_state.document_contents = []
        st.session_state.processed_documents = []
        st.session_state.processed_document_names = current_files
        
        # Process each PDF file
        with st.spinner("Reading PDF documents..."):
            for pdf_file in uploaded_files:
                text_content = extract_text_from_pdf(pdf_file)
                if text_content:
                    st.session_state.document_contents.append({
                        "filename": pdf_file.name,
                        "text": text_content
                    })
                    st.session_state.processed_documents.append(pdf_file)
                    
        if st.session_state.document_contents:
            st.sidebar.success(f"✅ {len(st.session_state.document_contents)} PDFs successfully processed!")
else:
    st.info("📥 Enter your OpenAI API key and upload PDF files to begin.")

# AI model and user context configuration
model_selection = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Choose AI Model", model_selection, index=0)

job_roles = ["Manager", "Executive", "Developer", "Designer", "Marketer", "Human Resources", "Other", "Fresher"]
selected_role = st.sidebar.selectbox("Your Professional Role", job_roles)

learning_areas = ["Leadership", "Technical Skills", "Communication", "Project Management", "Innovation", "Team Building", "Finance"]
selected_areas = st.sidebar.multiselect("Learning Priorities", learning_areas)

# Display document list in sidebar
if st.session_state.processed_document_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Processed Documents")
    for i, filename in enumerate(st.session_state.processed_document_names):
        st.sidebar.text(f"{i+1}. {filename}")

# Retrieval-augmented generation function
def perform_rag_query(question, documents, course_data=None):
    try:
        if not openai_api_key:
            return "API key required to generate responses."
        
        if not documents:
            return "Please upload and process documents first."
            
        # Build context from document texts
        context_data = ""
        for i, doc in enumerate(documents[:3]):  # Limit to first 3 docs
            doc_excerpt = doc["text"][:2000]  # Limit each doc to 2000 chars
            context_data += f"\nDocument {i+1} ({doc['filename']}):\n{doc_excerpt}\n"
        
        # Include course data if available
        learning_context = ""
        if course_data:
            learning_context = f"""
            Course Title: {course_data.get('course_title', '')}
            Course Description: {course_data.get('course_description', '')}
            
            Module Overview:
            """
            for i, module in enumerate(course_data.get('modules', []), 1):
                learning_context += f"""
                Module {i}: {module.get('title', '')}
                Learning Objectives: {', '.join(module.get('learning_objectives', []))}
                Content Preview: {module.get('content', '')[:200]}...
                """
        
        prompt = f"""
        As an AI assistant for professional learning, answer the following question 
        based on the provided document content. Be specific, accurate, and helpful.
        
        Question: {question}
        
        Document Content: {context_data}
        
        Course Information: {learning_context}
        
        Provide a comprehensive answer using information from the documents and course contents.
        If the question cannot be answered based on the provided information, say so politely.
        Reference specific documents when appropriate in your answer.
        """
        
        # Create OpenAI client and generate response
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Stakeholder question section
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Learning Questions")

new_question = st.sidebar.text_area("Add question:", height=100)
if st.sidebar.button("Submit Question"):
    if new_question:
        # Generate answer if documents are available
        response = ""
        if st.session_state.document_contents:
            with st.spinner("Generating answer..."):
                response = perform_rag_query(
                    new_question, 
                    st.session_state.document_contents,
                    st.session_state.learning_materials if st.session_state.materials_created else None
                )
        else:
            response = "Please upload and process documents first to enable question answering."
        
        st.session_state.stakeholder_questions.append({
            "question": new_question,
            "answer": response,
            "answered": bool(response)
        })
        st.sidebar.success("Question submitted and answered!")
        st.rerun()

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

# Course generation trigger function
def initiate_course_generation():
    # Set generation flag
    st.session_state.creation_in_progress = True
    st.session_state.materials_created = False
    st.rerun()  # Trigger UI update to show loading state

# Course content generation function
def create_course_content():
    try:
        # Combine document texts for processing
        document_data = ""
        for i, doc in enumerate(st.session_state.document_contents):
            doc_excerpt = f"\n--- DOCUMENT {i+1}: {doc['filename']} ---\n"
            doc_excerpt += doc['text'][:3000]  # Limit to avoid token limits
            document_data += doc_excerpt + "\n\n"
        
        learner_context = f"Role: {selected_role}, Focus: {', '.join(selected_areas)}"
        
        # Generate document summary first
        summary_prompt = "Create a comprehensive summary of these documents highlighting key concepts, theories, and practical applications across all materials."
        materials_summary = perform_rag_query(summary_prompt, st.session_state.document_contents)
        
        course_prompt = f"""
        Design a comprehensive professional learning course based on the multiple documents provided.
        Context: {learner_context}
        Document Summary: {materials_summary}
        
        Document Contents: {document_data[:5000]}
        
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

# Main content area with tabbed interface
tab1, tab2, tab3 = st.tabs(["📚 Learning Modules", "❓ Q&A Portal", "📑 Source Materials"])

# Check if course generation is in progress
if st.session_state.creation_in_progress:
    with st.spinner("Building your custom learning path from source materials..."):
        # Reset progress tracking
        st.session_state.answered_quiz_items = set()
        create_course_content()
        st.success("✅ Your Learning Path is Ready!")
        st.rerun()  # Refresh UI

with tab1:
    # Display course content if available
    if st.session_state.materials_created and st.session_state.learning_materials:
        course = st.session_state.learning_materials
        
        # Course header
        st.title(f"🌟 {course.get('course_title', 'Professional Development Course')}")
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
                st.info("This module provides practical skills directly applicable to your professional context.")
                
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
        st.title("Welcome to Professional Learning Platform")
        st.markdown("""
        ## Transform your professional development with AI-powered learning
        
        Upload multiple PDF documents, and our system will create a comprehensive, integrated learning course just for you!
        
        ### How it works:
        1. Enter your OpenAI API key in the sidebar
        2. Select your professional role and learning focus
        3. Upload multiple PDF documents related to your area of interest
        4. Click "Generate Course" to create your personalized learning journey
        
        Enhance your skills and accelerate your professional growth!
        """)
        
        # Course generation button
        if st.session_state.document_contents and openai_api_key and not st.session_state.creation_in_progress:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Create My Learning Path", use_container_width=True):
                    initiate_course_generation()
        elif st.session_state.creation_in_progress:
            st.info("Building your personalized learning path... Please wait.")

with tab2:
    st.title("💬 Learning Questions")
    st.markdown("""
    This section allows you to ask questions and get AI-generated answers about the course content or related topics.
    Submit your questions in the sidebar, and our AI will generate answers based on the uploaded documents.
    """)
    
    if not st.session_state.stakeholder_questions:
        st.info("No questions submitted yet. Add a question in the sidebar to begin.")
    else:
        for i, query in enumerate(st.session_state.stakeholder_questions):
            with st.expander(f"Question {i+1}: {query['question'][:50]}..." if len(query['question']) > 50 else f"Question {i+1}: {query['question']}"):
                st.write(f"**Question:** {query['question']}")
                
                if query['answered']:
                    st.write(f"**Answer:** {query['answer']}")
                else:
                    st.info("Generating answer...")
                    # Generate answer on-demand
                    if st.session_state.document_contents:
                        try:
                            answer = perform_rag_query(
                                query['question'], 
                                st.session_state.document_contents,
                                st.session_state.learning_materials if st.session_state.materials_created else None
                            )
                            st.session_state.stakeholder_questions[i]['answer'] = answer
                            st.session_state.stakeholder_questions[i]['answered'] = True
                            st.rerun()
                        except Exception as e:
                            error_msg = f"Error generating answer: {str(e)}. Please try resetting the application."
                            st.error(error_msg)
                            st.session_state.stakeholder_questions[i]['answer'] = error_msg
                            st.session_state.stakeholder_questions[i]['answered'] = True
                    else:
                        st.warning("No documents available. Please upload documents to generate answers.")

with tab3:
    st.title("📑 Source Materials")
    
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
                        summary_request = f"Create a comprehensive summary of this document highlighting key concepts, theories, and practical applications:"
                        summary = perform_rag_query(summary_request, [doc])
                        st.markdown("### AI-Generated Summary:")
                        st.write(summary)
