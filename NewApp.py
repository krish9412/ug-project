import streamlit as st
import os
import tempfile
import json
import pdfplumber
import uuid
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# Application configuration
st.set_page_config(page_title="🧠 Learning Ecosystem Platform", layout="wide")

# Initialize session state variables
def initialize_session_state():
    if 'learning_materials' not in st.session_state:
        st.session_state.learning_materials = None
    if 'curriculum_ready' not in st.session_state:
        st.session_state.curriculum_ready = False
    if 'building_curriculum' not in st.session_state:
        st.session_state.building_curriculum = False
    if 'answered_quiz_items' not in st.session_state:
        st.session_state.answered_quiz_items = set()
    if 'quiz_items_count' not in st.session_state:
        st.session_state.quiz_items_count = 0
    if 'document_contents' not in st.session_state:
        st.session_state.document_contents = []
    if 'stakeholder_questions' not in st.session_state:
        st.session_state.stakeholder_questions = []
    if 'user_identifier' not in st.session_state:
        st.session_state.user_identifier = str(uuid.uuid4())
    if 'pdf_documents' not in st.session_state:
        st.session_state.pdf_documents = []
    if 'pdf_document_names' not in st.session_state:
        st.session_state.pdf_document_names = []
    if 'vector_db_client' not in st.session_state:
        st.session_state.vector_db_client = None
    if 'vector_collection' not in st.session_state:
        st.session_state.vector_collection = None

# Initialize session
initialize_session_state()

# Sidebar design and functionality
st.sidebar.title("🧠 Learning Ecosystem Platform")

# Reset functionality
if st.sidebar.button("♻️ Clean Session Data"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.user_identifier = str(uuid.uuid4())
    st.session_state.document_contents = []
    st.session_state.pdf_documents = []
    st.session_state.pdf_document_names = []
    st.rerun()

# API Key input for OpenAI
openai_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password", 
                                help="Enter your OpenAI API key to enable content generation")

# File uploader for learning materials
uploaded_pdfs = st.sidebar.file_uploader("📚 Upload Learning Materials", 
                                        type=['pdf'], 
                                        accept_multiple_files=True,
                                        help="Upload PDF files containing learning materials")

# PDF text extraction functionality
def parse_pdf_content(pdf_document):
    try:
        pdf_document.seek(0)
        with pdfplumber.open(pdf_document) as pdf:
            full_text = ""
            for page in pdf.pages:
                extracted_text = page.extract_text() or ""
                full_text += extracted_text + "\n"
        return full_text
    except Exception as e:
        st.error(f"Failed to extract PDF content: {e}")
        return ""

# Text segmentation for vector database
def segment_document(document_text, segment_size=1000, segment_overlap=200):
    segments = []
    current_position = 0
    text_total_length = len(document_text)
    
    while current_position < text_total_length:
        segment_end = min(current_position + segment_size, text_total_length)
        # Create overlap with previous segment
        segment_start = max(0, current_position - segment_overlap) if current_position > 0 else current_position
        segments.append(document_text[segment_start:segment_end])
        current_position = segment_end
    
    return segments

# Setup vector database
def setup_vector_database():
    try:
        # Create persistent storage location
        temp_storage_path = os.path.join(tempfile.gettempdir(), f"vector_db_{st.session_state.user_identifier}")
        os.makedirs(temp_storage_path, exist_ok=True)
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=temp_storage_path)
        
        # Configure OpenAI embedding function
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small"
        )
        
        # Create or retrieve collection
        collection = client.get_or_create_collection(
            name=f"learning_materials_{st.session_state.user_identifier}",
            embedding_function=embedding_function
        )
        
        # Store in session state
        st.session_state.vector_db_client = client
        st.session_state.vector_collection = collection
        
        return True
    except Exception as e:
        st.error(f"Vector database initialization failed: {e}")
        return False

# Process uploaded PDFs
if uploaded_pdfs and openai_key:
    # Check if files have changed since last upload
    current_files = [file.name for file in uploaded_pdfs]
    if current_files != st.session_state.pdf_document_names:
        # Reset previous data
        st.session_state.document_contents = []
        st.session_state.pdf_documents = []
        st.session_state.pdf_document_names = current_files
        
        # Initialize vector database
        if setup_vector_database():
            # Process each PDF
            with st.spinner("Processing PDF documents and creating semantic index..."):
                for pdf_file in uploaded_pdfs:
                    document_text = parse_pdf_content(pdf_file)
                    if document_text:
                        st.session_state.document_contents.append({
                            "filename": pdf_file.name,
                            "text": document_text
                        })
                        st.session_state.pdf_documents.append(pdf_file)
                        
                        # Segment text and add to vector database
                        text_segments = segment_document(document_text)
                        
                        for idx, segment in enumerate(text_segments):
                            # Create unique identifier for segment
                            segment_id = f"{pdf_file.name.replace(' ', '_').replace('.', '_')}_{idx}"
                            
                            # Add to vector collection
                            st.session_state.vector_collection.add(
                                documents=[segment],
                                metadatas=[{"source": pdf_file.name, "segment_number": idx}],
                                ids=[segment_id]
                            )
                
            if st.session_state.document_contents:
                st.sidebar.success(f"✅ Successfully processed {len(st.session_state.document_contents)} documents")
else:
    st.info("🔍 Please provide your OpenAI API key and upload PDF files to begin.")

# Model selection and user profile
ai_model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_ai_model = st.sidebar.selectbox("AI Model Selection", ai_model_options, index=0)

profession_options = ["Developer", "Manager", "Executive", "Designer", "Marketer", "HR Professional", "Newcomer", "Other"]
professional_role = st.sidebar.selectbox("Your Professional Role", profession_options)

knowledge_areas = ["Technical Expertise", "Leadership", "Communication", "Project Management", "Creative Thinking", "Team Collaboration", "Financial Skills"]
learning_areas = st.sidebar.multiselect("Learning Priorities", knowledge_areas)

# Display uploaded files in sidebar
if st.session_state.pdf_document_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📑 Your Documents")
    for idx, filename in enumerate(st.session_state.pdf_document_names):
        st.sidebar.text(f"{idx+1}. {filename}")

# AI-assisted query function using vector search
def retrieve_contextual_answer(query_text, source_documents=None, curriculum_data=None):
    try:
        if not openai_key:
            return "API key required for answer generation."
        
        if not st.session_state.vector_collection:
            return "Document index not available. Please process documents first."
        
        # Search vector database for relevant content
        search_results = st.session_state.vector_collection.query(
            query_texts=[query_text],
            n_results=5
        )
        
        # Compile context from retrieved segments
        retrieved_context = ""
        if search_results and 'documents' in search_results and search_results['documents']:
            for idx, (doc_segment, metadata) in enumerate(zip(search_results['documents'][0], search_results['metadatas'][0])):
                doc_source = metadata.get('source', 'Unknown source')
                retrieved_context += f"\nSource: {doc_source}\n{doc_segment}\n"
        
        # Add curriculum context if available
        curriculum_context = ""
        if curriculum_data:
            curriculum_context = f"""
            Curriculum Title: {curriculum_data.get('course_title', '')}
            Description: {curriculum_data.get('course_description', '')}
            
            Module Structure:
            """
            for idx, module in enumerate(curriculum_data.get('modules', []), 1):
                curriculum_context += f"""
                Module {idx}: {module.get('title', '')}
                Objectives: {', '.join(module.get('learning_objectives', []))}
                Overview: {module.get('content', '')[:200]}...
                """
        
        # Construct AI prompt
        ai_prompt = f"""
        You are an advanced educational assistant for a professional learning platform.
        Please answer the following question using information from the provided resources.
        Provide detailed, accurate, and practical insights.
        
        Question: {query_text}
        
        Reference Materials: {retrieved_context}
        
        Curriculum Information: {curriculum_context}
        
        Provide a comprehensive answer based on the reference materials and curriculum.
        If the information is not available in the provided resources, acknowledge this limitation.
        Reference specific documents when appropriate to support your answer.
        """
        
        # Generate AI response
        ai_client = OpenAI(api_key=openai_key)
        response = ai_client.chat.completions.create(
            model=selected_ai_model,
            messages=[{"role": "user", "content": ai_prompt}],
            temperature=0.5
        )
        
        # Return generated response
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Stakeholder questions section
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Ask Questions")

stakeholder_query = st.sidebar.text_area("Enter your question:", height=100)
if st.sidebar.button("🔍 Get Answer"):
    if stakeholder_query:
        # Generate AI response if documents are available
        response_text = ""
        if st.session_state.vector_collection:
            with st.spinner("Analyzing documents and generating answer..."):
                response_text = retrieve_contextual_answer(
                    stakeholder_query, 
                    st.session_state.document_contents,
                    st.session_state.learning_materials if st.session_state.curriculum_ready else None
                )
        else:
            response_text = "Please upload and process documents first to enable question answering."
        
        # Store question and answer
        st.session_state.stakeholder_questions.append({
            "question": stakeholder_query,
            "answer": response_text,
            "answered": bool(response_text)
        })
        st.sidebar.success("Question submitted and answered!")
        st.rerun()

# Quiz response validation
def validate_quiz_response(quiz_item_id, user_response, correct_response):
    if user_response == correct_response:
        st.success("✅ Correct answer! Well done!")
        # Mark as completed
        st.session_state.answered_quiz_items.add(quiz_item_id)
        return True
    else:
        st.error(f"Incorrect. The correct answer is: {correct_response}")
        return False

# Curriculum generation function
def initiate_curriculum_generation():
    # Set flag to indicate generation in progress
    st.session_state.building_curriculum = True
    st.session_state.curriculum_ready = False
    st.rerun()  # Refresh UI to show loading state

# Function to execute curriculum generation
def execute_curriculum_generation():
    try:
        # Retrieve representative document segments
        search_results = st.session_state.vector_collection.query(
            query_texts=["comprehensive curriculum development learning materials overview"],
            n_results=20
        )
        
        # Compile document segments
        document_compilation = ""
        if search_results and 'documents' in search_results and search_results['documents']:
            for doc_segment, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
                source = metadata.get('source', 'Unknown source')
                document_compilation += f"\n--- SOURCE: {source} ---\n"
                document_compilation += doc_segment + "\n\n"
        
        user_context = f"Role: {professional_role}, Focus Areas: {', '.join(learning_areas)}"
        
        # Generate document summary first
        summary_query = "Create a comprehensive synthesis of these documents highlighting key concepts, methodologies, and practical applications."
        document_overview = retrieve_contextual_answer(summary_query)
        
        # Curriculum generation prompt
        curriculum_prompt = f"""
        Create a comprehensive professional learning curriculum based on the provided documents.
        User Profile: {user_context}
        Document Overview: {document_overview}
        
        Document Contents: {document_compilation[:5000]}
        
        Design an engaging curriculum by:
        1. Analyzing all provided documents to identify core themes, concepts, and practical applications
        2. Creating an inspiring curriculum title that reflects the integrated knowledge
        3. Writing a detailed curriculum description (300+ words) that explains the learning journey
        4. Developing 5-8 progressive modules that build knowledge systematically
        5. Creating 4-6 clear learning objectives for each module with practical applications
        6. Developing detailed content for each module (500+ words) including:
           - Real-world case studies and examples
           - Practical application guidelines
           - Visual learning concepts
           - Step-by-step instruction for complex topics
           - Multiple perspective analysis where appropriate
        7. Including a module assessment with 3-5 thought-provoking questions per module
        
        Return the response as a JSON object with this structure:
        {{
            "course_title": "Your Curriculum Title",
            "course_description": "Detailed curriculum description",
            "modules": [
                {{
                    "title": "Module Title",
                    "learning_objectives": ["Objective 1", "Objective 2", "Objective 3"],
                    "content": "Detailed module content with examples and applications",
                    "quiz": {{
                        "questions": [
                            {{
                                "question": "Assessment question?",
                                "options": ["Option A", "Option B", "Option C", "Option D"],
                                "correct_answer": "Option B"
                            }}
                        ]
                    }}
                }}
            ]
        }}
        
        Make the content highly practical, actionable, and tailored to the professional context.
        Include detailed explanations, examples, and applications in each module.
        Compare different approaches where documents present varying perspectives.
        """
        
        try:
            # Initialize OpenAI client
            ai_client = OpenAI(api_key=openai_key)
            response = ai_client.chat.completions.create(
                model=selected_ai_model,
                messages=[{"role": "user", "content": curriculum_prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            # Process response
            response_content = response.choices[0].message.content
            
            try:
                st.session_state.learning_materials = json.loads(response_content)
                st.session_state.curriculum_ready = True
                
                # Count assessment questions for progress tracking
                total_assessment_items = 0
                for module in st.session_state.learning_materials.get("modules", []):
                    quiz = module.get("quiz", {})
                    total_assessment_items += len(quiz.get("questions", []))
                st.session_state.quiz_items_count = total_assessment_items
                
            except json.JSONDecodeError as e:
                st.error(f"JSON parsing error: {e}")
                st.text(response_content)
        
        except Exception as e:
            st.error(f"OpenAI API Error: {e}")
            st.error("Please verify your API key and model selection.")
            
    except Exception as e:
        st.error(f"Error: {e}")
    
    # Reset generation flag
    st.session_state.building_curriculum = False

# Main interface with tabbed layout
tab1, tab2, tab3 = st.tabs(["📚 Learning Curriculum", "❓ Q&A Forum", "📑 Learning Resources"])

# Check if curriculum generation is in progress
if st.session_state.building_curriculum:
    with st.spinner("Creating your personalized learning curriculum..."):
        # Reset completed questions when generating new curriculum
        st.session_state.answered_quiz_items = set()
        execute_curriculum_generation()
        st.success("✅ Your Learning Curriculum is Ready!")
        st.rerun()

with tab1:
    # Learning Curriculum Display
    if st.session_state.curriculum_ready and st.session_state.learning_materials:
        curriculum = st.session_state.learning_materials
        
        # Curriculum Header
        st.title(f"✨ {curriculum.get('course_title', 'Professional Learning Curriculum')}")
        st.markdown(f"*Tailored for {professional_role}s focusing on {', '.join(learning_areas)}*")
        st.write(curriculum.get('course_description', 'A structured curriculum to develop your professional skills.'))
        
        # Progress Tracker
        completed_items = len(st.session_state.answered_quiz_items)
        total_items = st.session_state.quiz_items_count
        completion_percentage = (completed_items / total_items * 100) if total_items > 0 else 0
        
        st.progress(completion_percentage / 100)
        st.write(f"**Learning Progress:** {completed_items}/{total_items} assessments completed ({completion_percentage:.1f}%)")
        
        st.markdown("---")
        st.subheader("📋 Curriculum Structure")
        
        # Module Overview
        modules = curriculum.get("modules", [])
        if modules:
            module_titles = [module.get('title', f'Module {i+1}') for i, module in enumerate(modules)]
            for i, title in enumerate(module_titles, 1):
                st.write(f"**Module {i}:** {title}")
        else:
            st.warning("No modules found in curriculum content.")
        
        st.markdown("---")
        
        # Detailed Module Content
        for i, module in enumerate(modules, 1):
            module_title = module.get('title', f'Module {i}')
            with st.expander(f"📚 Module {i}: {module_title}"):
                # Learning Objectives
                st.markdown("### 🎯 Learning Objectives:")
                objectives = module.get('learning_objectives', [])
                if objectives:
                    for obj in objectives:
                        st.markdown(f"- {obj}")
                else:
                    st.write("No learning objectives defined.")
                
                # Module Content with improved formatting
                st.markdown("### 📖 Module Content:")
                content = module.get('content', 'No content available for this module.')
                
                # Format paragraphs for better readability
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if para.startswith('#'):
                        # Handle markdown headings
                        st.markdown(para)
                    elif para.startswith('*') and para.endswith('*'):
                        # Handle emphasized text
                        st.markdown(para)
                    elif para.startswith('1.') or para.startswith('- '):
                        # Handle list items
                        st.markdown(para)
                    else:
                        # Regular paragraphs
                        st.write(para)
                        st.write("")  # Add spacing
                
                # Key Insights Section
                st.markdown("### 💡 Key Insights:")
                st.info("The concepts in this module provide practical skills for immediate application in your professional context.")
                
                # Module Assessment
                st.markdown("### 📝 Knowledge Check:")
                quiz = module.get('quiz', {})
                questions = quiz.get('questions', [])
                
                if questions:
                    for q_idx, question in enumerate(questions, 1):
                        question_id = f"mod_{i}_q_{q_idx}"
                        question_text = question.get('question', f'Question {q_idx}')
                        
                        # Assessment container
                        assessment_container = st.container()
                        with assessment_container:
                            st.markdown(f"**Question {q_idx}:** {question_text}")
                            
                            options = question.get('options', [])
                            if options:
                                # Create unique radio button key
                                option_id = f"assessment_{i}_{q_idx}"
                                selected_answer = st.radio("Select your answer:", options, key=option_id)
                                
                                # Create unique submit button key
                                submit_id = f"verify_{i}_{q_idx}"
                                
                                # Show completion status
                                if question_id in st.session_state.answered_quiz_items:
                                    st.success("✓ Question completed")
                                else:
                                    if st.button(f"Verify Answer", key=submit_id):
                                        correct_answer = question.get('correct_answer', '')
                                        validate_quiz_response(question_id, selected_answer, correct_answer)
                            else:
                                st.write("No options available for this question.")
                        
                        st.markdown("---")
                else:
                    st.write("No assessment questions available for this module.")

    else:
        # Welcome Screen
        st.title("Welcome to the Learning Ecosystem Platform")
        st.markdown("""
        ## Transform your professional development with AI-powered learning system
        
        Upload multiple PDF documents and create a comprehensive, personalized learning curriculum!
        
        ### Getting Started:
        1. Enter your OpenAI API key in the sidebar
        2. Select your professional role and learning priorities
        3. Upload PDF documents containing relevant learning materials
        4. Generate your curriculum to begin your personalized learning journey
        
        Enhance your skills and accelerate your professional growth!
        """)
        
        # Generate Curriculum Button
        if st.session_state.document_contents and openai_key and not st.session_state.building_curriculum:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Generate My Curriculum", use_container_width=True):
                    initiate_curriculum_generation()
        elif st.session_state.building_curriculum:
            st.info("Creating your personalized curriculum... Please wait.")

with tab2:
    st.title("💬 Q&A Forum")
    st.markdown("""
    This section allows you to ask questions and receive AI-generated answers about the learning materials.
    Submit your questions in the sidebar, and our AI will provide answers based on the uploaded documents.
    """)
    
    if not st.session_state.stakeholder_questions:
        st.info("No questions have been submitted yet. Add a question in the sidebar to get started.")
    else:
        for i, query in enumerate(st.session_state.stakeholder_questions):
            with st.expander(f"Question {i+1}: {query['question'][:50]}..." if len(query['question']) > 50 else f"Question {i+1}: {query['question']}"):
                st.write(f"**Question:** {query['question']}")
                
                if query['answered']:
                    st.write(f"**Answer:** {query['answer']}")
                else:
                    st.info("Generating answer...")
                    # Generate answer on-demand
                    if st.session_state.vector_collection:
                        try:
                            answer = retrieve_contextual_answer(
                                query['question'], 
                                None,  # Using vector database directly
                                st.session_state.learning_materials if st.session_state.curriculum_ready else None
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
                        st.warning("No indexed documents available. Please upload documents to generate answers.")

with tab3:
    st.title("📑 Learning Resources")
    
    if not st.session_state.document_contents:
        st.info("No learning materials uploaded yet. Please upload PDF files in the sidebar to view their content here.")
    else:
        st.write(f"**{len(st.session_state.document_contents)} learning resources available:**")
        
        for i, doc in enumerate(st.session_state.document_contents):
            with st.expander(f"Resource {i+1}: {doc['filename']}"):
                # Display preview of document content
                preview = doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text']
                st.markdown("### Document Preview:")
                st.text_area("Content Sample:", value=preview, height=300, disabled=True)
                
                # AI-generated document summary
                if st.button(f"Generate Summary for {doc['filename']}", key=f"summary_{i}"):
                    with st.spinner("Creating document summary..."):
                        summary_query = f"Provide a comprehensive summary of {doc['filename']} highlighting key concepts, methodologies, and applications:"
                        summary = retrieve_contextual_answer(summary_query)
                        st.markdown("### AI-Generated Summary:")
                        st.write(summary)
