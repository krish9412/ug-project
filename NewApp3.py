import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
from openai import OpenAI

# Application configuration with custom theme
st.set_page_config(
    page_title="🧠 Knowledge Fusion Learning Hub", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom session state initialization with dictionary comprehension
default_states = {
    'learning_materials': None,
    'curriculum_created': False,
    'generation_in_progress': False,
    'answered_assessments': set(),
    'assessment_count': 0,
    'document_contents': [],
    'stakeholder_questions': [],
    'user_identifier': str(uuid.uuid4()),
    'document_library': [],
    'document_names': []
}

# Initialize session state variables
for key, default_value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Application sidebar configuration
with st.sidebar:
    st.title("🎓 Knowledge Fusion Hub")
    
    # Application reset functionality
    if st.button("🔄 New Session"):
        # Clear all session variables except user_identifier
        current_id = st.session_state.user_identifier
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Restore user identifier and reinitialize default states
        st.session_state.user_identifier = current_id
        for key, value in default_states.items():
            if key != 'user_identifier':
                st.session_state[key] = value
        st.rerun()

    # API key configuration
    api_key = st.text_input("🔑 OpenAI API Key", type="password")

    # Document uploader with improved UI
    st.subheader("📚 Learning Materials")
    uploaded_docs = st.file_uploader("Upload PDF Resources", type=['pdf'], accept_multiple_files=True)

# PDF processing function with improved error handling
def process_pdf_content(pdf_file):
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            full_content = ""
            for idx, page in enumerate(pdf.pages):
                page_content = page.extract_text() or ""
                full_content += f"[Page {idx+1}]\n{page_content}\n\n"
        return full_content
    except Exception as e:
        st.error(f"Unable to process PDF: {str(e)}")
        return ""

# Process uploaded documents with contextual feedback
if uploaded_docs and api_key:
    # Check if document list has changed
    current_names = [doc.name for doc in uploaded_docs]
    if current_names != st.session_state.document_names:
        # Reset document-related state
        st.session_state.document_contents = []
        st.session_state.document_library = []
        st.session_state.document_names = current_names
        
        # Process each document with detailed progress updates
        with st.sidebar:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, doc in enumerate(uploaded_docs):
                status_text.text(f"Processing: {doc.name}")
                extracted_content = process_pdf_content(doc)
                if extracted_content:
                    st.session_state.document_contents.append({
                        "filename": doc.name,
                        "text": extracted_content,
                        "word_count": len(extracted_content.split())
                    })
                    st.session_state.document_library.append(doc)
                
                # Update progress
                progress_value = (idx + 1) / len(uploaded_docs)
                progress_bar.progress(progress_value)
            
            if st.session_state.document_contents:
                status_text.success(f"✅ Processed {len(st.session_state.document_contents)} documents")
            else:
                status_text.error("No documents could be processed")
else:
    with st.sidebar:
        st.info("Please provide API key and upload materials to begin")

# Additional sidebar configuration options
with st.sidebar:
    # Model selection with recommendations
    model_choices = {
        "gpt-4o-mini": "Fast & Budget-friendly",
        "gpt-4o": "Balanced Performance", 
        "gpt-4": "Legacy Stability"
    }
    model_options = list(model_choices.keys())
    model_labels = [f"{model} ({desc})" for model, desc in model_choices.items()]
    selected_index = st.selectbox(
        "AI Engine", 
        range(len(model_options)),
        format_func=lambda i: model_labels[i]
    )
    selected_model = model_options[selected_index]

    # User role selection with organized categories
    role_categories = {
        "Leadership": ["Executive", "Manager", "Team Lead"],
        "Technical": ["Developer", "Designer", "Systems Analyst"],
        "Business": ["Marketer", "Sales Executive", "Business Analyst"],
        "Support": ["Human Resources", "Customer Service", "New Graduate"]
    }
    
    # Flatten categories for selection
    all_roles = []
    for category, roles in role_categories.items():
        all_roles.extend(roles)
    
    selected_role = st.selectbox("Professional Role", all_roles)

    # Learning priorities with multi-select
    learning_areas = ["Strategic Leadership", "Technical Proficiency", 
                     "Communication Skills", "Project Management", 
                     "Design Thinking", "Team Collaboration", 
                     "Financial Acumen"]
    learning_priorities = st.multiselect("Learning Priorities", learning_areas)

# Display documents in sidebar with metadata
if st.session_state.document_names:
    with st.sidebar:
        st.markdown("---")
        st.subheader("📑 Document Library")
        for i, filename in enumerate(st.session_state.document_names):
            if i < len(st.session_state.document_contents):
                word_count = st.session_state.document_contents[i].get("word_count", 0)
                st.text(f"{i+1}. {filename} ({word_count} words)")

# Enhanced knowledge retrieval function
def retrieve_contextual_answer(query, documents, curriculum=None):
    if not api_key:
        return "Please provide an API key to enable this feature."
    
    if not documents:
        return "No learning materials available. Please upload documents first."
        
    # Build context with document metadata and content samples
    context_builder = ""
    for i, doc in enumerate(documents[:3]):  # Limit context to prevent token overflow
        # Include document metadata and sample text with page markers
        sample_text = doc["text"][:3000]  # Sample first 3000 chars
        context_builder += f"\n---- Source {i+1}: {doc['filename']} ----\n{sample_text}\n"
    
    # Add curriculum context if available
    curriculum_context = ""
    if curriculum:
        curriculum_context = f"""
        Curriculum: {curriculum.get('curriculum_title', '')}
        Overview: {curriculum.get('curriculum_description', '')}
        
        Key Modules:
        """
        for i, module in enumerate(curriculum.get('modules', []), 1):
            curriculum_context += f"""
            Module {i}: {module.get('title', '')}
            Goals: {', '.join(module.get('learning_objectives', []))}
            Summary: {module.get('content', '')[:250]}...
            """
    
    # Construct comprehensive prompt
    knowledge_prompt = f"""
    As an educational AI assistant, provide a comprehensive answer to the question below using 
    information from the provided learning materials. Be precise, informative, and educational.
    
    Question: {query}
    
    Source Materials: {context_builder}
    
    Curriculum Context: {curriculum_context}
    
    Important instructions:
    1. Base your answer primarily on the source materials provided
    2. Synthesize information from multiple sources when available
    3. Indicate when information comes from specific sources
    4. If the information is not in the sources, clearly state this
    5. Format your response with clear sections and bullet points when appropriate
    """
    
    try:
        # Initialize OpenAI client and generate response
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": knowledge_prompt}],
            temperature=0.4,  # Lower temperature for more factual responses
            max_tokens=1000
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Error retrieving information: {str(e)}"

# Stakeholder inquiry section in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("💬 Stakeholder Inquiries")

    stakeholder_query = st.text_area("Ask a question:", height=100)
    if st.button("Submit Inquiry"):
        if stakeholder_query:
            # Generate answer from available documents
            answer = ""
            if st.session_state.document_contents:
                with st.spinner("Researching answer..."):
                    answer = retrieve_contextual_answer(
                        stakeholder_query, 
                        st.session_state.document_contents,
                        st.session_state.learning_materials if st.session_state.curriculum_created else None
                    )
            else:
                answer = "Please upload and process learning materials first."
            
            # Store question and answer
            st.session_state.stakeholder_questions.append({
                "question": stakeholder_query,
                "answer": answer,
                "status": "complete" if answer else "pending"
            })
            st.success("Question received and processed!")
            st.rerun()

# Assessment evaluation function
def evaluate_assessment(question_id, participant_answer, correct_response):
    if participant_answer == correct_response:
        st.success("✓ Correct! Well done on your understanding.")
        st.session_state.answered_assessments.add(question_id)
        return True
    else:
        st.error(f"Not quite right. The correct answer is: {correct_response}")
        return False

# Curriculum generation function
def create_curriculum():
    # Set generation flag
    st.session_state.generation_in_progress = True
    st.session_state.curriculum_created = False
    st.rerun()  # Trigger UI update

# Actual curriculum generation implementation
def execute_curriculum_generation():
    try:
        # Prepare document content for processing
        combined_sources = ""
        for i, doc in enumerate(st.session_state.document_contents):
            source_summary = f"\n--- SOURCE {i+1}: {doc['filename']} ---\n"
            source_summary += doc['text'][:4000]  # Sample content from each source
            combined_sources += source_summary + "\n\n"
        
        professional_context = f"Role: {selected_role}, Focus: {', '.join(learning_priorities)}"
        
        # Generate content overview first
        overview_query = "Create a comprehensive synthesis of these documents identifying key themes, concepts, and practical applications."
        content_overview = retrieve_contextual_answer(overview_query, st.session_state.document_contents)
        
        curriculum_prompt = f"""
        Design a comprehensive professional development curriculum synthesizing multiple source materials.
        Context: {professional_context}
        Content Overview: {content_overview}
        
        Source Materials: {combined_sources[:6000]}
        
        Create a structured learning experience by:
        1. Analyzing the collective insights from all provided sources
        2. Creating an engaging curriculum title reflecting integrated knowledge
        3. Writing a detailed curriculum description (300+ words) explaining the synthesis approach
        4. Developing 5-7 cohesive modules with logical progression
        5. Providing 4-5 specific learning objectives per module with practical applications
        6. Creating detailed content for each module (500+ words) including:
           - Real-world applications and case studies
           - Implementation strategies
           - Visual concepts where appropriate
           - Step-by-step methodologies
           - Comparative analysis of different perspectives
        7. Including a knowledge assessment with 3-4 thought-provoking questions per module
        
        Return the response in this JSON format:
        {{
            "curriculum_title": "Title",
            "curriculum_description": "Detailed description",
            "modules": [
                {{
                    "title": "Module Title",
                    "learning_objectives": ["Objective 1", "Objective 2", "Objective 3"],
                    "content": "Module content with detailed explanations and applications",
                    "assessment": {{
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
        
        Ensure content is practical, actionable, and tailored to the professional context.
        Include detailed explanations and applications in each module.
        Compare differing perspectives from source materials where relevant.
        """
        
        try:
            # Generate curriculum using OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": curriculum_prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            # Process response
            response_content = response.choices[0].message.content
            
            try:
                st.session_state.learning_materials = json.loads(response_content)
                st.session_state.curriculum_created = True
                
                # Count total assessment questions
                question_count = sum(
                    len(module.get("assessment", {}).get("questions", [])) 
                    for module in st.session_state.learning_materials.get("modules", [])
                )
                st.session_state.assessment_count = question_count
                
            except json.JSONDecodeError as e:
                st.error(f"Error parsing curriculum data: {e}")
                st.text(response_content)
        
        except Exception as e:
            st.error(f"API Error: {e}")
            st.error("Please verify your API key and model selection.")
            
    except Exception as e:
        st.error(f"Generation error: {e}")
    
    # Reset generation flag
    st.session_state.generation_in_progress = False

# Main content area with organized tabs
main_tab, inquiries_tab, sources_tab = st.tabs(["📚 Learning Curriculum", "❓ Stakeholder Inquiries", "📑 Source Materials"])

# Check for ongoing curriculum generation
if st.session_state.generation_in_progress:
    with st.spinner("Synthesizing your customized curriculum from multiple sources..."):
        # Reset assessment tracking
        st.session_state.answered_assessments = set()
        execute_curriculum_generation()
        st.success("✅ Your Learning Curriculum is Ready!")
        st.rerun()

# Curriculum tab content
with main_tab:
    if st.session_state.curriculum_created and st.session_state.learning_materials:
        curriculum = st.session_state.learning_materials
        
        # Header with personalization
        st.title(f"🌟 {curriculum.get('curriculum_title', 'Professional Development Program')}")
        st.markdown(f"*Customized for {selected_role}s focusing on {', '.join(learning_priorities)}*")
        st.write(curriculum.get('curriculum_description', 'A structured learning pathway to enhance your professional skills.'))
        
        # Learning progress tracking
        completed = len(st.session_state.answered_assessments)
        total = st.session_state.assessment_count
        progress_ratio = (completed / total * 100) if total > 0 else 0
        
        # Visual progress indicator
        st.progress(progress_ratio / 100)
        st.write(f"**Learning Progress:** {completed}/{total} assessments completed ({progress_ratio:.1f}%)")
        
        st.markdown("---")
        st.subheader("📋 Program Overview")
        
        # Module outline
        modules = curriculum.get("modules", [])
        if modules:
            module_titles = [module.get('title', f'Module {i+1}') for i, module in enumerate(modules)]
            for i, title in enumerate(module_titles, 1):
                st.write(f"**Module {i}:** {title}")
        else:
            st.warning("No modules found in curriculum data.")
        
        st.markdown("---")
        
        # Detailed module content with improved layout
        for i, module in enumerate(modules, 1):
            module_title = module.get('title', f'Module {i}')
            with st.expander(f"📚 Module {i}: {module_title}"):
                # Learning objectives
                st.markdown("### 🎯 Learning Objectives:")
                objectives = module.get('learning_objectives', [])
                if objectives:
                    for obj in objectives:
                        st.markdown(f"- {obj}")
                else:
                    st.write("No learning objectives specified.")
                
                # Module content with enhanced formatting
                st.markdown("### 📖 Module Content:")
                module_content = module.get('content', 'No content available for this module.')
                
                # Format paragraphs with proper spacing and styling
                paragraphs = module_content.split('\n\n')
                for para in paragraphs:
                    cleaned_para = para.strip()
                    if cleaned_para.startswith('#'):
                        # Format headings
                        st.markdown(cleaned_para)
                    elif cleaned_para.startswith('*') and cleaned_para.endswith('*'):
                        # Format emphasized text
                        st.markdown(cleaned_para)
                    elif cleaned_para.startswith('1.') or cleaned_para.startswith('- '):
                        # Format lists
                        st.markdown(cleaned_para)
                    else:
                        # Standard paragraph with spacing
                        st.write(cleaned_para)
                        st.write("")  # Add paragraph spacing
                
                # Implementation guidance
                st.markdown("### 💡 Application Strategies:")
                st.info("Apply these concepts in your daily work through practical implementation strategies.")
                
                # Knowledge assessment with interactive elements
                st.markdown("### 📝 Knowledge Check:")
                assessment = module.get('assessment', {})
                questions = assessment.get('questions', [])
                
                if questions:
                    for q_idx, q in enumerate(questions, 1):
                        question_id = f"mod_{i}_q_{q_idx}"
                        question_text = q.get('question', f'Question {q_idx}')
                        
                        # Question container with styling
                        with st.container():
                            st.markdown(f"**Question {q_idx}:** {question_text}")
                            
                            options = q.get('options', [])
                            if options:
                                # Create unique key for each question
                                response_key = f"assess_{i}_{q_idx}"
                                participant_answer = st.radio("Select answer:", options, key=response_key)
                                
                                # Submit button with unique key
                                submit_key = f"check_{i}_{q_idx}"
                                
                                # Show completion status
                                if question_id in st.session_state.answered_assessments:
                                    st.success("✓ Assessment completed")
                                else:
                                    if st.button(f"Submit Answer", key=submit_key):
                                        correct_response = q.get('correct_answer', '')
                                        evaluate_assessment(question_id, participant_answer, correct_response)
                            else:
                                st.write("No answer options available.")
                        
                        st.markdown("---")
                else:
                    st.write("No assessment questions available for this module.")

    else:
        # Welcome screen with value proposition
        st.title("Knowledge Fusion Learning Hub")
        st.markdown("""
        ## Transform Multiple Resources into Cohesive Learning Experiences
        
        Upload your professional development materials and create a synthesized learning curriculum!
        
        ### Platform Benefits:
        1. **Knowledge Integration** - Combine insights from multiple documents
        2. **Personalized Learning** - Tailored to your role and priorities
        3. **Interactive Assessment** - Track understanding with knowledge checks
        4. **Stakeholder Engagement** - Answer questions using comprehensive knowledge base
        
        Start your integrated learning journey today!
        """)
        
        # Generate button - only when documents are ready
        if st.session_state.document_contents and api_key and not st.session_state.generation_in_progress:
            cols = st.columns([1, 2, 1])
            with cols[1]:
                if st.button("🚀 Create My Learning Curriculum", use_container_width=True):
                    create_curriculum()
        elif st.session_state.generation_in_progress:
            st.info("Creating your personalized curriculum... Please wait.")
        elif not api_key:
            st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        elif not st.session_state.document_contents:
            st.warning("Please upload PDF documents in the sidebar to continue.")

# Stakeholder inquiries tab
with inquiries_tab:
    st.title("💬 Stakeholder Inquiries")
    st.markdown("""
    This section allows stakeholders to ask questions about learning materials, curriculum content, or implementation strategies.
    Submit your inquiries in the sidebar, and our AI will research answers based on uploaded materials.
    """)
    
    if not st.session_state.stakeholder_questions:
        st.info("No inquiries have been submitted. Add a question in the sidebar to begin.")
    else:
        for i, inquiry in enumerate(st.session_state.stakeholder_questions):
            # Format question preview
            question_preview = inquiry['question']
            if len(question_preview) > 50:
                question_preview = f"{question_preview[:50]}..."
                
            with st.expander(f"Inquiry {i+1}: {question_preview}"):
                st.write(f"**Question:** {inquiry['question']}")
                
                if inquiry['status'] == "complete":
                    st.write(f"**Response:** {inquiry['answer']}")
                else:
                    st.info("Researching answer...")
                    # Generate answer if needed
                    if st.session_state.document_contents:
                        try:
                            answer = retrieve_contextual_answer(
                                inquiry['question'], 
                                st.session_state.document_contents,
                                st.session_state.learning_materials if st.session_state.curriculum_created else None
                            )
                            st.session_state.stakeholder_questions[i]['answer'] = answer
                            st.session_state.stakeholder_questions[i]['status'] = "complete"
                            st.rerun()
                        except Exception as e:
                            error_message = f"Error generating answer: {str(e)}. Please try refreshing."
                            st.error(error_message)
                            st.session_state.stakeholder_questions[i]['answer'] = error_message
                            st.session_state.stakeholder_questions[i]['status'] = "complete"
                    else:
                        st.warning("No learning materials available. Please upload documents.")

# Source materials tab
with sources_tab:
    st.title("📑 Source Materials")
    
    if not st.session_state.document_contents:
        st.info("No materials uploaded. Please add PDF files in the sidebar.")
    else:
        st.write(f"**{len(st.session_state.document_contents)} learning resources available:**")
        
        for i, doc in enumerate(st.session_state.document_contents):
            with st.expander(f"Resource {i+1}: {doc['filename']}"):
                # Show document preview
                preview_text = doc['text'][:1500] + "..." if len(doc['text']) > 1500 else doc['text']
                st.markdown("### Content Preview:")
                st.text_area("First 1500 characters:", value=preview_text, height=250, disabled=True)
                
                # Generate document summary
                if st.button(f"Analyze {doc['filename']}", key=f"analyze_{i}"):
                    with st.spinner("Analyzing document content..."):
                        summary_query = f"Create a comprehensive analysis of this document including key themes, concepts, and practical applications:"
                        summary = retrieve_contextual_answer(summary_query, [doc])
                        st.markdown("### Document Analysis:")
                        st.write(summary)
