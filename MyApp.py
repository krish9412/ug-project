import streamlit as st
import os
import tempfile
import json
import io
import markdown_it
import requests
from openai import OpenAI
import PyPDF2  # Add this library for PDF processing

# Page Configuration
st.set_page_config(page_title="📚 Professional Learning Platform", layout="wide")

# Sidebar
st.sidebar.title("🎓 Professional Learning Setup")

# OpenAI API Key
openai_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

# Model selection - only GPT-4o-mini
selected_model = "gpt-4o-mini"

# File Uploader - now including PDF files
uploaded_file = st.sidebar.file_uploader("📝 Upload Training Document", type=['txt', 'md', 'pdf'])

# Initialize session state
if 'course_content' not in st.session_state:
    st.session_state.course_content = None
if 'course_generated' not in st.session_state:
    st.session_state.course_generated = False
if 'completed_questions' not in st.session_state:
    st.session_state.completed_questions = set()
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'employer_queries' not in st.session_state:
    st.session_state.employer_queries = []

# Extract Text Function - Modified to handle PDFs
def extract_text(file):
    try:
        # Reset file pointer to beginning
        file.seek(0)
        
        # Get file name and extension
        file_name = file.name.lower() if hasattr(file, 'name') else ''
        
        # Handle different file types
        if file_name.endswith('.pdf'):
            # Process PDF file
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            return text
        else:
            # Process text or markdown file
            text = file.read().decode('utf-8')
            return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

# RAG Function to generate answer from document content
def generate_rag_answer(question, document_text, course_content=None):
    try:
        if not openai_api_key:
            return "API key is required to generate answers."
        
        context = document_text[:4000]  # Limit context to avoid token issues
        
        # Include course content in context if available
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
        
        Document Content: {context}
        
        Course Information: {course_context}
        
        Provide a comprehensive answer using information from the document and course content.
        If the question cannot be answered based on the provided information, say so politely.
        """
        
        # Create OpenAI client
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        # Return generated answer
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Employer Queries Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Questions")

new_query = st.sidebar.text_area("Add a new question:", height=100)
if st.sidebar.button("Submit Question"):
    if new_query:
        # Generate answer automatically if document is available
        answer = ""
        if st.session_state.extracted_text:
            with st.spinner("Generating answer..."):
                answer = generate_rag_answer(
                    new_query, 
                    st.session_state.extracted_text,
                    st.session_state.course_content if st.session_state.course_generated else None
                )
        
        st.session_state.employer_queries.append({
            "question": new_query,
            "answer": answer,
            "answered": bool(answer)
        })
        st.sidebar.success("Question submitted and answered!")
        st.rerun()

# Generate Course Button
if uploaded_file and openai_api_key:
    if st.sidebar.button("🚀 Generate Course"):
        st.session_state.course_generated = False
        # Reset completed questions when generating a new course
        st.session_state.completed_questions = set()
        
        try:
            # Show a spinner while processing
            with st.spinner("Generating your personalized professional course..."):
                # Extract text from uploaded file and store it using the updated function
                extracted_text = extract_text(uploaded_file)
                st.session_state.extracted_text = extracted_text
                
                prompt = f"""
                Design a professional learning course based on the given text.
                Document: {extracted_text[:3000]}
                
                Create an engaging and comprehensive course with:
                1. An inspiring course title
                2. Detailed course description (at least 200 words)
                3. 3-6 modules that build upon each other
                4. Clear learning objectives for each module (3-6 objectives per module)
                5. Detailed and informative content for each module (at least 300 words per module with examples, case studies, and practical applications)
                6. A quiz with 2-4 thought-provoking questions per module
                
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
                
                Make the content practical, actionable, and tailored to professional development.
                Provide detailed explanations, real-world examples, and practical applications in each module content.
                """
                
                try:
                    # New OpenAI API format
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model=selected_model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.7
                    )
                    
                    # Access the response content
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
                        
                        st.success("✅ Professional Course Generated Successfully!")
                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing JSON response: {e}")
                        st.text(response_content)
                
                except Exception as e:
                    st.error(f"OpenAI API Error: {e}")
                    st.error("Please check your API key.")
                    
        except Exception as e:
            st.error(f"Error: {e}")

# The rest of your code remains the same
# Function to check answer and update progress
def check_answer(question_id, user_answer, correct_answer):
    if user_answer == correct_answer:
        st.success("🎉 Correct! Well done!")
        # Add to completed questions set if not already there
        st.session_state.completed_questions.add(question_id)
        return True
    else:
        st.error(f"Not quite. The correct answer is: {correct_answer}")
        return False

# Main content area with tabs
tab1, tab2 = st.tabs(["📚 Course Content", "❓ Questions"])

with tab1:
    # Display Course Content
    if st.session_state.course_generated and st.session_state.course_content:
        course = st.session_state.course_content
        
        # Course Header
        st.title(f"🌟 {course.get('course_title', 'Professional Learning Course')}")
        st.write(course.get('course_description', 'A structured course to enhance your professional skills.'))
        
        # Progress tracker
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
        
        # Detailed Module Content
        for i, module in enumerate(modules, 1):
            module_title = module.get('title', f'Module {i}')
            with st.expander(f"📚 Module {i}: {module_title}"):
                st.markdown("### Learning Objectives:")
                objectives = module.get('learning_objectives', [])
                if objectives:
                    for obj in objectives:
                        st.markdown(f"- {obj}")
                else:
                    st.write("No learning objectives specified.")
                
                st.markdown("### Module Content:")
                module_content = module.get('content', 'No content available for this module.')
                
                # Split content into paragraphs and add proper formatting
                paragraphs = module_content.split('\n\n')
                for para in paragraphs:
                    st.markdown(para)
                    st.write("")  # Add spacing between paragraphs
                
                st.markdown("### Key Takeaways:")
                st.info("The content in this module will help you develop practical skills that you can apply immediately in your professional context.")
                
                # Safely access quiz questions
                st.markdown("### Module Quiz:")
                quiz = module.get('quiz', {})
                questions = quiz.get('questions', [])
                
                if questions:
                    for q_idx, q in enumerate(questions, 1):
                        question_id = f"module_{i}_question_{q_idx}"
                        question_text = q.get('question', f'Question {q_idx}')
                        st.markdown(f"**Question {q_idx}:** {question_text}")
                        
                        options = q.get('options', [])
                        if options:
                            # Create a unique key for each radio button
                            option_key = f"quiz_{i}_{q_idx}"
                            user_answer = st.radio("Select your answer:", options, key=option_key)
                            
                            # Create a unique key for each submit button
                            submit_key = f"submit_{i}_{q_idx}"
                            if st.button(f"Check Answer", key=submit_key):
                                correct_answer = q.get('correct_answer', '')
                                check_answer(question_id, user_answer, correct_answer)
                        else:
                            st.write("No options available for this question.")
                        
                        # Show completion status for this question
                        if question_id in st.session_state.completed_questions:
                            st.success("✓ Question completed")
                        
                        st.markdown("---")
                else:
                    st.write("No quiz questions available for this module.")

    else:
        # Welcome screen when no course is generated yet
        st.title("Welcome to Professional Learning Platform")
        st.markdown("""
        ## Transform your professional development with AI-powered learning
        
        Upload a Markdown, text, or PDF document, and we'll create a personalized professional learning course just for you!
        
        ### How it works:
        1. Enter your OpenAI API key in the sidebar
        2. Upload a Markdown (.md), text (.txt), or PDF (.pdf) document related to your area of interest
        3. Click "Generate Course" to create your personalized professional learning journey
        
        Get ready to enhance your skills and accelerate your professional growth!
        """)

with tab2:
    st.title("💬 Questions")
    st.markdown("""
    This section allows you to ask questions and get AI-generated answers about the course content or related topics.
    Submit your questions in the sidebar, and our AI will automatically generate answers based on the uploaded document.
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
                    if st.session_state.extracted_text:
                        answer = generate_rag_answer(
                            query['question'], 
                            st.session_state.extracted_text,
                            st.session_state.course_content if st.session_state.course_generated else None
                        )
                        st.session_state.employer_queries[i]['answer'] = answer
                        st.session_state.employer_queries[i]['answered'] = True
                        st.rerun()
                    else:
                        st.warning("No document uploaded yet. Please upload a document to generate answers.")
