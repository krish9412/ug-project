import streamlit as st
import os
import tempfile
import openai
import json
import io
import PyPDF2

# Set page configuration
st.set_page_config(
    page_title="Professional Learning Platform",
    page_icon="💼",
    layout="wide"
)

# Initialize session state
if 'course_generated' not in st.session_state:
    st.session_state.course_generated = False
if 'course_content' not in st.session_state:
    st.session_state.course_content = None
if 'current_module' not in st.session_state:
    st.session_state.current_module = 0
if 'quiz_results' not in st.session_state:
    st.session_state.quiz_results = {}
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

def extract_pdf_text(pdf_path):
    """
    Extract text from PDF document using PyPDF2
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
    return text

def generate_professional_course(document_text, professional_context):
    """
    Generate a professional development course
    """
    # Truncate text to prevent extremely long inputs
    truncated_text = document_text[:3000]
    
    prompt = f"""
    Create a professional development course based on the following document:

    Professional Context:
    {professional_context}

    Document Content:
    {truncated_text}

    Course Requirements:
    1. Design a course with 3-4 professional modules
    2. Focus on practical workplace applications
    3. Include learning objectives for each module
    4. Create scenario-based learning content
    5. Develop multiple-choice quizzes to assess understanding

    Output Format (Strict JSON):
    {{
        "course_title": "Professional Development Course",
        "course_description": "Practical skills for professional growth",
        "modules": [
            {{
                "module_number": 1,
                "title": "Module Title",
                "learning_objectives": ["Objective 1", "Objective 2"],
                "content": "Detailed module content with professional insights",
                "workplace_scenarios": ["Scenario 1", "Scenario 2"],
                "quiz": {{
                    "questions": [
                        {{
                            "question": "Quiz question",
                            "options": ["Option A", "Option B", "Option C", "Option D"],
                            "correct_answer": "Correct Option"
                        }}
                    ]
                }}
            }}
        ]
    }}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert instructional designer creating professional development courses."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.6
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Course generation error: {e}")
        return None

def generate_course_query_response(course_content, query):
    """
    Generate a response to user's query based on course content
    """
    # Prepare course context
    course_context = json.dumps(course_content, indent=2)
    
    prompt = f"""
    You are an expert course instructor. 
    Course Content: {course_context}

    User Query: {query}

    Provide a comprehensive and professional response to the query based on the course content. 
    If the query cannot be directly answered from the course content, provide a helpful, 
    professionally-worded explanation.

    Response Guidelines:
    - Be precise and informative
    - Use professional language
    - If the answer is not in the course content, explain why
    - Offer additional context or guidance if possible
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert course instructor providing detailed, professional answers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {e}"

def display_course_module(module, module_index):
    """
    Display a single course module with interactive elements
    """
    st.subheader(f"Module {module_index + 1}: {module['title']}")
    
    # Learning Objectives
    st.markdown("**Learning Objectives:**")
    for obj in module['learning_objectives']:
        st.markdown(f"- {obj}")
    
    # Module Content
    st.markdown("**Module Content:**")
    st.write(module['content'])
    
    # Workplace Scenarios
    st.markdown("**Workplace Scenarios:**")
    for scenario in module['workplace_scenarios']:
        st.markdown(f"🏢 {scenario}")
    
    # Quiz Section
    st.markdown("**Module Quiz:**")
    with st.form(key=f"quiz_form_{module_index}"):
        quiz_results = {}
        for q_idx, question in enumerate(module['quiz']['questions']):
            quiz_results[q_idx] = st.radio(
                question['question'], 
                question['options'],
                key=f"quiz_{module_index}_{q_idx}"
            )
        
        submit_quiz = st.form_submit_button("Submit Quiz")
        
        if submit_quiz:
            # Evaluate Quiz
            score = 0
            total_questions = len(module['quiz']['questions'])
            
            for q_idx, question in enumerate(module['quiz']['questions']):
                if quiz_results.get(q_idx) == question['correct_answer']:
                    score += 1
            
            # Store and display results
            pass_percentage = (score / total_questions) * 100
            st.session_state.quiz_results[module_index] = {
                'score': score,
                'total': total_questions,
                'percentage': pass_percentage
            }
            
            if pass_percentage >= 70:
                st.success(f"Great job! You scored {score}/{total_questions} ({pass_percentage:.1f}%)")
            else:
                st.warning(f"You scored {score}/{total_questions} ({pass_percentage:.1f}%). Review the material and try again.")

def add_query_section(course_content):
    """
    Add an interactive query section to the Streamlit app
    """
    st.sidebar.header("Course Queries")
    
    # Query input
    user_query = st.sidebar.text_area("Ask a Question about the Course", height=100)
    
    if st.sidebar.button("Get Answer"):
        if user_query.strip():
            # Generate response
            query_response = generate_course_query_response(course_content, user_query)
            
            # Store query in history
            st.session_state.query_history.append({
                'query': user_query,
                'response': query_response
            })
    
    # Display query history
    if st.session_state.query_history:
        st.sidebar.header("Query History")
        for query_item in reversed(st.session_state.query_history):
            with st.sidebar.expander(f"Q: {query_item['query'][:50]}..."):
                st.markdown("**Question:**")
                st.write(query_item['query'])
                st.markdown("**Answer:**")
                st.write(query_item['response'])

def main():
    st.title("💼 Professional Learning Platform")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Course Setup")
        
        # OpenAI API Key
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        
        # Professional Context
        professional_role = st.selectbox(
            "Your Professional Role",
            [
                "Manager", 
                "Individual Contributor", 
                "Executive", 
                "Team Leader", 
                "Specialist"
            ]
        )
        
        learning_focus = st.multiselect(
            "Learning Focus Areas",
            [
                "Leadership Skills",
                "Communication",
                "Technical Skills",
                "Project Management",
                "Personal Development"
            ]
        )
        
        # PDF Upload
        uploaded_file = st.file_uploader("Upload Training PDF", type=['pdf'])
        
        # Generate Course Button
        if uploaded_file and openai_api_key:
            if st.button("Generate Professional Course"):
                # Set API Key
                openai.api_key = openai_api_key
                
                # Save temporary PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name
                
                try:
                    # Extract PDF text
                    pdf_text = extract_pdf_text(pdf_path)
                    
                    # Prepare professional context
                    professional_context = f"""
                    Professional Role: {professional_role}
                    Learning Focus: {', '.join(learning_focus)}
                    """
                    
                    # Generate Course
                    course_content = generate_professional_course(pdf_text, professional_context)
                    
                    if course_content:
                        st.session_state.course_content = course_content
                        st.session_state.course_generated = True
                        st.session_state.current_module = 0
                        st.session_state.query_history = []  # Reset query history
                        st.success("Professional Course Generated Successfully!")
                
                except Exception as e:
                    st.error(f"Course generation error: {e}")
                
                # Clean up temporary file
                os.unlink(pdf_path)
    
    # Course Display
    if st.session_state.course_generated:
        course = st.session_state.course_content
        
        st.header(course['course_title'])
        st.write(course['course_description'])
        
        # Add Query Section
        add_query_section(course)
        
        # Module Navigation
        module_tabs = st.tabs([f"Module {m+1}" for m in range(len(course['modules']))])
        
        for i, tab in enumerate(module_tabs):
            with tab:
                display_course_module(course['modules'][i], i)
        
        # Overall Course Progress
        st.subheader("Course Progress")
        progress_container = st.container()
        with progress_container:
            for module_idx, results in st.session_state.quiz_results.items():
                st.metric(
                    f"Module {module_idx + 1}", 
                    f"{results['score']}/{results['total']} ({results['percentage']:.1f}%)"
                )

if __name__ == "__main__":
    main()