"""
Autism Assessment Application

This Streamlit application uses a trained machine learning model to assess autism risk
based on responses to domain-specific questions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import sys
from reportlab.lib import colors
import plotly.graph_objects as go
from reportlab.lib.colors import blue, whitesmoke, beige, gray, black
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import base64


# Add the current directory to the path so we can import the model module
sys.path.append(os.path.dirname(__file__))

# Import the ML model module if it exists, otherwise show a message to train the model first
try:
    from model import predict_autism_risk
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Advanced Autism Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Custom CSS that works with both light and dark mode
st.markdown("""
    <style>
    /* Base colors that adapt to theme */
    :root {
        --text-color: rgb(49, 51, 63);
        --background-color: #f8f9fa;
        --card-background: #ffffff;
        --highlight-background: #fffeeb;
        --question-background: #f1f8ff;
        --button-color: #5b88a5;
        --border-color: #5b88a5;
        --heading-color: #2b6777;
    }

    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: rgb(250, 250, 250);
            --background-color: #0e1117;
            --card-background: #262730;
            --highlight-background: #3b3a25;
            --question-background: #1e2a3a;
            --heading-color: #70ccc5;
        }
    }

    /* Streamlit dark mode overrides */
    .dark {
        --text-color: rgb(250, 250, 250);
        --background-color: #0e1117;
        --card-background: #262730;
        --highlight-background: #3b3a25;
        --question-background: #1e2a3a;
        --heading-color: #70ccc5;
    }

    /* Global text color */
    .main * {
        color: var(--text-color);
    }

    /* Button styling */
    .stButton button {
        background-color: var(--button-color);
        color: white !important;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }

    /* Card styling */
    .result-card {
        background-color: var(--card-background);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Headings */
    h1, h2, h3 {
        color: var(--heading-color) !important;
    }

    /* Question boxes */
    .question {
        background-color: var(--question-background);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid var(--border-color);
    }

    /* Highlight boxes */
    .highlight {
        background-color: var(--highlight-background);
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #ffc107;
    }

    /* Force specific text elements to use theme colors */
    p, li, em, strong, span, div, label {
        color: var(--text-color) !important;
    }

    /* Make expander text visible */
    .streamlit-expanderHeader {
        color: var(--text-color) !important;
    }
    
    .streamlit-expanderContent {
        color: var(--text-color) !important;
    }
    
    /* Make slider labels and values visible */
    .stSlider label, .stSlider p {
        color: var(--text-color) !important;
    }
    
    /* Make radio button and checkbox labels visible */
    .stRadio label, .stCheckbox label {
        color: var(--text-color) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Check if models are trained and available
def check_model_availability():
    """Check if the ML models and necessary files exist."""
    required_files = [
        'models/autism_domain_model.pkl',
        'models/domain_scaler.pkl',
        'models/domain_weights.pkl',
        'models/domain_averages.pkl',
        'models/domain_mapping.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.error(f"Missing required model files: {', '.join(missing_files)}")
        st.info("Please run the autism_ml_model.py script first to train the models.")
        return False
    
    return True

# Initialize session state for multi-page app
if 'page' not in st.session_state:
    st.session_state.page = 'intro'
    
if 'responses' not in st.session_state:
    st.session_state.responses = {
        'pattern_recognition': {},
        'sensory_processing': {},
        'social_communication': {},
        'repetitive_behaviors': {},
        'open_ended': {}
    }
    
if 'results' not in st.session_state:
    st.session_state.results = {
        'pattern_recognition': 0,
        'sensory_processing': 0,
        'social_communication': 0,
        'repetitive_behaviors': 0,
        'overall_score': 0,
        'probability': 0
    }

if 'child_info' not in st.session_state:
    st.session_state.child_info = {
        'age': "",
        'gender': ""
    }

# Functions for navigation and response management
def go_to_page(page_name):
    """Navigate to a specific page."""
    st.session_state.page = page_name
    
def save_responses(section, responses):
    """Save responses for a specific section."""
    st.session_state.responses[section] = responses

def save_child_info(info):
    """Save child information."""
    st.session_state.child_info = info
    
def calculate_section_score(section_responses):
    """Calculate score for a section based on responses."""
    values = list(section_responses.values())
    if not values:
        return 0
    
    # Convert to numeric if needed
    numeric_values = []
    for v in values:
        try:
            # Handle slider values (0-10)
            numeric_values.append(float(v))
        except:
            # Skip non-numeric values
            continue
    
    if not numeric_values:
        return 0
        
    return sum(numeric_values) / len(numeric_values)

# Domain-specific insight text
def get_domain_insight(domain, level):
    """Return insight text for a specific domain and level."""
    insights = {
        "Pattern Recognition & Detailed Perception": {
            "high": "Your responses indicate your child shows strong characteristics of detailed perception and pattern recognition often associated with autism. They likely notice small details others miss, may excel at pattern-based activities, and might focus intensely on specific details rather than seeing the 'big picture.' These traits can be strengths in many contexts, such as mathematics, science, quality control, or detail-oriented work.",
            "moderate": "Your responses suggest your child has some characteristics of detailed perception and pattern recognition that may align with autism spectrum patterns of thinking. They might sometimes focus on details more than others, show skill with patterns, or prefer organized information.",
            "low": "Your responses indicate your child's pattern recognition and detail perception align more closely with typical development patterns. They likely balance detail focus with big-picture thinking effectively."
        },
        "Sensory Processing": {
            "high": "Your responses indicate your child has significant sensory processing differences that align with autism. They may be highly sensitive to certain sensory inputs (sounds, textures, lights), seek specific sensory experiences, or be under-responsive to other sensations. Understanding and accommodating these sensory needs can significantly improve their comfort and functioning.",
            "moderate": "Your responses suggest your child has some sensory processing differences that may align with autism spectrum patterns. They might have specific sensory sensitivities or preferences, though these may not significantly impact daily functioning.",
            "low": "Your responses indicate your child's sensory processing aligns more closely with typical development patterns. They likely process and respond to sensory information in expected ways."
        },
        "Social Communication": {
            "high": "Your responses indicate your child has social communication characteristics that strongly align with autism. They may process social information in a more analytical way, find aspects of social communication challenging, or have a more literal understanding of language. These different approaches to social understanding are central aspects of autism.",
            "moderate": "Your responses suggest your child has some social communication characteristics that may align with autism spectrum patterns. They might find certain social situations challenging or approach social understanding somewhat differently than peers.",
            "low": "Your responses indicate your child's social communication aligns more closely with typical development patterns. They likely navigate social situations and understand social communication in expected ways."
        },
        "Repetitive Behaviors & Focused Interests": {
            "high": "Your responses indicate your child shows patterns of intense interests and/or repetitive behaviors that strongly align with autism. They may have deep knowledge in specific areas of interest, prefer consistent routines, or engage in repetitive behaviors. These focused interests can be significant strengths and sources of expertise and joy.",
            "moderate": "Your responses suggest your child has some patterns of interests and behaviors that may align with autism spectrum patterns. They might have particular areas of strong interest or prefer routine and predictability more than some peers.",
            "low": "Your responses indicate your child's interests and behavioral patterns align more closely with typical development. They likely have a flexible approach to activities and routines."
        }
    }
    
    return insights[domain][level]

# Introduction Page
def intro_page():
    """Display the introduction page."""
    st.title("Advanced Autism Assessment Tool")
    
    st.markdown("""
    <div class="highlight">
    <p>Welcome to this advanced autism assessment tool designed for parents and guardians. 
    This assessment goes beyond simple screening questions to capture the complex patterns 
    of thinking, behavior, and processing that characterize autism spectrum conditions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Understanding this Assessment
    
    This tool explores four key domains that are characteristic of autism spectrum conditions:
    
    1. **Pattern Recognition & Detailed Perception**: How your child processes visual and auditory information, notices details, and recognizes patterns.
    
    2. **Sensory Processing**: How your child responds to sensory information like sounds, textures, lights, and tastes.
    
    3. **Social Communication**: How your child understands and participates in social interactions and communications.
    
    4. **Repetitive Behaviors & Focused Interests**: Your child's interests, routines, and repetitive behaviors.
    
    ### Important Notes:
    
    - This assessment takes approximately 15-20 minutes to complete
    - Answer as honestly as possible based on your observations of your child
    - There are no "right" or "wrong" answers
    - This tool is not intended to replace professional diagnosis
    - The assessment uses a machine learning model trained on autism diagnostic data
    """)
    
    st.markdown("### Child's Information")
    
    # Create a form for child information
    with st.form(key="child_info_form"):
        col1, col2 = st.columns(2)
        with col1:
            child_age = st.text_input("Child's Age (years)", value=st.session_state.child_info.get('age', ""))
        with col2:
            child_gender = st.selectbox(
                "Child's Gender", 
                options=["", "Male", "Female", "Non-binary", "Prefer not to say"],
                index=0 if not st.session_state.child_info.get('gender') else 
                     ["", "Male", "Female", "Non-binary", "Prefer not to say"].index(st.session_state.child_info.get('gender'))
            )
        
        submit_button = st.form_submit_button(label="Begin Assessment")
        
        if submit_button:
            if not child_age or not child_gender:
                st.error("Please provide both age and gender information.")
            else:
                try:
                    float(child_age)  # Check if age is a number
                    save_child_info({'age': child_age, 'gender': child_gender})
                    go_to_page('pattern_recognition')
                except ValueError:
                    st.error("Please enter a valid number for age.")
    
    st.markdown("""
    <div class="highlight">
    <p><strong>Note:</strong> All data entered in this assessment is processed locally on your device and is not stored or transmitted externally.</p>
    </div>
    """, unsafe_allow_html=True)

# Pattern Recognition Page
def pattern_recognition_page():
    """Display the pattern recognition assessment page."""
    st.title("Pattern Recognition & Detailed Perception")
    
    st.markdown("""
    <div class="highlight">
    <p>This section explores how your child perceives details, patterns, and information organization. 
    Autistic individuals often excel at noticing details and patterns that others might miss, 
    and may have a preference for systematic organization of information.</p>
    </div>
    """, unsafe_allow_html=True)
    
    responses = st.session_state.responses['pattern_recognition'].copy()
    
    # Question 1
    st.markdown("""
    <div class="question">
    <p><strong>1. How often does your child notice small details or changes in their environment that others typically miss?</strong></p>
    <p><em>Examples: A small object moved from its usual place, subtle changes in someone's appearance, distant sounds others don't notice</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['detail_noticing'] = st.slider(
        "Rate from 0 (rarely notices) to 10 (notices exceptionally often)",
        0, 10, responses.get('detail_noticing', 5), key="q1_pattern"
    )
    
    # Question 2
    st.markdown("""
    <div class="question">
    <p><strong>2. How intensely does your child organize, categorize, or collect items?</strong></p>
    <p><em>Examples: Organizing toys by color/size/type, creating elaborate classifications, arranging objects in precise patterns</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['organization'] = st.slider(
        "Rate from 0 (minimal organizing) to 10 (intense categorization)",
        0, 10, responses.get('organization', 5), key="q2_pattern"
    )
    
    # Question 3
    st.markdown("""
    <div class="question">
    <p><strong>3. When engaged with visual information, does your child tend to focus on specific details rather than the overall context or big picture?</strong></p>
    <p><em>Examples: Focusing on a small part of an illustration rather than the whole scene, noticing specific features of objects rather than their function</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['detail_focus'] = st.slider(
        "Rate from 0 (mostly sees the big picture) to 10 (intensely detail-focused)",
        0, 10, responses.get('detail_focus', 5), key="q3_pattern"
    )
    
    # Question 4
    st.markdown("""
    <div class="question">
    <p><strong>4. How skilled is your child at recognizing patterns in information, numbers, or visual elements?</strong></p>
    <p><em>Examples: Quickly solving pattern-based puzzles, noticing numerical or letter patterns, recognizing visual sequences</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['pattern_recognition'] = st.slider(
        "Rate from 0 (average pattern recognition) to 10 (exceptional pattern recognition)",
        0, 10, responses.get('pattern_recognition', 5), key="q4_pattern"
    )
    
    # Question 5
    st.markdown("""
    <div class="question">
    <p><strong>5. How precisely does your child prefer things to be arranged or organized?</strong></p>
    <p><em>Examples: Objects must be perfectly aligned or symmetrical, items need to be in exact positions, strong preference for orderliness</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['precision'] = st.slider(
        "Rate from 0 (flexible about arrangement) to 10 (requires precise arrangement)",
        0, 10, responses.get('precision', 5), key="q5_pattern"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous", key="prev_pattern"):
            save_responses('pattern_recognition', responses)
            go_to_page('intro')
    with col2:
        if st.button("Next", key="next_pattern"):
            save_responses('pattern_recognition', responses)
            go_to_page('sensory_processing')

# Sensory Processing Page
def sensory_processing_page():
    """Display the sensory processing assessment page."""
    st.title("Sensory Processing")
    
    st.markdown("""
    <div class="highlight">
    <p>This section examines how your child processes and responds to sensory information.
    Many autistic individuals experience sensory information differently, sometimes more intensely 
    or less intensely than neurotypical individuals. These differences can significantly impact daily life.</p>
    </div>
    """, unsafe_allow_html=True)
    
    responses = st.session_state.responses['sensory_processing'].copy()
    
    # Question 1
    st.markdown("""
    <div class="question">
    <p><strong>1. How does your child respond to unexpected or loud sounds?</strong></p>
    <p><em>Examples: Covering ears with hands, becoming distressed in noisy environments, being disturbed by sounds others don't notice</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['sound_sensitivity'] = st.slider(
        "Rate from 0 (tolerates sounds well) to 10 (extremely sensitive to sounds)",
        0, 10, responses.get('sound_sensitivity', 5), key="q1_sensory"
    )
    
    # Question 2
    st.markdown("""
    <div class="question">
    <p><strong>2. How particular is your child about food textures, tastes, or presentation?</strong></p>
    <p><em>Examples: Strong aversions to certain textures, very limited food preferences, needs foods not to touch on plate</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['food_sensitivity'] = st.slider(
        "Rate from 0 (flexible with food) to 10 (extremely particular about food)",
        0, 10, responses.get('food_sensitivity', 5), key="q2_sensory"
    )
    
    # Question 3
    st.markdown("""
    <div class="question">
    <p><strong>3. How does your child respond to being touched or to certain clothing textures?</strong></p>
    <p><em>Examples: Disliking light touch but seeking deep pressure, removing clothing tags, only wearing certain fabrics, avoiding hugs</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['touch_sensitivity'] = st.slider(
        "Rate from 0 (comfortable with touch/textures) to 10 (highly sensitive to touch/textures)",
        0, 10, responses.get('touch_sensitivity', 5), key="q3_sensory"
    )
    
    # Question 4
    st.markdown("""
    <div class="question">
    <p><strong>4. How sensitive is your child to visual stimuli such as bright lights, certain colors, or visual patterns?</strong></p>
    <p><em>Examples: Squinting or covering eyes in bright environments, discomfort with fluorescent lighting, fascination with certain visual patterns</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['visual_sensitivity'] = st.slider(
        "Rate from 0 (minimal visual sensitivity) to 10 (extreme visual sensitivity)",
        0, 10, responses.get('visual_sensitivity', 5), key="q4_sensory"
    )
    
    # Question 5
    st.markdown("""
    <div class="question">
    <p><strong>5. Does your child seek out specific sensory experiences repeatedly?</strong></p>
    <p><em>Examples: Spinning, rocking, hand-flapping, seeking deep pressure, watching spinning objects, specific repetitive sounds</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['sensory_seeking'] = st.slider(
        "Rate from 0 (rarely seeks sensory input) to 10 (frequently seeks specific sensations)",
        0, 10, responses.get('sensory_seeking', 5), key="q5_sensory"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous", key="prev_sensory"):
            save_responses('sensory_processing', responses)
            go_to_page('pattern_recognition')
    with col2:
        if st.button("Next", key="next_sensory"):
            save_responses('sensory_processing', responses)
            go_to_page('social_communication')

# Social Communication Page
def social_communication_page():
    """Display the social communication assessment page."""
    st.title("Social Communication")
    
    st.markdown("""
    <div class="highlight">
    <p>This section explores how your child understands and engages in social interactions.
    Autistic individuals often process social information differently, which can affect how they
    interpret social cues, understand non-literal language, and engage in social interactions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    responses = st.session_state.responses['social_communication'].copy()
    
    # Question 1
    st.markdown("""
    <div class="question">
    <p><strong>1. How would you describe your child's eye contact during conversations?</strong></p>
    <p><em>Note: Some autistic individuals find eye contact overwhelming or distracting, and may listen better when not making eye contact</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['eye_contact'] = st.slider(
        "Rate from 0 (consistent eye contact) to 10 (consistently avoids eye contact)",
        0, 10, responses.get('eye_contact', 5), key="q1_social"
    )
    
    # Question 2
    st.markdown("""
    <div class="question">
    <p><strong>2. How does your child understand and respond to non-literal language such as metaphors, idioms, or sarcasm?</strong></p>
    <p><em>Examples: Taking idioms literally (like "it's raining cats and dogs"), misunderstanding sarcasm, needing explicit rather than implied instructions</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['literal_language'] = st.slider(
        "Rate from 0 (easily understands non-literal language) to 10 (consistently interprets language literally)",
        0, 10, responses.get('literal_language', 5), key="q2_social"
    )
    
    # Question 3
    st.markdown("""
    <div class="question">
    <p><strong>3. How would you rate your child's ability to understand unspoken social rules and expectations?</strong></p>
    <p><em>Examples: Understanding appropriate personal space, taking turns in conversation, adjusting behavior to different social contexts</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['social_rules'] = st.slider(
        "Rate from 0 (intuitively understands social rules) to 10 (significant difficulty with unspoken social rules)",
        0, 10, responses.get('social_rules', 5), key="q3_social"
    )
    
    # Question 4
    st.markdown("""
    <div class="question">
    <p><strong>4. How deeply does your child analyze or overthink social interactions or others' intentions?</strong></p>
    <p><em>Examples: Analyzing past conversations in great detail, seeking explicit rules for social situations, needing extensive preparation for social events</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['social_analysis'] = st.slider(
        "Rate from 0 (approaches social situations intuitively) to 10 (extensively analyzes social situations)",
        0, 10, responses.get('social_analysis', 5), key="q4_social"
    )
    
    # Question 5
    st.markdown("""
    <div class="question">
    <p><strong>5. How does your child respond when others express strong emotions?</strong></p>
    <p><em>Examples: Becoming overwhelmed by others' emotions, difficulty determining appropriate emotional responses, logical rather than emotional responses</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['emotional_processing'] = st.slider(
        "Rate from 0 (responds to emotions intuitively) to 10 (finds others' emotions difficult to process)",
        0, 10, responses.get('emotional_processing', 5), key="q5_social"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous", key="prev_social"):
            save_responses('social_communication', responses)
            go_to_page('sensory_processing')
    with col2:
        if st.button("Next", key="next_social"):
            save_responses('social_communication', responses)
            go_to_page('repetitive_behaviors')

# Repetitive Behaviors Page
def repetitive_behaviors_page():
    """Display the repetitive behaviors assessment page."""
    st.title("Repetitive Behaviors & Focused Interests")
    
    st.markdown("""
    <div class="highlight">
    <p>This section explores repetitive behaviors, routines, and focused interests.
    Autistic individuals often engage in repetitive behaviors and may develop deep interests
    in specific topics, learning exhaustively about them and finding them deeply fulfilling.</p>
    </div>
    """, unsafe_allow_html=True)
    
    responses = st.session_state.responses['repetitive_behaviors'].copy()
    
    # Question 1
    st.markdown("""
    <div class="question">
    <p><strong>1. How intensely does your child focus on particular topics or interests?</strong></p>
    <p><em>Examples: Developing expert knowledge on specific subjects, talking at length about special interests, collecting related items</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['special_interests'] = st.slider(
        "Rate from 0 (typical level of interest) to 10 (extremely intense focus on specific interests)",
        0, 10, responses.get('special_interests', 5), key="q1_repetitive"
    )
    
    # Question 2
    st.markdown("""
    <div class="question">
    <p><strong>2. How important are routines and predictability to your child?</strong></p>
    <p><em>Examples: Distress when routines change unexpectedly, need for extensive preparation before transitions, preference for consistent schedules</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['routine_changes'] = st.slider(
        "Rate from 0 (adapts easily to changes) to 10 (extreme distress with changes to routine)",
        0, 10, responses.get('routine_changes', 5), key="q2_repetitive"
    )
    
    # Question 3
    st.markdown("""
    <div class="question">
    <p><strong>3. How frequently does your child engage in repetitive physical movements?</strong></p>
    <p><em>Examples: Hand flapping, rocking, spinning, pacing, repetitive finger movements, tapping</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['repetitive_movements'] = st.slider(
        "Rate from 0 (rarely shows repetitive movements) to 10 (frequent repetitive movements)",
        0, 10, responses.get('repetitive_movements', 5), key="q3_repetitive"
    )
    
    # Question 4
    st.markdown("""
    <div class="question">
    <p><strong>4. How systematically does your child approach learning new information or skills?</strong></p>
    <p><em>Examples: Creating detailed systems for organizing information, learning every aspect of a topic in depth, following precise steps</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['systematic_learning'] = st.slider(
        "Rate from 0 (flexible learning approach) to 10 (highly systematic learning approach)",
        0, 10, responses.get('systematic_learning', 5), key="q4_repetitive"
    )
    
    # Question 5
    st.markdown("""
    <div class="question">
    <p><strong>5. How precisely must certain activities or routines be performed?</strong></p>
    <p><em>Examples: Needs exact phrasing in stories/songs, specific sequence of actions before bed, things must be "just right"</em></p>
    </div>
    """, unsafe_allow_html=True)
    responses['precision_needs'] = st.slider(
        "Rate from 0 (flexible about how things are done) to 10 (requires precise execution)",
        0, 10, responses.get('precision_needs', 5), key="q5_repetitive"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous", key="prev_repetitive"):
            save_responses('repetitive_behaviors', responses)
            go_to_page('social_communication')
    with col2:
        if st.button("Next", key="next_repetitive"):
            save_responses('repetitive_behaviors', responses)
            go_to_page('open_ended')

# Open-ended Questions Page
def open_ended_page():
    """Display the open-ended questions page."""
    st.title("Qualitative Assessment")
    
    st.markdown("""
    <div class="highlight">
    <p>Please answer the following open-ended questions about your child's behaviors, thinking patterns, and experiences.
    These responses provide valuable qualitative information that captures the unique aspects of your child that
    standardized questions might miss.</p>
    </div>
    """, unsafe_allow_html=True)
    
    responses = st.session_state.responses['open_ended'].copy() if 'open_ended' in st.session_state.responses else {}
    
    # Question 1
    st.markdown("""
    <div class="question">
    <p><strong>1. Please describe how your child responds to changes in routine or unexpected transitions. What strategies help them navigate these changes?</strong></p>
    </div>
    """, unsafe_allow_html=True)
    responses['open_ended_1'] = st.text_area(
        "Your response",
        value=responses.get('open_ended_1', ''),
        height=150,
        key="q1_open"
    )
    
    # Question 2
    st.markdown("""
    <div class="question">
    <p><strong>2. Describe any specific areas of intense interest your child has. How do they engage with these interests, and what depth of knowledge have they developed?</strong></p>
    </div>
    """, unsafe_allow_html=True)
    responses['open_ended_2'] = st.text_area(
        "Your response",
        value=responses.get('open_ended_2', ''),
        height=150,
        key="q2_open"
    )
    
    # Question 3
    st.markdown("""
    <div class="question">
    <p><strong>3. How does your child approach social situations? Are there particular environments or interactions they find especially challenging or comfortable?</strong></p>
    </div>
    """, unsafe_allow_html=True)
    responses['open_ended_3'] = st.text_area(
        "Your response",
        value=responses.get('open_ended_3', ''),
        height=150,
        key="q3_open"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous", key="prev_open"):
            save_responses('open_ended', responses)
            go_to_page('repetitive_behaviors')
    with col2:
        if st.button("Calculate Results", key="calc_results"):
            save_responses('open_ended', responses)
            process_assessment()
            go_to_page('results')

# Process assessment results
# Process assessment results
def process_assessment():
    """Process assessment responses and calculate domain scores and overall risk."""
    
    # Calculate scores for each domain
    pattern_recognition_responses = st.session_state.responses['pattern_recognition']
    sensory_processing_responses = st.session_state.responses['sensory_processing']
    social_communication_responses = st.session_state.responses['social_communication']
    repetitive_behaviors_responses = st.session_state.responses['repetitive_behaviors']
    open_ended_responses = st.session_state.responses.get('open_ended', {})
    
    # Calculate average scores for each domain
    pattern_score = calculate_section_score(pattern_recognition_responses)
    sensory_score = calculate_section_score(sensory_processing_responses)
    social_score = calculate_section_score(social_communication_responses)
    repetitive_score = calculate_section_score(repetitive_behaviors_responses)
    
    # Store scores in session state
    st.session_state.results['pattern_recognition'] = pattern_score
    st.session_state.results['sensory_processing'] = sensory_score
    st.session_state.results['social_communication'] = social_score
    st.session_state.results['repetitive_behaviors'] = repetitive_score
    
    # Analyze open-ended responses
    text_score = analyze_text_responses(open_ended_responses)
    st.session_state.results['text_analysis'] = text_score
    
    # Prepare input for ML model (including text analysis)
# Pass child info (including age) to the prediction function
    responses = {
        'pattern_recognition': pattern_score,
        'sensory_processing': sensory_score,
        'social_communication': social_score,
        'repetitive_behaviors': repetitive_score,
        'age': st.session_state.child_info.get('age', '')
    }
    
    # Include text analysis in final probability calculation
    text_weight = 0.15  # Weight for text analysis (adjust as needed)
    
    # Check if the ML model is available
    if MODEL_AVAILABLE and check_model_availability():
        try:
            # Combine all text responses for ML analysis
            all_text = ' '.join([
                open_ended_responses.get('open_ended_1', ''),
                open_ended_responses.get('open_ended_2', ''),
                open_ended_responses.get('open_ended_3', '')
            ])
            
            # Use the ML model to predict autism risk with text analysis
            prediction = predict_autism_risk(responses, all_text)
            
            # Store prediction results
            st.session_state.results['probability'] = prediction['probability']
            st.session_state.results['overall_score'] = prediction['overall_score']
            st.session_state.results['classification'] = prediction['classification']
            st.session_state.results['domain_percentiles'] = prediction.get('domain_percentiles', {})
            st.session_state.results['text_insights'] = prediction.get('text_insights', [])
        except Exception as e:
            st.error(f"Error using ML model: {e}")
            # Fallback to simple averaging if model fails
            overall_score = (pattern_score + sensory_score + social_score + repetitive_score) / 4
            adjusted_score = (overall_score * (1 - text_weight)) + (text_score['overall_score'] * text_weight)
            st.session_state.results['overall_score'] = adjusted_score
            st.session_state.results['probability'] = adjusted_score / 10  # Simple scaling
            st.session_state.results['text_insights'] = text_score['insights']
    else:
        # Fallback to simple averaging if model not available
        overall_score = (pattern_score + sensory_score + social_score + repetitive_score) / 4
        adjusted_score = (overall_score * (1 - text_weight)) + (text_score['overall_score'] * text_weight)
        st.session_state.results['overall_score'] = adjusted_score
        st.session_state.results['probability'] = adjusted_score / 10  # Simple scaling
        st.session_state.results['text_insights'] = text_score['insights']
def analyze_text_responses(responses):
    """Analyze open-ended text responses for autism indicators."""
    
    # Define keywords associated with autism characteristics
    autism_keywords = {
        'pattern_recognition': ['detail', 'pattern', 'notice', 'specific', 'organize', 'order', 'arrange', 'categorize', 'line up', 'sort'],
        'sensory_processing': ['loud', 'noise', 'bright', 'light', 'texture', 'touch', 'smell', 'taste', 'sensitive', 'overwhelm'],
        'social_communication': ['eye contact', 'literal', 'understand', 'social', 'conversation', 'friend', 'interact', 'play', 'share', 'emotion'],
        'repetitive_behaviors': ['routine', 'change', 'upset', 'repeat', 'interest', 'spin', 'flap', 'rock', 'ritual', 'same']
    }
    
    # Initialize domain scores
    domain_scores = {
        'pattern_recognition': 0,
        'sensory_processing': 0,
        'social_communication': 0,
        'repetitive_behaviors': 0
    }
    
    # Combine all text responses
    all_text = ' '.join([
        responses.get('open_ended_1', ''),
        responses.get('open_ended_2', ''),
        responses.get('open_ended_3', '')
    ]).lower()
    
    # Count keywords in each domain
    for domain, keywords in autism_keywords.items():
        count = 0
        for keyword in keywords:
            if keyword in all_text:
                count += 1
        # Calculate score as percentage of keywords found
        domain_scores[domain] = min(10, count * 10 / len(keywords)) if keywords else 0
    
    # Generate insights based on text analysis
    insights = []
    
    # Check for routine resistance indicators
    if 'routine' in all_text and ('upset' in all_text or 'difficult' in all_text or 'distress' in all_text):
        insights.append("Your description suggests your child may find changes in routine challenging, which is common in autism.")
    
    # Check for intense interests
    if 'interest' in all_text and ('intense' in all_text or 'deep' in all_text or 'focus' in all_text):
        insights.append("You've described focused interests that appear to be particularly intense or deep, which is often seen in autism.")
    
    # Check for social challenges
    if ('social' in all_text or 'interact' in all_text) and ('challenge' in all_text or 'difficult' in all_text or 'avoid' in all_text):
        insights.append("Your description indicates some social interaction challenges that align with autism characteristics.")
    
    # Calculate overall text score
    overall_score = sum(domain_scores.values()) / len(domain_scores) if domain_scores else 0
    
    return {
        'domain_scores': domain_scores,
        'overall_score': overall_score,
        'insights': insights
    }

# Results Page
def results_page():
    """Display the assessment results."""
    st.title("Assessment Results")

    # Get results from session state
    results = st.session_state.results

    st.markdown(
        """
        <div class="highlight">
        <p>Below are the results of the assessment. Remember that this tool is not a diagnostic 
        instrument, but rather an educational resource to help understand patterns of thinking 
        and behavior that may align with autism spectrum conditions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Overall probability
    st.subheader("Overall Assessment")

    probability = results.get('probability', 0) * 100

    if probability >= 70:
        result_text = "High alignment with autism characteristics"
    elif probability >= 40:
        result_text = "Moderate alignment with autism characteristics"
    else:
        result_text = "Low alignment with autism characteristics"

    st.success(f"Result: {result_text}")
    st.write(f"Probability score: {probability:.1f}%")

    # Add a gauge chart for overall probability
    try:
        # Add a gauge chart for overall probability
        probability_percentage = results.get('probability', 0) * 100

        # Define the threshold values and colors
        threshold_values = [0, 40, 70, 100]
        threshold_colors = ['green', 'yellow', 'red']

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Autism Alignment Probability", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "blue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(44, 160, 44, 0.6)'},
                    {'range': [40, 70], 'color': 'rgba(255, 144, 14, 0.6)'},
                    {'range': [70, 100], 'color': 'rgba(214, 39, 40, 0.6)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': probability_percentage
                }
            },
            number={'suffix': '%'}
        ))

        gauge_fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
        )

        st.plotly_chart(gauge_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate gauge chart: {e}")

    # Domain Scores
    st.subheader("Domain Breakdown")

    domains = {
        "Pattern Recognition & Detailed Perception": results['pattern_recognition'],
        "Sensory Processing": results['sensory_processing'],
        "Social Communication": results['social_communication'],
        "Repetitive Behaviors & Focused Interests": results['repetitive_behaviors']
    }

    # Create enhanced radar chart with Plotly
    try:
        categories = list(domains.keys())
        values = list(domains.values())

        # Add reference values for comparison
        typical_values = [3.5, 3.2, 3.7, 3.4]  # Example values for neurotypical baseline
        asd_values = [7.8, 7.5, 7.2, 7.6]  # Example values for ASD baseline

        fig = go.Figure()

        # Add the user's score
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Child',
            line=dict(color='rgb(31, 119, 180)', width=3),
            fillcolor='rgba(31, 119, 180, 0.25)'
        ))

        # Add neurotypical reference
        fig.add_trace(go.Scatterpolar(
            r=typical_values,
            theta=categories,
            fill='toself',
            name='Typical Development Reference',
            line=dict(color='rgba(44, 160, 44, 0.7)', width=2, dash='dot'),
            fillcolor='rgba(44, 160, 44, 0.1)'
        ))

        # Add ASD reference
        fig.add_trace(go.Scatterpolar(
            r=asd_values,
            theta=categories,
            fill='toself',
            name='ASD Reference',
            line=dict(color='rgba(214, 39, 40, 0.7)', width=2, dash='dot'),
            fillcolor='rgba(214, 39, 40, 0.1)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickvals=[2, 4, 6, 8, 10],
                    ticktext=['2', '4', '6', '8', '10'],
                    tickfont=dict(size=10)
                )
            ),
            title="Domain Score Profile",
            showlegend=True,
            legend=dict(
                font=dict(size=10),
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=80, r=80, t=50, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate domain chart: {e}")

    # Add a bar chart showing domain scores
    try:
        domain_names = list(domains.keys())
        domain_values = list(domains.values())

        # Create a color scale based on scores
        colors = ['rgba(44, 160, 44, 0.7)' if v < 4 else 
                'rgba(255, 127, 14, 0.7)' if v < 7 else 
                'rgba(214, 39, 40, 0.7)' for v in domain_values]

        bar_fig = go.Figure(data=[
            go.Bar(
                x=domain_names,
                y=domain_values,
                marker_color=colors,
                text=[f"{v:.1f}/10" for v in domain_values],
                textposition='auto',
            )
        ])

        bar_fig.update_layout(
            title="Domain Scores Breakdown",
            yaxis=dict(
                title="Score (0-10)",
                range=[0, 10]
            ),
            xaxis=dict(
                title="Domains",
                tickfont=dict(size=10)
            ),
            margin=dict(l=50, r=50, t=50, b=100),
            height=400
        )

        st.plotly_chart(bar_fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate bar chart: {e}")

    # Create a scatter plot comparing domain percentiles
    if 'domain_percentiles' in results:
        try:
            domain_percentiles = results['domain_percentiles']
            
            domains_list = list(domain_percentiles.keys())
            percentiles = [domain_percentiles[d] * 100 for d in domains_list]
            
            percentile_fig = go.Figure()
            
            # Add percentile points
            percentile_fig.add_trace(go.Scatter(
                x=domains_list,
                y=percentiles,
                mode='markers+lines',
                marker=dict(
                    size=15,
                    color=percentiles,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Percentile",
                        ticksuffix="%"
                    )
                ),
                line=dict(
                    width=2
                ),
                name='Domain Percentiles'
            ))
            
            # Add reference lines
            percentile_fig.add_shape(
                type="line",
                x0=-0.5,
                y0=75,
                x1=len(domains_list)-0.5,
                y1=75,
                line=dict(
                    color="rgba(255, 0, 0, 0.3)",
                    width=2,
                    dash="dash",
                )
            )
            
            percentile_fig.add_annotation(
                x=domains_list[-1],
                y=75,
                text="High Alignment",
                showarrow=False,
                yshift=10,
                font=dict(color="rgba(255, 0, 0, 0.7)")
            )
            
            percentile_fig.update_layout(
                title="Domain Percentile Comparison",
                yaxis=dict(
                    title="Percentile",
                    range=[0, 100]
                ),
                xaxis=dict(
                    title="Domains",
                    tickfont=dict(size=10)
                ),
                margin=dict(l=50, r=50, t=50, b=100),
                height=400
            )
            
            st.plotly_chart(percentile_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate percentile chart: {e}")

    # Display domain scores with progress bars
    for domain, score in domains.items():
        st.markdown(
            f"""
            <div style="margin-bottom: 10px;">
                <strong>{domain}:</strong> {score:.1f}/10
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(score / 10)

    # Domain-specific insights
    st.subheader("Domain-Specific Insights")

    for domain, score in domains.items():
        if score >= 7:
            level = "high"
        elif score >= 4:
            level = "moderate"
        else:
            level = "low"

        insight_text = get_domain_insight(domain, level)
        
        with st.expander(f"{domain} ({level.capitalize()} alignment)"):
            st.markdown(insight_text)

    # Text Analysis Insights
    if 'text_insights' in results and results.get('text_insights'):
        st.subheader("Qualitative Assessment Insights")
        
        for insight in results['text_insights']:
            st.info(insight)
        
    elif 'open_ended' in st.session_state.responses and any(st.session_state.responses['open_ended'].values()):
        st.subheader("Qualitative Assessment Insights")
        
        # Generate simple insights based on text responses even if we don't have ML-generated insights
        responses = st.session_state.responses['open_ended']
        all_text = ' '.join([str(val) for val in responses.values()]).lower()
        
        if 'routine' in all_text or 'change' in all_text:
            st.info("Your written responses mention routines or changes, which are important areas to consider when understanding autism characteristics.")
        
        if 'interest' in all_text or 'focus' in all_text:
            st.info("You described specific interests or focus areas, which can be notable features in autism profiles.")
        
        if 'social' in all_text or 'interact' in all_text:
            st.info("Your descriptions of social interactions provide valuable context for understanding your child's communication style.")
        
        if 'sensory' in all_text or 'sensitive' in all_text:
            st.info("Your observations about sensory experiences are important indicators when assessing autism characteristics.")
        
        if not any(keyword in all_text for keyword in ['routine', 'change', 'interest', 'focus', 'social', 'interact', 'sensory', 'sensitive']):
            st.info("Your written responses have been analyzed and provide additional context to the quantitative scores.")

    # Explanation of results
    st.subheader("Understanding the Results")
    st.markdown("""
        This assessment evaluates four key domains associated with autism spectrum conditions:

        1. **Pattern Recognition & Detailed Perception** - This reflects how your child processes
           visual and auditory information, notices details, and recognizes patterns.

        2. **Sensory Processing** - This reflects how your child responds to sensory information
           like sounds, textures, lights, and tastes.

        3. **Social Communication** - This reflects how your child understands and participates
           in social interactions and communications.

        4. **Repetitive Behaviors & Focused Interests** - This reflects your child's interests,
           routines, and repetitive behaviors.

        Higher scores in these domains indicate greater alignment with characteristics often
        associated with autism. The overall probability score is based on a machine learning model
        trained on diagnostic data.
    """)

    # Age-specific recommendations
    if 'age_recommendations' in results and results.get('age_recommendations'):
        st.subheader("Age-Appropriate Recommendations")
        for recommendation in results['age_recommendations']:
            st.info(recommendation)

    # Recommendations
    st.subheader("Next Steps")
    
    if probability >= 70:
        st.info("""
            The assessment indicates a high alignment with autism characteristics. 
            Consider discussing these results with a healthcare professional who specializes 
            in autism diagnosis, such as a developmental pediatrician, child psychologist, 
            or child psychiatrist.
        """)
    elif probability >= 40:
        st.info("""
            The assessment indicates a moderate alignment with autism characteristics. 
            You may want to monitor your child's development and consider discussing 
            these results with your child's healthcare provider during their next visit.
        """)
    else:
        st.info("""
            The assessment indicates a low alignment with autism characteristics. 
            While this suggests your child's development patterns are more neurotypical, 
            if you have specific concerns about certain behaviors or developmental aspects, 
            you can always discuss them with your child's healthcare provider.
        """)

    st.markdown("""
        **General recommendations:**
        
        1. **Consult with healthcare professionals**: A pediatrician, child psychologist, or
           child psychiatrist can provide professional guidance.
        
        2. **Seek a comprehensive evaluation**: A formal diagnosis can only be made by qualified
           professionals through comprehensive assessment.
        
        3. **Access support resources**: Regardless of diagnosis, understanding your child's unique
           thinking and processing style can help you support their needs.
        
        4. **Remember the strengths**: Autism is associated with many strengths, including attention
           to detail, pattern recognition, and deep focus on areas of interest.
    """)

    st.subheader("Save Results")
    
    if st.button("Download Results as PDF", key="download_pdf"):
        # Create PDF in memory
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Add custom styles
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            textColor=blue
        ))
        
        styles.add(ParagraphStyle(
            name='RecommendationText',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceAfter=8
        ))
        
        elements = []
        
        # Title
        title_style = styles["Heading1"]
        title_style.alignment = 1  # Center alignment
        elements.append(Paragraph("Autism Assessment Results", title_style))
        elements.append(Spacer(1, 12))
        
        # Child information
        if st.session_state.child_info.get('age') or st.session_state.child_info.get('gender'):
            info_text = "Child Information: "
            if st.session_state.child_info.get('age'):
                info_text += f"Age: {st.session_state.child_info.get('age')} years"
            if st.session_state.child_info.get('gender'):
                info_text += f", Gender: {st.session_state.child_info.get('gender')}"
            elements.append(Paragraph(info_text, styles["Normal"]))
            elements.append(Spacer(1, 12))
        
        # Assessment date
        from datetime import datetime
        elements.append(Paragraph(f"Assessment Date: {datetime.now().strftime('%B %d, %Y')}", styles["Normal"]))
        elements.append(Spacer(1, 24))
        
        # Overall result
        elements.append(Paragraph("Summary of Results", styles["SectionHeader"]))
        elements.append(Paragraph(f"Overall Assessment: {result_text}", styles["Normal"]))
        elements.append(Paragraph(f"Probability score: {probability:.1f}%", styles["Normal"]))
        elements.append(Spacer(1, 20))
        
        # Domain scores
        elements.append(Paragraph("Domain Scores", styles["SectionHeader"]))
        elements.append(Spacer(1, 6))
        
        # Create domain score table
        data = [["Domain", "Score (out of 10)", "Level"]]
        for domain, score in domains.items():
            level = "High" if score >= 7 else "Moderate" if score >= 4 else "Low"
            data.append([domain, f"{score:.1f}", level])
        
        domain_table = Table(data, colWidths=[250, 100, 100])
        domain_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), beige),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        elements.append(domain_table)
        elements.append(Spacer(1, 24))
        
        # Domain insights
        elements.append(Paragraph("Domain-Specific Insights", styles["SectionHeader"]))
        
        for domain, score in domains.items():
            if score >= 7:
                level = "high"
            elif score >= 4:
                level = "moderate"
            else:
                level = "low"
            
            insight_text = get_domain_insight(domain, level)
            elements.append(Paragraph(f"{domain} ({level.capitalize()} alignment)", styles["Heading3"]))
            elements.append(Paragraph(insight_text, styles["Normal"]))
            elements.append(Spacer(1, 12))
        
        # Text insights if available
        if 'text_insights' in results and results.get('text_insights'):
            elements.append(Paragraph("Qualitative Assessment", styles["SectionHeader"]))
            
            for insight in results['text_insights']:
                elements.append(Paragraph("• " + insight, styles["RecommendationText"]))
            
            elements.append(Spacer(1, 12))
        
        # Age-specific recommendations
        if 'age_recommendations' in results and results.get('age_recommendations'):
            elements.append(Paragraph("Age-Appropriate Recommendations", styles["SectionHeader"]))
            
            for recommendation in results['age_recommendations']:
                elements.append(Paragraph("• " + recommendation, styles["RecommendationText"]))
            
            elements.append(Spacer(1, 12))
        
        # Next steps
        elements.append(Paragraph("Next Steps", styles["SectionHeader"]))
        
        if probability >= 70:
            elements.append(Paragraph("The assessment indicates a high alignment with autism characteristics. Consider discussing these results with a healthcare professional who specializes in autism diagnosis.", styles["Normal"]))
        elif probability >= 40:
            elements.append(Paragraph("The assessment indicates a moderate alignment with autism characteristics. You may want to monitor your child's development and consider discussing these results with your child's healthcare provider during their next visit.", styles["Normal"]))
        else:
            elements.append(Paragraph("The assessment indicates a low alignment with autism characteristics. While this suggests your child's development patterns are more neurotypical, if you have specific concerns about certain behaviors or developmental aspects, you can always discuss them with your child's healthcare provider.", styles["Normal"]))
        
        elements.append(Spacer(1, 12))
        
        # Resources
        elements.append(Paragraph("Helpful Resources", styles["SectionHeader"]))
        elements.append(Paragraph("• Autism Society of America: www.autism-society.org", styles["RecommendationText"]))
        elements.append(Paragraph("• Autism Speaks: www.autismspeaks.org", styles["RecommendationText"]))
        elements.append(Paragraph("• Child Mind Institute: www.childmind.org", styles["RecommendationText"]))
        elements.append(Paragraph("• Autism Science Foundation: www.autismsciencefoundation.org", styles["RecommendationText"]))
        
        elements.append(Spacer(1, 24))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle(
            "Disclaimer", 
            parent=styles["Normal"],
            fontSize=8,
            textColor=gray
        )
        elements.append(Paragraph("DISCLAIMER: This assessment is for educational purposes and is not a substitute for professional diagnosis. Always consult with healthcare professionals for proper evaluation and diagnosis of autism spectrum conditions.", disclaimer_style))
        
        # Build PDF
        doc.build(elements)
        
        # Create download link
        pdf_data = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
        pdf_filename = "autism_assessment_results.pdf"
        href = f'<a href="data:application/pdf;base64,{pdf_data}" download="{pdf_filename}">Click here to download your PDF</a>'
        
        st.markdown(href, unsafe_allow_html=True)
        st.success("PDF generated successfully! Click the link above to download.")
    
    # Option to restart
    if st.button("Start New Assessment", key="restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.page = 'intro'
        st.rerun()

# Main application logic
def main():
    """Main application entry point."""
    
    # Check if the ML model is available
    if not MODEL_AVAILABLE:
        st.sidebar.warning("Machine learning model not available. Using simplified scoring.")
    
    # Sidebar information
    st.sidebar.title("About This Tool")
    st.sidebar.info(
        """
        This autism assessment tool uses machine learning to evaluate responses
        across four key domains characteristic of autism spectrum conditions.
        
        The model was trained on diagnostic data to identify patterns that align
        with autism characteristics.
        
        **Note:** This tool is for educational purposes and is not a substitute
        for professional diagnosis.
        """
    )
    
    # Navigation in sidebar (for debugging)
    st.sidebar.subheader("Navigation")
    pages = {
        'Introduction': 'intro',
        'Pattern Recognition': 'pattern_recognition',
        'Sensory Processing': 'sensory_processing',
        'Social Communication': 'social_communication',
        'Repetitive Behaviors': 'repetitive_behaviors',
        'Open-ended Questions': 'open_ended',
        'Results': 'results'
    }
    
    # Only show debug navigation in development
    debug_mode = False
    if debug_mode:
        selected_page = st.sidebar.radio("Go to page:", list(pages.keys()))
        if st.sidebar.button("Navigate"):
            st.session_state.page = pages[selected_page]
            st.experimental_rerun()
    
    # Show progress in sidebar
    current_page = st.session_state.page
    progress_value = {
        'intro': 0,
        'pattern_recognition': 0.2,
        'sensory_processing': 0.4,
        'social_communication': 0.6,
        'repetitive_behaviors': 0.8,
        'open_ended': 0.9,
        'results': 1.0
    }.get(current_page, 0)
    
    st.sidebar.subheader("Assessment Progress")
    st.sidebar.progress(progress_value)
    
    # Run the appropriate page based on session state
    if st.session_state.page == 'intro':
        intro_page()
    elif st.session_state.page == 'pattern_recognition':
        pattern_recognition_page()
    elif st.session_state.page == 'sensory_processing':
        sensory_processing_page()
    elif st.session_state.page == 'social_communication':
        social_communication_page()
    elif st.session_state.page == 'repetitive_behaviors':
        repetitive_behaviors_page()
    elif st.session_state.page == 'open_ended':
        open_ended_page()
    elif st.session_state.page == 'results':
        results_page()

if __name__ == "__main__":
    main()