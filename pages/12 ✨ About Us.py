import streamlit as st
from PIL import Image
import base64

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown('⛔ [ATTENTION] Please login through the main app to access this page.')
else:
    st.title("✨ About Us")
    # st.markdown('HELLO WORLD ^_^')

    # Custom CSS for styling
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #2E86AB;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 2rem;
            padding: 1rem 0;
        }
        
        .team-member-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        
        .member-name {
            color: #2E86AB;
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .member-role {
            color: #A23B72;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            font-style: italic;
        }
        
        .member-bio {
            color: #333;
            font-size: 1.1rem;
            line-height: 1.6;
            text-align: justify;
        }
        
        .skills-tag {
            background-color: #2E86AB;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
            margin: 0.2rem;
            display: inline-block;
        }
        
        .interests-section {
            background-color: #f8f9fa;
            border-left: 4px solid #A23B72;
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 5px;
        }
        
        .project-contribution {
            background-color: #e8f4f8;
            border: 1px solid #2E86AB;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }
        
        .contribution-title {
            color: #2E86AB;
            font-weight: bold;
            font-size: 1.1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<h1 class="main-header">Meet Our Team</h1>', unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem; max-width: 800px; margin-left: auto; margin-right: auto;">
    We're a dynamic duo combining systems expertise with data science innovation. 
    Our diverse backgrounds in IT operations and quality engineering, enhanced by advanced analytics training, 
    drive us to create meaningful solutions from complex data challenges.
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for team members
    col1, col2 = st.columns(2)

    # Janice's profile
    with col1:
        st.markdown("""
        <div class="team-member-card">
            <div class="member-name">Janice</div>
            <div class="member-role">Systems Operations & Data Imputation Specialist</div>
            
            <div class="member-bio">
                With a solid background in systems and network operations, Janice knows how to keep complex systems 
                running smoothly. While diving into postgraduate studies in Data Analytics, she picked up new skills 
                in data cleaning, scientific imputation, and dashboard development — blending real-world IT experience 
                with data-driven thinking.
            </div>
            
            <div class="project-contribution">
                <div class="contribution-title">🎯 Project Contribution</div>
                Led the design of the imputation logic that filled in the gaps without losing the story hidden in the data.
            </div>
            
            <div class="interests-section">
                <strong>Beyond the Code:</strong><br>
                Enjoys writing, long drives, and unraveling layered mysteries in shows like <em>CSI</em>, <em>X-Files</em>, 
                and <em>Alias</em> — always drawn to the kind of puzzles where every detail matters.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Skills tags for Janice
        st.markdown("**Core Expertise:**")
        skills_janice = ["Systems Operations", "Network Management", "Data Cleaning", "Scientific Imputation", "Dashboard Development"]
        for skill in skills_janice:
            st.markdown(f'<span class="skills-tag">{skill}</span>', unsafe_allow_html=True)

    # Faye's profile
    with col2:
        st.markdown("""
        <div class="team-member-card">
            <div class="member-name">Faye</div>
            <div class="member-role">Quality Engineering Lead & Data Strategist</div>
            
            <div class="member-bio">
                By day, Faye leads quality engineering for complex systems at one of New Zealand's top banks. 
                By passion, she's a data strategist who blends logic, creativity, and precision to make sense 
                of real-world patterns. Her postgraduate studies in Data Analytics deepened her interest in 
                forecasting, feature engineering, and applied machine learning.
            </div>
            
            <div class="project-contribution">
                <div class="contribution-title">🎯 Project Contribution</div>
                Led the development of the PM10 forecasting system — transforming raw environmental signals into actionable insight.
            </div>
            
            <div class="interests-section">
                <strong>Beyond the Code:</strong><br>
                When not wrangling code or catching bugs, you'll likely find her swimming in the ocean or 
                chasing new horizons through travel — always curious, always learning.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Skills tags for Faye
        st.markdown("**Core Expertise:**")
        skills_faye = ["Quality Engineering", "Data Strategy", "Forecasting", "Feature Engineering", "Machine Learning"]
        for skill in skills_faye:
            st.markdown(f'<span class="skills-tag">{skill}</span>', unsafe_allow_html=True)

    # Team collaboration section
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0;">
        <h2 style="color: #2E86AB; margin-bottom: 1rem;">Our Collaborative Approach</h2>
        <div style="font-size: 1.1rem; color: #666; max-width: 900px; margin: 0 auto; line-height: 1.6;">
            Together, we bring a unique combination of operational reliability and analytical innovation. 
            Our complementary skills in systems management, quality assurance, and data science enable us 
            to tackle complex challenges from multiple angles — ensuring both technical soundness and 
            meaningful insights in every project we undertake.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Contact or connect section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 2rem; border-radius: 15px; margin: 2rem 0;">
            <h3 style="margin-bottom: 1rem;">Ready to Collaborate?</h3>
            <p style="font-size: 1.1rem; margin-bottom: 0;">
                We're always excited to tackle new data challenges and build innovative solutions.
            </p>
        </div>
        """, unsafe_allow_html=True)