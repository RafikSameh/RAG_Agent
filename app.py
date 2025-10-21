# app.py - The main entry point and navigation page

import streamlit as st
import base64
from pathlib import Path

# ==================================
# === INITIAL CONFIGURATION ===
# ==================================
st.set_page_config(
    page_title="AI Agent Platform Selector", 
    page_icon="🤖", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize page state (for generic tracking, though not strictly required for routing)
st.session_state.setdefault("page", "home")

# --- Load Image for Banner ---
IMAGE_FILE_NAME = "robot_image.jpeg"
b64 = ""
try:
    if Path(IMAGE_FILE_NAME).exists():
        with open(IMAGE_FILE_NAME, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
    else:
        # Placeholder tiny image
        b64 = "iVBORw0KGgoAAAANSUhEUAAAAEAAAABACAYAAACqMYADAAAACXBIWXMAAAsTAAALEwEAmpwYAAABeklEQVR42u3bQUsGQRQA4D8tO3b3FkLsw+QvB/XbYQe3D2N34gTuxl3CjrsS3FqRk2zYwJ2JcR4N4j0b1G+e2B04p+B3Q9nC3Uv+rTfgQkL4I+wN3X2Y7+k7R7/w3Y7b43c68+g+eM73523yG98/x/x3h3d+hT+G98/p/k7l3d/Vz/n753T/D3W8d3fEaH88xvf71/X8H79r9+e5+/d3eM/L1d+hZ3980v4vH9v/F3T893f4T5ff93Bf75X9w+Q/r4r/v9s4/v7NlF8/x/p+u7vjPL7/u4P//G9u35/7x3f5T/ff9/Bf2s/3f8b1vH9v/F3O9/B3S8/Hj9yW8L7Bw/c1vB+g/v1Bf/d+/F9Yw//X/fx2r/f7t+hL9j3h38/2uF9oP/dGfv3l/N2xP/j+8f/35nE/9jfh//9x/H/j3fxf/j/8f/33P8//v8Qv/D+R93X3T+P9vH+n1d/d3P/8c///+x/79z9w19f+X59b/V3/kL/13Xh/4a3T+z3Q/v5/s+Xg3QAAAABJRU5ErkJggg=="
except Exception:
    b64 = "iVBORw0KGgoAAAANSUhEUAAAAEAAAABACAYAAACqMYADAAAACXBIWXMAAAsTAAALEwEAmpwYAAABeklEQVR42u3bQUsGQRQA4D8tO3b3FkLsw+QvB/XbYQe3D2N34gTuxl3CjrsS3FqRk2zYwJ2JcR4N4j0b1G+e2B04p+B3Q9nC3Uv+rTfgQkL4I+wN3X2Y7+k7R7/w3Y7b43c68+g+eM73523yG98/x/x3h3d+hT+G98/p/k7l3d/Vz/n753T/D3W8d3fEaH88xvf71/X8H79r9+e5+/d3eM/L1d+hZ3980v4vH9v/F3T893f4T5ff93Bf75X9w+Q/r4r/v9s4/v7NlF8/x/p+u7vjPL7/u4P//G9u35/7x3f5T/ff9/Bf2s/3f8b1vH9v/F3O9/B3S8/Hj9yW8L7Bw/c1vB+g/v1Bf/d+/F9Yw//X/fx2r/f7t+hL9j3h38/2uF9oP/dGfv3l/N2xP/j+8f/35nE/9jfh//9x/H/j3fxf/j/8f/33P8//v8Qv/D+R93X3T+P9vH+n1d/d3P/8c///+x/79z9w19f+X59b/V3/kL/13Xh/4a3T+z3Q/v5/s+Xg3QAAAABJRU5ErkJggg=="


# --- CSS STYLING (Shared) ---
st.markdown(f"""
<style>
    .stApp > header {{ visibility: hidden; }}
    .stApp {{ margin-top: 0 !important; }} 
    
    @keyframes subtleFloat {{
        0% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-3px); }}
        100% {{ transform: translateY(0); }}
    }}

    .welcome-banner {{
        min-height: 450px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white; 
        padding: 80px 50px;
        background-color: #2c3e50; 
        position: relative;
        overflow: hidden; 
        border-radius: 0 0 15px 15px; 
        margin-bottom: 30px;
    }}
    
    .welcome-banner::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 1; 
        
        background-image: url('data:image/jpeg;base64,{b64}');
        background-size: cover; 
        background-repeat: no-repeat;
        background-position: center;

        mask-image: radial-gradient(
            circle at center, 
            white 40%, 
            transparent 95%
        );
        background-color: rgba(0, 0, 0, 0.5); 
        background-blend-mode: multiply; 
    }}

    .banner-content * {{
        z-index: 2; 
        position: relative;
    }}
    .banner-content h1 {{
        font-size: 3.5rem; 
        margin-bottom: 0.3em;
        font-weight: 800;
        color: #ecf0f1; 
        text-shadow: 0 2px 5px rgba(0,0,0,0.7); 
    }}
    .banner-content p {{
        font-size: 1.4rem;
        max-width: 900px;
        margin-bottom: 30px;
        font-weight: 300;
    }}
    .tool-button {{ 
        padding: 15px 35px;
        font-size: 1.2rem;
        background-color: #3498db; 
        border: 2px solid #3498db; 
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        color: white;
        text-decoration: none;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        animation: subtleFloat 4s ease-in-out infinite; 
    }}
    .tool-button:hover {{
        background-color: #2980b9; 
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
        border-color: #2980b9; 
    }}
    
    .stApp {{ background-color: #f8f9fa; }}
    .main .block-container {{ padding: 2rem 3rem 4rem; }}
    
    /* Hide the default multi-page app sidebar navigation */
    div[data-testid="stSidebarNav"] {{ display: none; }}

</style>
""", unsafe_allow_html=True)


# ==================================
# === HOME PAGE CONTENT ===
# ==================================

# Banner
st.markdown(f"""
<div class="welcome-banner">
    <div class="banner-content">
        <h1>
            Welcome to the Agent Platform 🤖
        </h1>
        <p>
            An intelligent platform offering specialized agent frameworks. Please select an agent framework to begin your session.
        </p>
    </div>
</div>
""", unsafe_allow_html=True) 

# Buttons
st.markdown("<h2 style='text-align: center; margin-top: -100px; margin-bottom: 50px;'>Select Your Agent Framework</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Use st.switch_page for reliable navigation to the page files
# Note: st.switch_page requires Streamlit >= 1.29.0
if 'switch_page' not in st.__dict__:
    st.error("⚠️ Streamlit version too old! Please upgrade to 1.29.0+ for multi-page app navigation to work correctly.")

with col1:
    if st.button("Agno RAG Agent", key="btn_agno", use_container_width=True, help="Use the RAG Agent for UPM Document Q&A."):
        # Navigate using the page file name (without extension)
        st.switch_page("pages/streamlit_ui_file.py")
        

with col2:
    if st.button("CrewAI Multi-Agent", key="btn_crewai", use_container_width=True, help="Simulate a multi-agent team with CrewAI (Placeholder)."):
        # Navigate using the page file name (without extension)
        st.switch_page("pages/2_CrewAI_Platform.py")


with st.sidebar:
    st.header("Platform Info")
    st.info("Select a framework in the main window to begin an agent session.")
    st.markdown("---")
    st.caption("Architecture: Multi-Page App for Dependency Isolation.")