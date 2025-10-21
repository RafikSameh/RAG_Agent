# pages/1_Agno_RAG_Agent.py

# ==================================
# === AGNO-SPECIFIC IMPORTS (REQUIRED) ===
# ==================================
import streamlit as st
import os
from pathlib import Path
import re
import numpy as np 
import os
import asyncio
import constants as c
from src.Agno_API import AgnoAPI



os.environ["PATH"] += os.pathsep + r"C:\\poppler\\Library\bin"
# ==================================
# === GLOBAL CONFIGURATION (UNCHANGED) ===
# ==================================

PDF_PATH = c.PDF_PATH
PDF_PAGES_FOLDER = Path(f"./{c.PDF_PAGES_FOLDER}/default")

# Initialize session state variables
# NOTE: We use "agno_session_id" for the active session
st.session_state.setdefault("agno_session_id", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("show_images", False)
st.session_state.setdefault("uploaded_pdf_path", None) # NEW: Stores the path of the temp uploaded file
st.session_state.setdefault("last_uploaded_filename", None) # NEW: Stores filename to detect changes
st.session_state.setdefault("reload",1)
# -------------------------------------------------------------
# NEW HELPER FUNCTION TO DETERMINE THE ACTIVE DOCUMENT PATH
# -------------------------------------------------------------
def get_current_pdf_path() -> str:
    """Returns the path of the dynamically uploaded PDF, or the default hardcoded path."""
    # Prioritize the uploaded file path if it exists, otherwise use the hardcoded default
    return st.session_state.get("uploaded_pdf_path") or PDF_PATH

# --- Check if user uploaded a NEW file ---
if "last_loaded_pdf" not in st.session_state:
    st.session_state["last_loaded_pdf"] = None

pdf_changed = st.session_state["last_loaded_pdf"] != PDF_PATH

# ==================================
# {3} Image Page Detection (UNCHANGED)
pdf_path = st.session_state.get("last_loaded_pdf") or "default.pdf"
current_pdf = Path(pdf_path).stem
IMAGE_FOLDER = Path(f"./{c.PDF_IMAGE_FOLDER}/{current_pdf}")
PDF_PAGES_FOLDER = Path(f"./{c.PDF_PAGES_FOLDER}/{current_pdf}")

def get_pages_with_images() -> list[int]:
    pages = []
    if IMAGE_FOLDER.exists():
        for f in IMAGE_FOLDER.iterdir():
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                match = re.search(r"(\d+)", f.stem)
                if match:
                    pages.append(int(match.group(1)))
    return sorted(list(set(pages)))
pages_with_images = get_pages_with_images()


# ==================================
# === AGNO CHAT PAGE FUNCTION (The heart of the original app) ===
# ==================================
def agno_chat_page(rag_agent, current_session_id: str, pdf_view_mode):
    st.title("🧠 Agno RAG Agent: Document Q&A")
    st.caption(f"Active Session ID: `{current_session_id}`")
    
    # --- Print Agent Session Data Immediately (from original logic) ---
    st.markdown("---")
    st.subheader(f"Session State Verified: `{current_session_id}...`")

    
    # Display chat history (UNCHANGED)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input (UNCHANGED)
    if prompt := st.chat_input("Ask a question about the document...",width="stretch"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 RAG Agent is analyzing the document..."):
                result = rag_agent.run(prompt, session_id=session_id)
                api.update_session(result,st.session_state['session_id'])
                response_text = getattr(result, "content", None) or getattr(result, "output_text", None)
                if not response_text:
                    response_text = "⚠️ No textual response found."

                # 1. Display the main answer text
                st.markdown(response_text)

                # 2. Extract and display referenced pages (the code you asked about)
                referenced_pages = set()
                # Searches for patterns like 'Pages: 1, 2-5' or 'Page 7'
                page_phrases = re.findall(r"Pages?\s*[:\-]?\s*([\d,\-\– ]+)", response_text, re.IGNORECASE)
                for phrase in page_phrases:
                    # Logic to split by comma/space and handle ranges (like 2-5)
                    parts = re.split(r"[, ]+", phrase.strip())
                    for part in parts:
                        if re.match(r"^\d+$", part):
                            referenced_pages.add(int(part))
                        elif re.match(r"^\d+[\-–]\d+$", part):
                            start, end = re.split(r"[\-–]", part)
                            referenced_pages.update(range(int(start), int(end) + 1))

                # Display the list of detected pages
                st.write(f"📖 Referenced pages detected: {sorted(referenced_pages)}")

                # 3. Display referenced images — only if toggle is ON
                if st.session_state.get("show_images", False):
                    if referenced_pages:
                        # Backend call only
                        image_data = api.get_referenced_images(referenced_pages, pdf_view_mode)

                        # Streamlit visualization here
                        if not image_data["pages"]:
                            st.info("No images found for the referenced pages.")
                        else:
                            st.markdown(f"### {image_data['mode']}")
                            cols = st.columns(min(len(referenced_pages), 4))

                            for idx, (page, paths) in enumerate(image_data["pages"].items()):
                                if paths:
                                    for path in paths:
                                        with cols[idx % 4]:
                                            st.image(path, caption=f"Page {page}", use_container_width=True)
                                else:
                                    with cols[idx % 4]:
                                        st.warning(f"⚠️ No image found for Page {page}")

        # Persist response
        st.session_state.messages.append({"role": "assistant", "content": response_text})


# ==========================
# ⚙️ Session-Safe Initialization
# ==========================

# --- Detect current PDF path (you already have this helper) ---
PDF_PATH = get_current_pdf_path()

# --- Check if user uploaded a NEW file ---
if "last_loaded_pdf" not in st.session_state:
    st.session_state["last_loaded_pdf"] = None

pdf_changed = st.session_state["last_loaded_pdf"] != PDF_PATH


# --- Only reload if first time OR PDF changed ---
if ("agent" not in st.session_state) or pdf_changed: 
    with st.spinner("🚀 Loading RAG agent and models... please wait"):
        # optional small async delay to let Streamlit stabilize
        st.cache_data.clear()
        asyncio.run(asyncio.sleep(0.2))
        api = AgnoAPI(PDF_PATH)
        reload=0
        # initialize your agent (exact same function you already have)
        agent,reload = api.initialize_agent(session_id=st.session_state.get("session_id"), pdf_changed=pdf_changed)
        # store in session_state so it's reused on reruns
        st.session_state["agent"] = agent
        st.session_state['api_call'] = api
        st.session_state["last_loaded_pdf"] = PDF_PATH

    st.success("✅ Agent and models loaded successfully!")

else:
    agent = st.session_state.get("agent")
    api = st.session_state.get('api_call') 
    st.info(f"📄 Using cached agent for `{os.path.basename(PDF_PATH)}`")


# ==========================
# ✅ After this point — agent is ready to use
# ==========================
st.write("✅ Agent is ready! Ask your questions below.")

# ==================================
# === MAIN AGNO PAGE EXECUTION ===
# ==================================

with st.sidebar:
    st.header("👤 Session & Tools")
    st.subheader("➕ Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a new document (PDF, PNG, JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="new_doc_uploader",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # --- STEP 1: Save uploaded file ---
        uploads_dir = Path(f"./{c.UPLOADS_FOLDER}")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        uploaded_path = uploads_dir / uploaded_file.name
        with open(uploaded_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # --- STEP 2: Update session state ---
        if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
            st.session_state["uploaded_pdf_path"] = str(uploaded_path)
            st.session_state["last_uploaded_filename"] = uploaded_file.name
            st.session_state["messages"] = []  # Clear old chat
            st.session_state["agno_session_id"] = None  # Reset agent session
            st.success(f"📄 Uploaded: {uploaded_file.name}")

            # --- STEP 3: Force clean rerun ---
            st.rerun()

    st.markdown("---")

    existing_sessions,session_labels,label_to_session_id = api.Sessions_manager()


    if "session_id" not in st.session_state:
        st.session_state["session_id"] = existing_sessions[0] if existing_sessions else api.create_new_session()
        st.session_state["messages"] = []

    # Select existing session if any
    if existing_sessions:
        try:
            default_index = existing_sessions.index(st.session_state["session_id"])
        except ValueError:
            default_index = 0
        selected_session_label = st.selectbox( # Renamed variable to avoid confusion
            "Select/Load Existing Session:", 
            session_labels,
            index=default_index,
            key="session_selectbox",
            help="Switch between historical chat sessions stored in `agno.db`."
        )
        selected_session = label_to_session_id[selected_session_label] # Use label to look up ID
        if selected_session != st.session_state["session_id"]:
            st.session_state["session_id"] = selected_session
            st.session_state["messages"] = [
                {"role": "assistant", "content": f"Session **`{selected_session}`** loaded. Continue your conversation or ask a new question."}
            ]
            st.rerun()
    else:
        st.info("No historical sessions found. Starting a new one.")
        selected_session = st.session_state["session_id"]

    # Start New Session Button
    if st.button("➕ Start New Session", use_container_width=True):
        session_id = api.create_new_session()
        new_session_id = session_id
        st.session_state["session_id"] = new_session_id
        st.session_state["messages"] = [
            {"role": "assistant", "content": "🆕 **New Session Started.** Your previous history is saved. How can I help you?"}
        ]
        st.success(f"Created new session: `{new_session_id}`")
        st.rerun()

    st.caption(f"Active Session ID: `{st.session_state['session_id']}`")
    
    # Display Knowledge Base Metric
    vector_db = st.session_state.get("agent").knowledge.vector_db
    df = vector_db.table.to_pandas()
    st.metric(label="📊 Knowledge Chunks Loaded", value=len(df))

    # --- 1. NEW TOGGLE BUTTON (MUST COME FIRST) ---
    # The value of this toggle determines the state of the radio buttons
    st.session_state["show_images"] = st.toggle(
        label="🖼️ **Show Referenced Images**", 
        value=st.session_state["show_images"], 
        help="Toggle to show/hide images for referenced pages in all assistant messages."
    )
    
    # Determine the disabled state based on the toggle value
    # If show_images is False, we set disabled=True (meaning they are disabled)
    is_visualization_disabled = not st.session_state["show_images"]

    # --- 2. PDF Visualization Mode Switch (CONDITIONAL) ---
    st.session_state["pdf_view_mode"] = st.radio(
        "📑 PDF Visualization Mode",
        ["Extracted Images", "Full PDF Pages"],
        index=0 if st.session_state.get("pdf_view_mode", "Extracted Images") == "Extracted Images" else 1,
        help="Choose whether to show cropped images from the PDF or full rendered pages when pages are referenced.",
        # KEY CHANGE: Disable the radio buttons if the main toggle is off
        disabled=is_visualization_disabled 
    )

    # Display raw session data
    with st.expander("🔑 Raw Session Data (agno_sessions)"):
        
        session_data = api.get_session_data(st.session_state["session_id"])
        if "error" in session_data:
            st.error(session_data["error"])
        else:
            st.json(session_data)




# 4️⃣ Direct Agent Initialization
# Use the already-loaded agent from session_state
rag_agent = st.session_state["agent"]
session_id = st.session_state["session_id"]


# Ensure messages are loaded when first entering the page
# This logic ensures messages are retrieved for the active session ID
try:
    db_messages = api.get_session_messages(session_id)

    if db_messages:
        st.session_state["messages"] = db_messages
    else:
        st.session_state["messages"] = [
            {"role": "assistant", "content": f"Hello! I'm your RAG agent for the UPM document. Ask me anything about the content. (Active Session: `{session_id}`)"}
        ]
except Exception as e:
    st.warning(f"⚠️ Could not retrieve messages from DB: {e}")
    st.session_state["messages"] = [
        {"role": "assistant", "content": f"Hello! I'm your RAG agent for the UPM document. Ask me anything about the content. (Active Session: `{session_id}`)"}
    ]


agno_chat_page(rag_agent, session_id, st.session_state.get("pdf_view_mode"))