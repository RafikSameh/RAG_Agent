import streamlit as st
import sys
import time
from pathlib import Path
from src.database import SessionManager, init_database, cleanup_orphaned_embeddings_on_startup
from src.chatbot import query_document


# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# ==================================
# PAGE CONFIGURATION 
# ==================================
st.set_page_config(
    page_title="CrewAI Document Q&A Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================
# DATABASE INITIALIZATION 
# ==================================
init_database()

# Run cleanup on startup (only once)
if "startup_cleanup_done" not in st.session_state:
    cleanup_orphaned_embeddings_on_startup()
    st.session_state.startup_cleanup_done = True

# ==================================
# LOAD STYLES 
# ==================================
import os

try:
    styles_path = os.path.join(os.path.dirname(__file__), "styles.css")
    with open(styles_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Styles file not found")

# ==================================
# INITIALIZE SESSION STATE 
# ==================================
def initialize_session_state():
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    if "last_loaded_session" not in st.session_state:
        st.session_state.last_loaded_session = None

    if "chatbot_system" not in st.session_state:
        st.session_state.chatbot_system = None
        st.session_state.initialized = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None

    if "chat_renamed" not in st.session_state:
        st.session_state.chat_renamed = False

    # Initialize sessions
    existing_sessions = st.session_state.session_manager.list_sessions()
    if not existing_sessions:
        st.session_state.current_session_id = st.session_state.session_manager.create_session()
    elif not st.session_state.current_session_id:
        st.session_state.current_session_id = existing_sessions[0]["id"]

initialize_session_state()

# ==================================
# LOAD MESSAGES AND CHATBOT SYSTEM WHEN SESSION CHANGES
# ==================================
def load_session_data(session_id):
    """Load messages and initialize chatbot system from database."""
    manager = st.session_state.session_manager
    loaded_session = manager.load_session(session_id)
    
    messages = []
    doc_name = None
    chatbot_system = None
    
    if loaded_session:
        # Load messages
        session_messages = manager.get_session_messages(session_id)
        
        for msg in session_messages:
            messages.append({
                "role": "user",
                "content": msg["question"]
            })
            messages.append({
                "role": "assistant",
                "content": msg["answer"],
                "time": msg.get("time_taken", None)
            })
        
        doc_name = loaded_session.get("document_name", None)
        
        # Try to reload chatbot system from existing embeddings
        if doc_name:
            try:
                from src.chatbot import reload_system_from_session
                chatbot_system = reload_system_from_session(session_id, doc_name)
                
                if chatbot_system:
                    print(f"✓ Chatbot system reloaded from existing data")
                else:
                    print(f"⚠ Could not reload chatbot system - embeddings may not exist")
            except Exception as e:
                print(f"⚠ Error reloading chatbot system: {e}")
                chatbot_system = None
    
    return messages, doc_name, chatbot_system

# Only load if session changed
if st.session_state.current_session_id != st.session_state.last_loaded_session:
    st.session_state.last_loaded_session = st.session_state.current_session_id
    messages, doc_name, chatbot_system = load_session_data(st.session_state.current_session_id)
    
    st.session_state.messages = messages
    st.session_state.uploaded_file_name = doc_name
    
    # Set chatbot system and initialized flag
    if chatbot_system:
        st.session_state.chatbot_system = chatbot_system
        st.session_state.initialized = True
        print(f"✓ Session fully loaded: {len(messages)} messages, chatbot ready")
    else:
        st.session_state.chatbot_system = None
        st.session_state.initialized = False
        print(f"✓ Session loaded: {len(messages)} messages, awaiting file upload")

# ==================================
# HELPER FUNCTIONS 
# ==================================
def render_message(message):
    """Render a single chat message with optional images."""
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">You</div>
            <div class="message-content">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-header">Assistant</div>
            <div class="message-content">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if "time" in message and message["time"] is not None:
            st.markdown(f"""
            <div class="stats-box">⏱️ Time: {message["time"]:.2f}s</div>
            """, unsafe_allow_html=True)
        
        # Display images if available
        if "images" in message and message["images"]:
            image_data = message["images"]
            pages = image_data.get("pages", {})
            
            if pages:
                st.markdown("---")
                st.markdown("**📎 Referenced Content:**")
                
                for page, image_paths in pages.items():
                    if image_paths:
                        st.markdown(f"**Page {page}:**")
                        # Display images in columns for better layout
                        if len(image_paths) > 1:
                            cols = st.columns(min(len(image_paths), 3))
                            for idx, img_path in enumerate(image_paths):
                                with cols[idx % 3]:
                                    try:
                                        st.image(img_path, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"Could not load image: {img_path}")
                        else:
                            try:
                                st.image(image_paths[0], use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not load image: {image_paths[0]}")

def display_chat_history():
    """Display all chat messages."""
    for message in st.session_state.messages:
        render_message(message)

def generate_dynamic_title(prompt, chatbot_system, session_manager):
    """Generate and save a dynamic chat title."""
    try:
        from src.chatbot import query_document
        
        title_prompt = f"Create a very short title (3-5 words max) for a chat about: {prompt}. Just the title, nothing else."
        
        title_answer, _, _ = query_document(
            title_prompt,
            chatbot_system["agent"],
            chatbot_system["crew"],
            chatbot_system["search_tool"],
            ""
        )
        
        new_title = title_answer.strip()[:50]
        session_manager.update_session_title(
            st.session_state.current_session_id,
            new_title
        )
        st.session_state.chat_renamed = True
        
    except Exception as e:
        print(f"Title generation error: {str(e)}")
        try:
            new_title = prompt[:50]
            session_manager.update_session_title(
                st.session_state.current_session_id,
                new_title
            )
            st.session_state.chat_renamed = True
        except:
            pass

def safe_cleanup_system(chatbot_system):
    """Safely cleanup chatbot system without raising exceptions."""
    if chatbot_system:
        try:
            from src.chatbot import cleanup_system
            cleanup_system(chatbot_system)
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

def initialize_chatbot(uploaded_file):
    """Initialize chatbot with uploaded file."""
    try:
        from src.chatbot import initialize_system_with_file
        
        safe_cleanup_system(st.session_state.chatbot_system)
        
        chatbot_system = initialize_system_with_file(uploaded_file, session_id=st.session_state.current_session_id)
        if chatbot_system:
            st.session_state.chatbot_system = chatbot_system
            st.session_state.initialized = True
            st.success("✅ Document processed successfully! You can now ask questions.")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Failed to process document.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def process_user_query(prompt):
    """Process user query and generate response."""
    try:
        from src.chatbot import query_document
        
        if not st.session_state.chatbot_system:
            st.error("Document not loaded. Please re-upload to continue.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">You</div>
            <div class="message-content">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Generating response..."):
            context_str = st.session_state.session_manager.get_conversation_history(
                st.session_state.current_session_id,
                max_entries=3
            )

            # Get answer and referenced pages
            answer, response_time, referenced_pages = query_document(
                prompt,
                st.session_state.chatbot_system["agent"],
                st.session_state.chatbot_system["crew"],
                st.session_state.chatbot_system["search_tool"],
                context_str,
                view_mode=st.session_state.image_mode
            )
            
            # Get images using your original function via the wrapper
            image_data = None
            if st.session_state.image_mode != "None" and referenced_pages:
                image_handler = st.session_state.chatbot_system.get("image_handler")
                if image_handler:
                    # Call get_images which uses your original get_referenced_images
                    image_data = image_handler.get_images(
                        referenced_pages, 
                        view_mode=st.session_state.image_mode
                    )
                    print(f"🖼️ Image data: {image_data}")
            
            # Save message with images
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "time": response_time,
                "images": image_data  # Store image data with message
            })
            
            st.session_state.session_manager.save_message(
                st.session_state.current_session_id,
                prompt,
                answer,
                response_time
            )

            # Generate dynamic title on first question
            if not st.session_state.chat_renamed and len(st.session_state.messages) == 2:
                generate_dynamic_title(prompt, st.session_state.chatbot_system, st.session_state.session_manager)
            
            st.rerun()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# ==================================
# SIDEBAR
# ==================================
with st.sidebar:
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = st.session_state.session_manager.create_session()
        st.session_state.messages = []
        st.session_state.chatbot_system = None
        st.session_state.initialized = False
        st.session_state.uploaded_file_name = None
        st.session_state.chat_renamed = False
        st.session_state.last_loaded_session = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Chat History")
    
    sessions = st.session_state.session_manager.list_sessions()
    
    if not sessions:
        st.info("No previous chats")
    else:
        for session in sessions:
            is_current = session['id'] == st.session_state.current_session_id
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    button_label = f"📄 {session['title']}"
                    if is_current:
                        button_label = f"✓ {button_label}"
                    
                    if st.button(button_label, key=f"load_{session['id']}", use_container_width=True):
                        st.session_state.current_session_id = session['id']
                        st.session_state.last_loaded_session = None
                        st.session_state.chatbot_system = None
                        st.session_state.initialized = False
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_{session['id']}"):
                        # Delete from database immediately
                        st.session_state.session_manager.delete_session(session['id'])

                        # If this was the current session, reset UI state
                        if st.session_state.current_session_id == session['id']:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                            st.session_state.last_loaded_session = None
                        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.current_session_id:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.chatbot_system:
                    try:
                        st.session_state.chatbot_system["crew"].memory.reset()
                    except:
                        pass
                st.success("Chat cleared!")
                time.sleep(0.5)
                st.rerun()
        
        with col2:
            if st.button("🏠 Home", use_container_width=True):
                st.switch_page("app.py")
    
    st.markdown("---")
    st.markdown("### 🖼️ Image Mode")

    image_mode = st.radio(
        "Choose how to display visuals:",
        ["None", "Extracted Images", "Full PDF Pages"],
        index=0,
        help="Choose 'Extracted Images' to show detected figures or 'Full PDF Pages' to show page thumbnails."
    )

    st.session_state.image_mode = image_mode


# ==================================
# MAIN CONTENT 
# ==================================
st.title("🤖 CrewAI Document Q&A Platform")

# Get current session info
current_session_data = st.session_state.session_manager.load_session(st.session_state.current_session_id)
saved_doc_name = current_session_data.get("document_name") if current_session_data else None

# Show document upload section
st.markdown("### 📁 Document Upload")

# If chatbot is already initialized from previous session
if st.session_state.initialized and st.session_state.chatbot_system:
    st.markdown(f"""
    <div class="success-box">
    ✅ <strong>Document Loaded:</strong> {saved_doc_name}<br>
    🔄 <strong>Status:</strong> Ready to chat (loaded from previous session)
    </div>
    """, unsafe_allow_html=True)
    
    # Optional: Allow re-upload to refresh
    with st.expander("🔄 Re-upload document (optional)"):
        uploaded_file = st.file_uploader(
            "Upload a new version of the document",
            type=["pdf", "docx", "doc"],
            key=f"uploader_{st.session_state.current_session_id}",
        )
        
        if uploaded_file:
            if uploaded_file.name != saved_doc_name:
                st.warning(f"⚠️ You're uploading {uploaded_file.name} but this chat was about {saved_doc_name}")
            
            if st.button("🔄 Refresh with new file"):
                initialize_chatbot(uploaded_file)
else:
    # Need file upload
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file" + (f" (Expected: {saved_doc_name})" if saved_doc_name else ""),
        type=["pdf", "docx", "doc"],
        key=f"uploader_{st.session_state.current_session_id}",
        help="Upload a PDF or Word document"
    )
    
    # Check for wrong document
    if uploaded_file is not None:
        file_name = uploaded_file.name
        
        if saved_doc_name and file_name != saved_doc_name:
            st.markdown(f"""
            <div class="warning-box">
            ⚠️ <strong>Warning:</strong> This chat was about <strong>{saved_doc_name}</strong><br>
            You're trying to upload <strong>{file_name}</strong> (different document)<br><br>
            <strong>Options:</strong><br>
            1. Upload <strong>{saved_doc_name}</strong> to continue this conversation<br>
            2. Create a "New Chat" from the sidebar to ask about <strong>{file_name}</strong>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = None
    
    # Initialize chatbot if file uploaded
    if uploaded_file is not None:
        file_name = uploaded_file.name
        
        if not saved_doc_name:
            st.session_state.session_manager.update_document_name(
                st.session_state.current_session_id, 
                file_name
            )
        
        if not st.session_state.initialized or st.session_state.uploaded_file_name != file_name:
            st.session_state.uploaded_file_name = file_name
            
            file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
            st.markdown(f"""
            <div class="upload-box">
            ✅ <strong>File Loaded:</strong> {file_name}<br>
            📊 <strong>Size:</strong> {file_size_mb:.2f} MB<br>
            🔄 <strong>Status:</strong> Processing...
            </div>
            """, unsafe_allow_html=True)
            
            initialize_chatbot(uploaded_file)

# Chat interface (always show if initialized or has messages)
if st.session_state.initialized or st.session_state.messages:
    st.markdown("### 💬 Chat Interface")
    
    if st.session_state.messages:
        display_chat_history()
    else:
        st.markdown("""
        <div class="info-box">
        Ask me anything about the uploaded document!
        </div>
        """, unsafe_allow_html=True)
    
    # Only allow input if chatbot is initialized
    if st.session_state.initialized:
        if prompt := st.chat_input("Ask a question..."):
            process_user_query(prompt)
    else:
        st.info("💡 Please upload the document to continue chatting")
else:
    st.info("📤 Upload a document to get started")