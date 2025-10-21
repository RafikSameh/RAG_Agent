# Agno RAG Agent - File Separation Architecture

## Overview
The original monolithic `AGNO_API.py` has been separated into two focused files following the **separation of concerns** principle:

1. **agno_api.py** - Framework Implementation & Business Logic
2. **streamlit_app.py** - UI/Design Layer & User Interactions

---

## 📁 File Structure

```
project/
├── api/agno_api.py      # API & Framework Implementation
├── streamlit_app.py     # Streamlit UI & Design
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🔧 agno_api.py - API Layer

**Purpose:** Contains all Agno framework interactions and business logic.

### Key Sections:

#### 1. **Imports**
- All Agno-specific imports (Agent, Knowledge, LanceDb, etc.)
- PDF processing libraries (fitz, pdf2image)
- Utility libraries (sqlite3, uuid, etc.)

#### 2. **Global Configuration**
- Database paths and model IDs
- File paths for PDFs
- API keys and environment variables

#### 3. **PDF Processing Functions**
```python
render_pdf_pages(pdf_path, out_dir)      # Convert PDF to JPEG images
extract_pdf_images(pdf_path, save_dir)   # Extract embedded images
```

#### 4. **LanceDB & Vector Store**
```python
safe_lancedb(table_name, uri, embedder)  # Initialize vector database
```

#### 5. **Model Loading**
```python
load_hf_model()                           # Load HuggingFace model with caching
```

#### 6. **Agent Initialization**
```python
initialize_agent(pdf_path)                # Create and configure RAG agent
```

#### 7. **Session Management**
```python
get_session_data(session_id)              # Retrieve session from DB
create_new_session()                      # Create new agent session
get_all_sessions()                        # Fetch all sessions
```

#### 8. **Query & Response Handling**
```python
run_agent_query(agent, session_id, prompt)        # Execute agent query
extract_referenced_pages(response_text)            # Parse page numbers
get_pages_with_images(pdf_stem)                    # List pages with images
persist_messages_to_db(session_id, messages)       # Save messages
load_messages_from_db(session_id)                  # Load message history
```

### Usage in streamlit_app.py:
```python
from agno_api import (
    initialize_agent,
    run_agent_query,
    extract_referenced_pages,
    # ... other functions
)
```

---

## 🎨 streamlit_app.py - UI Layer

**Purpose:** Handles all Streamlit UI components and calls API functions.

### Key Sections:

#### 1. **Page Configuration**
```python
st.set_page_config()                      # Streamlit page settings
```

#### 2. **Session State Initialization**
Initialize all session variables with defaults.

#### 3. **Sidebar Components**

**Upload Section:**
```python
render_sidebar_upload_section()           # Document upload interface
```

**Session Management:**
```python
render_sidebar_session_section()          # Session selector & new session button
```

**Visualization Options:**
```python
render_sidebar_visualization_section()    # Toggle images & view mode
```

**Session Data Display:**
```python
render_sidebar_session_data()             # Show raw session info
```

#### 4. **Main Chat Interface**
```python
render_chat_messages()                    # Display message history
render_chat_input(agent, session_id)      # Handle user input & display response
```

#### 5. **Image Rendering**
```python
render_referenced_images(referenced_pages)        # Show all referenced images
render_extracted_images(image_folder, pages)      # Show extracted images
render_full_pdf_pages(pdf_pages_folder, pages)    # Show full page previews
```

#### 6. **Agent Loading**
```python
load_or_cache_agent()                     # Load with caching & PDF change detection
```

#### 7. **Message Loading**
```python
load_initial_messages(session_id)         # Load chat history from DB
```

#### 8. **Main Execution**
```python
def main():                               # Orchestrate all UI components
```

---

## 🔄 Data Flow

```
User Input
    ↓
streamlit_app.py (render_chat_input)
    ↓
agno_api.py (run_agent_query)
    ↓
Agent processes query
    ↓
agno_api.py (extract_referenced_pages)
    ↓
streamlit_app.py (render_referenced_images)
    ↓
Display to User
    ↓
agno_api.py (persist_messages_to_db)
```

---

## 📋 Function Mapping

### From Original → Split Into:

| Original Function | New Location | Purpose |
|---|---|---|
| `render_pdf_pages()` | agno_api.py | PDF processing |
| `extract_pdf_images()` | agno_api.py | Image extraction |
| `safe_lancedb()` | agno_api.py | Vector DB setup |
| `load_hf_model()` | agno_api.py | Model loading |
| `initialize_agent()` | agno_api.py | Agent setup |
| `get_session_data()` | agno_api.py | DB queries |
| `create_new_session()` | agno_api.py | Session creation |
| `agno_chat_page()` | streamlit_app.py | Main chat rendering |
| Chat history display | streamlit_app.py | Message rendering |
| Sidebar rendering | streamlit_app.py | UI components |
| Image display logic | streamlit_app.py | Image rendering |

---

## ⚙️ Installation & Setup

### 1. Install Dependencies
```bash
pip install streamlit agno lancedb transformers pdf2image torch
```

### 2. Set Environment Variables
```bash
export HF_API_KEY="your_huggingface_api_key"
```

### 3. Configure Paths
Edit **agno_api.py**:
```python
PDF_PATH = "path/to/your/document.pdf"
MODEL_ID = "your/model/id"
```

### 4. Run Application
```bash
streamlit run streamlit_app.py
```

---

## 🎯 Benefits of Separation

### ✅ Maintainability
- Each file has a single responsibility
- Easier to locate and modify specific functionality

### ✅ Testability
- API functions can be tested independently
- Mock Streamlit UI without running full app

### ✅ Reusability
- agno_api.py can be imported by other applications
- No Streamlit dependency in API layer

### ✅ Scalability
- Easy to add new features without cluttering files
- Clear interfaces between layers

### ✅ Readability
- Smaller files are easier to understand
- Clear section organization

---

## 🔌 Integration Points

### streamlit_app.py imports from agno_api.py:
```python
from agno_api import (
    initialize_agent,
    get_current_pdf_path,
    get_session_data,
    create_new_session,
    get_all_sessions,
    run_agent_query,
    extract_referenced_pages,
    persist_messages_to_db,
    load_messages_from_db,
    AGENT_DATABASE,
    PDF_PATH,
)
```

### No circular imports
- agno_api.py does NOT import from streamlit_app.py
- Clean unidirectional dependency

---

## 📝 Configuration Files

### requirements.txt
```
streamlit>=1.28.0
agno>=0.1.0
lancedb>=0.1.0
transformers>=4.30.0
pdf2image>=1.16.0
PyMuPDF>=1.23.0
torch>=2.0.0
huggingface-hub>=0.16.0
pandas>=1.5.0
```

---

## 🚀 Extension Guide

### Adding New API Function
1. Create function in **agno_api.py**
2. Import in **streamlit_app.py**
3. Call from UI rendering function

### Adding New UI Component
1. Create rendering function in **streamlit_app.py**
2. Call from `main()` or sidebar
3. Use imported API functions as needed

---

## 📊 File Statistics

| Aspect | Before | After |
|---|---|---|
| Single File | ~600 lines | Split into 2 files |
| Complexity | High | Low per file |
| Functions | Mixed | Organized by layer |
| Dependencies | All mixed | Organized imports |

---

## ✅ Testing Approach

### Test agno_api.py (No Streamlit)
```python
import agno_api

# Test PDF processing
agno_api.render_pdf_pages("test.pdf", "output/")

# Test session management
session_id = agno_api.create_new_session()
data = agno_api.get_session_data(session_id)

# Test page extraction
pages = agno_api.extract_referenced_pages("Pages: 1, 3-5")
```

### Test streamlit_app.py
```bash
streamlit run streamlit_app.py
```

---

## 🔐 Security Notes

- **API Keys**: Store in environment variables, not in code
- **Database**: SQLite file (`agno.db`) should be in `.gitignore`
- **Uploads**: Temporary files stored in `./tmp_2/uploads/`

---

## 📞 Support

For issues or questions:
1. Check function docstrings
2. Review data flow diagram above
3. Check configuration in **agno_api.py**

---

## 📄 License
Created by:

Rafik Sameh Yanni

Nour Ahmed Fouad