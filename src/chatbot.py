import os
import time
import tempfile
import shutil
from pathlib import Path
from chromadb.config import Settings
from crewai.rag.config.utils import set_rag_config
from dotenv import load_dotenv
from crewai.utilities.paths import db_storage_path
from crewai.rag.chromadb.config import ChromaDBConfig
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import DOCXSearchTool, PDFSearchTool
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import fitz  

# ============================================================================
# Image Handler Wrapper 
# ============================================================================
class ImageHandler:
    """
    Wrapper to make utils.images.get_referenced_images work as an instance method.
    This allows us to use your original function without modification.
    """
    def __init__(self, pdf_images_dir, pdf_pages_dir):
        self.pdf_images_dir = Path(pdf_images_dir)
        self.pdf_pages_dir = Path(pdf_pages_dir)
    
    # Just delegate to your original function - no rewriting!
    def get_images(self, referenced_pages, view_mode="Extracted Images"):
        """Wrapper that calls your original get_referenced_images function"""
        return self.get_referenced_images(referenced_pages, view_mode)
    
    def get_referenced_images(self, referenced_pages, view_mode="Extracted Images"):
        """
        Returns a dictionary of image paths for the referenced pages.
        Does NOT display anything (pure backend logic).
        """
        result = {"mode": view_mode, "pages": {}}

        if not referenced_pages:
            return result  # empty dict

        if view_mode == "Extracted Images":
            if not self.pdf_images_dir.exists():
                return result

            for page in sorted(referenced_pages):
                pattern = re.compile(rf"^page{page}_img\d+\.(jpg|jpeg|png)$", re.IGNORECASE)
                matching_images = [
                    str(f) for f in self.pdf_images_dir.iterdir() if pattern.match(f.name)
                ]
                result["pages"][page] = matching_images

        elif view_mode == "Full PDF Pages":
            if not self.pdf_pages_dir.exists():
                return result

            for page in sorted(referenced_pages):
                page_path = self.pdf_pages_dir / f"page{page}.jpg"
                if page_path.exists():
                    result["pages"][page] = [str(page_path)]
                else:
                    result["pages"][page] = []

        return result

# ============================================================================
# Image Extraction Function
# ============================================================================
def extract_pdf_images(pdf_path, session_id, base_dir):
    """Extract images from PDF and save to session-specific directories."""
    base_path = Path(base_dir) / session_id
    extracted_images_dir = base_path / "extracted_images"
    pages_dir = base_path / "pages"
    
    extracted_images_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
        
        # Extract individual images from each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Save extracted image
                    image_path = extracted_images_dir / f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                except Exception as e:
                    print(f" Could not extract image {img_index} from page {page_num + 1}: {e}")
            
            # Render full page as image
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  
                page_image_path = pages_dir / f"page{page_num + 1}.jpg"
                pix.save(str(page_image_path))
            except Exception as e:
                print(f" Could not render page {page_num + 1}: {e}")
        
        doc.close()
        print(f" Extracted images to: {extracted_images_dir}")
        print(f" Rendered pages to: {pages_dir}")
        
        return extracted_images_dir, pages_dir
        
    except Exception as e:
        print(f" Error extracting images: {e}")
        return extracted_images_dir, pages_dir

# ============================================================================
# Configuration 
# ============================================================================

DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_MODEL_NAME = os.getenv("MODEL")

try:
    load_dotenv()
    env_token = os.getenv("HF_TOKEN")
    env_model = os.getenv("MODEL")

    # Set HF_TOKEN
    if env_token and env_token.strip():
        HF_TOKEN = env_token
        print(" Using HF_TOKEN from .env")
    else:
        HF_TOKEN = DEFAULT_HF_TOKEN
        print(" Using default HF_TOKEN (not found in .env)")

    # Set MODEL_NAME
    if env_model and env_model.strip():
        if not env_model.startswith("huggingface/"):
            MODEL_NAME = f"huggingface/{env_model}"
        else:
            MODEL_NAME = env_model
        print(" Using MODEL from .env")
    else:
        MODEL_NAME = DEFAULT_MODEL_NAME
        print(" Using default MODEL_NAME (not found in .env)")

except Exception as e:
    print(f" Error loading .env: {e}")
    HF_TOKEN = DEFAULT_HF_TOKEN
    MODEL_NAME = DEFAULT_MODEL_NAME
    print(" Using default fallback values")

print(f" Final MODEL_NAME: {MODEL_NAME}")



# ============================================================================
# Get Project Root Directory
# ============================================================================
def get_embeddings_dir():
    """
    Get the embeddings directory relative to the project root.
    Works across different systems.
    """
    # Get the directory where this script is located
    project_root = Path(__file__).parent.parent.absolute()
    
    # Create embeddings folder in project root
    embeddings_dir = project_root / "data" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Project root: {project_root}")
    print(f"✓ Embeddings directory: {embeddings_dir}")
    
    return embeddings_dir

def get_images_dir():
    """Get the images directory relative to the project root."""
    project_root = Path(__file__).parent.parent.absolute()
    images_dir = project_root / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir

def get_documents_dir():
    """Get the documents directory for storing uploaded files permanently."""
    project_root = Path(__file__).parent.parent.absolute()
    documents_dir = project_root / "data" / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    return documents_dir
# ============================================================================
# Create LLM 
# ============================================================================
def create_llm():
    
    try:
        llm = LLM(
            model=MODEL_NAME,  
            api_key=HF_TOKEN,
            request_timeout=120
        )
        return llm
    except Exception as e:
        print(f" Error loading model: {e}")
        raise

# ============================================================================
# Dynamic Search Tool 
# ============================================================================
def create_search_tool(file_path, session_id=None):
    """
    Create appropriate search tool based on file type.
    Each session gets its own ChromaDB persist directory to avoid embedding contamination.
    
    Args:
        file_path: Path to the PDF or DOCX file
        session_id: Unique session identifier (uses UUID if not provided)
    
    Returns:
        Initialized search tool (PDFSearchTool or DOCXSearchTool)
    """
    file_ext = Path(file_path).suffix.lower()

    # Get base embedding directory (project-relative)
    base_embedding_dir = get_embeddings_dir()
    
    # Create session-specific directory
    session_persist_dir = base_embedding_dir / session_id
    session_persist_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Creating search tool for: {file_ext}")
    print(f" Embedding directory: {session_persist_dir}")
    print(f" Session ID: {session_id}")

    try:
        # Create embedding function
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create custom ChromaDB settings for this session
        custom_settings = Settings(
            persist_directory=str(session_persist_dir),
            is_persistent=True,
            allow_reset=True
        )
        
        # Manually construct ChromaDBConfig with custom settings
        chroma_config = object.__new__(ChromaDBConfig)
        object.__setattr__(chroma_config, 'provider', 'chromadb')
        object.__setattr__(chroma_config, 'embedding_function', embedding_function)
        object.__setattr__(chroma_config, 'settings', custom_settings)
        
        # Create tool based on file type
        if file_ext == '.pdf':
            tool = PDFSearchTool(
                pdf=file_path,
                config=chroma_config
            )
        elif file_ext in ['.docx', '.doc']:
            tool = DOCXSearchTool(
                docx=file_path,
                config=chroma_config
            )
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        print(" Search tool initialized successfully")
        return tool, session_persist_dir
        
    except Exception as e:
        print(f" Error creating search tool: {e}")
        raise

# ============================================================================
# Create Agent
# ============================================================================
def create_agent(search_tool, llm, document_name):
 
    agent = Agent(
        role="Document Expert",
        goal=f"Find and provide accurate answers from the uploaded document ({document_name}) using the search tool effectively.",
        backstory="""You are an expert document analyst with access to search functionality.

CRITICAL SEARCH INSTRUCTIONS:
1. You MUST use the document search tool for EVERY question - it's your primary information source
2. Try multiple search queries if the first one doesn't return good results:
   - Search for keywords from the question
   - Try related terms and synonyms
   - Search for broader or narrower terms
3. The search tool returns relevant sections from the document - READ THEM CAREFULLY
4. Based on what the search tool returns, provide a clear and concise answer without unnecessary details
5. If the search returns results, use them! Don't say information is unavailable when it's in the search results
6. For follow-up questions, check the conversation history AND search the document again
7. ONLY say information is unavailable if multiple search attempts truly return nothing relevant

Remember: The search tool is reliable - if it returns content, that content is from the document!""",
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    
    return agent

# ============================================================================
#  Build Crew
# ============================================================================
def create_crew(agent):
    
    crew = Crew(
        agents=[agent],
        tasks=[],
        process=Process.sequential,
        verbose=True,
        embedder={
            "provider": "sentence-transformer",
            "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
        },
        project_name="dynamic_document_chatbot"
    )
    
    return crew

# ============================================================================
# Query Function
# ============================================================================
def query_document(question, agent, crew, search_tool, context_str, view_mode=None):
    """Query the document and return answer with referenced pages"""
    try:

        task_description = f"""TASK: Answer the question about the uploaded document.

{context_str}

CURRENT QUESTION: {question}

MANDATORY STEPS - FOLLOW EXACTLY:
1. Use the document search tool RIGHT NOW to search for information about this question
2. Extract keywords from the question and search for them
3. If first search doesn't help, try searching with different related keywords
4. Read the search results carefully - they contain content from the actual document
5. Based on what you find in the search results, write a comprehensive answer
6. If this is a follow-up question (references "it", "that", "more", etc.), also consider the previous conversation context above
7. Provide specific details from what you found
8. ONLY if you truly searched multiple times and found absolutely nothing relevant, then state the information is not available

IMPORTANT: The search tool works! If it returns results, use them in your answer. Do not ignore search results."""

        current_task = Task(
            description=task_description,
            agent=agent,
            expected_output="""A detailed, accurate answer containing:
1. Information found from searching the document
2. Specific details and facts from the document
3. Reference to previous conversation if this is a follow-up question
4. Clear statement only if information genuinely cannot be found after multiple search attempts""",
            tools=[search_tool]
        )
        
        # Execute
        crew.tasks = [current_task]
        start_time = time.time()
        
        # Perform search to get referenced pages
        search_results = search_tool.run(question)
        
        result = crew.kickoff()
        end_time = time.time()
        
        answer = str(result).strip()
        response_time = end_time - start_time

        # Extract referenced pages from search results
        referenced_pages = set()
        try:
            if isinstance(search_results, str):
                import re
                page_matches = re.findall(r'[Pp]age[:\s]+(\d+)', search_results)
                referenced_pages = {int(p) for p in page_matches}
        except Exception as e:
            print(f" Could not extract page numbers: {e}")
        
        print(f" Referenced pages: {referenced_pages}")
        
        return answer, response_time, referenced_pages
        
    except Exception as e:
        print(f" Error querying document: {e}")
        raise

# ============================================================================
#  Initialize System with Uploaded File
# ============================================================================
def initialize_system_with_file(uploaded_file, session_id=None):
    """
    Initialize chatbot system with uploaded file.
    Saves file permanently in data/documents/[session_id]/ directory.
    """
    try:
        # Create permanent storage directory for this session
        documents_dir = get_documents_dir()
        session_doc_dir = documents_dir / session_id
        session_doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file permanently with original name
        permanent_file_path = session_doc_dir / uploaded_file.name
        with open(permanent_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        print(f" File saved permanently: {permanent_file_path}")
        
        # Extract images if PDF
        image_handler = None
        if permanent_file_path.suffix.lower() == '.pdf':
            images_base_dir = get_images_dir()
            pdf_images_dir, pdf_pages_dir = extract_pdf_images(
                str(permanent_file_path), 
                session_id, 
                images_base_dir
            )
            # Create ImageHandler that uses your original function
            image_handler = ImageHandler(pdf_images_dir, pdf_pages_dir)
            print(f" Images extracted and ImageHandler created")
        
        # Create LLM
        llm = create_llm()
        print(" LLM initialized")
        
        # Create search tool with permanent file path
        search_tool, persist_dir = create_search_tool(str(permanent_file_path), session_id=session_id)
        
        # Create agent
        agent = create_agent(search_tool, llm, uploaded_file.name)
        print(" Agent created")
        
        # Create crew
        crew = create_crew(agent)
        print(" Crew created")
        
        return {
            "llm": llm,
            "agent": agent,
            "crew": crew,
            "search_tool": search_tool,
            "embedding_dir": persist_dir,
            "image_handler": image_handler,
            "file_path": str(permanent_file_path),  # Store permanent path
            "document_name": uploaded_file.name
        }
        
    except Exception as e:
        print(f" Error initializing system: {e}")
        raise

        
# ============================================================================
#  Reload System
# ============================================================================       

def reload_system_from_session(session_id, document_name):
    """
    Reload chatbot system from existing embeddings, images, and stored file.
    No file upload needed - uses saved data from previous session.
    
    Args:
        session_id: The session ID to reload
        document_name: Name of the original document
    
    Returns:
        dict: Initialized chatbot system or None if data doesn't exist
    """
    try:
        # Check if embeddings exist for this session
        base_embedding_dir = get_embeddings_dir()
        session_persist_dir = base_embedding_dir / session_id
        
        if not session_persist_dir.exists():
            print(f" No embeddings found for session: {session_id}")
            return None
        
        print(f" Found existing embeddings at: {session_persist_dir}")
        
        #  Find the stored document file
        documents_dir = get_documents_dir()
        session_doc_dir = documents_dir / session_id
        stored_file_path = session_doc_dir / document_name
        
        if not stored_file_path.exists():
            print(f" Original document file not found: {stored_file_path}")
            print(f"   The file may have been deleted or moved.")
            return None
        
        print(f" Found original document: {stored_file_path}")
        
        # Check if images exist (for PDFs)
        image_handler = None
        images_base_dir = get_images_dir()
        session_images_base = images_base_dir / session_id
        
        if session_images_base.exists():
            pdf_images_dir = session_images_base / "extracted_images"
            pdf_pages_dir = session_images_base / "pages"
            
            if pdf_images_dir.exists() and pdf_pages_dir.exists():
                image_handler = ImageHandler(pdf_images_dir, pdf_pages_dir)
                print(f" Found existing images at: {session_images_base}")
        
        # Create LLM
        llm = create_llm()
        print(" LLM initialized")
        
        # Recreate search tool pointing to existing embeddings and file
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        custom_settings = Settings(
            persist_directory=str(session_persist_dir),
            is_persistent=True,
            allow_reset=True
        )
        
        chroma_config = object.__new__(ChromaDBConfig)
        object.__setattr__(chroma_config, 'provider', 'chromadb')
        object.__setattr__(chroma_config, 'embedding_function', embedding_function)
        object.__setattr__(chroma_config, 'settings', custom_settings)
        
        # Determine file type from document name
        file_ext = stored_file_path.suffix.lower()
        
        #  Provide the stored file path to the search tool
        if file_ext == '.pdf':
            search_tool = PDFSearchTool(
                pdf=str(stored_file_path),  # Pass the stored file path
                config=chroma_config
            )
        elif file_ext in ['.docx', '.doc']:
            search_tool = DOCXSearchTool(
                docx=str(stored_file_path),  # Pass the stored file path
                config=chroma_config
            )
        else:
            print(f" Unsupported file type: {file_ext}")
            return None
        
        print(" Search tool reconnected with file and embeddings")
        
        # Create agent
        agent = create_agent(search_tool, llm, document_name)
        print(" Agent created")
        
        # Create crew
        crew = create_crew(agent)
        print(" Crew created")
        
        return {
            "llm": llm,
            "agent": agent,
            "crew": crew,
            "search_tool": search_tool,
            "embedding_dir": session_persist_dir,
            "image_handler": image_handler,
            "file_path": str(stored_file_path), 
            "document_name": document_name,
            "reloaded": True 
        }
        
    except Exception as e:
        print(f" Error reloading system from session: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# ============================================================================
# Cleanup  Data
# ============================================================================

def cleanup_orphaned_data(active_session_ids):
    """
    Delete ALL data (embeddings, images, and documents) for sessions that 
    don't exist in the database.
    
    This should be called at startup after loading all sessions from the database.
    
    Args:
        active_session_ids: List of session IDs that exist in the database
    
    Returns:
        dict: Statistics about cleanup
    """
    try:
        stats = {
            'scanned': 0,
            'embeddings_deleted': 0,
            'images_deleted': 0,
            'documents_deleted': 0,
            'errors': 0
        }
        
        # Track which session IDs we find
        all_session_ids = set()
        
        # 1. Clean up embeddings
        embeddings_dir = get_embeddings_dir()
        if embeddings_dir.exists():
            for session_dir in embeddings_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name
                all_session_ids.add(session_id)
                stats['scanned'] += 1
                
                # Check if this session ID exists in the database
                if session_id not in active_session_ids:
                    try:
                        shutil.rmtree(session_dir)
                        stats['embeddings_deleted'] += 1
                        print(f" Deleted orphaned embeddings: {session_id}")
                    except Exception as e:
                        stats['errors'] += 1
                        print(f" Could not delete embeddings {session_id}: {e}")
        
        # 2. Clean up images
        images_dir = get_images_dir()
        if images_dir.exists():
            for session_dir in images_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name
                all_session_ids.add(session_id)
                
                # Check if this session ID exists in the database
                if session_id not in active_session_ids:
                    try:
                        shutil.rmtree(session_dir)
                        stats['images_deleted'] += 1
                        print(f" Deleted orphaned images: {session_id}")
                    except Exception as e:
                        stats['errors'] += 1
                        print(f" Could not delete images {session_id}: {e}")
        
        # 3. Clean up documents (NEW)
        documents_dir = get_documents_dir()
        if documents_dir.exists():
            for session_dir in documents_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name
                all_session_ids.add(session_id)
                
                # Check if this session ID exists in the database
                if session_id not in active_session_ids:
                    try:
                        shutil.rmtree(session_dir)
                        stats['documents_deleted'] += 1
                        print(f" Deleted orphaned documents: {session_id}")
                    except Exception as e:
                        stats['errors'] += 1
                        print(f" Could not delete documents {session_id}: {e}")
        
        total_deleted = stats['embeddings_deleted'] + stats['images_deleted'] + stats['documents_deleted']
        
        if total_deleted > 0:
            print(f"\n Cleanup complete: {total_deleted} orphaned item(s) removed")
            print(f"  - Embeddings: {stats['embeddings_deleted']}")
            print(f"  - Images: {stats['images_deleted']}")
            print(f"  - Documents: {stats['documents_deleted']}")
        else:
            print(f" No orphaned data found")
        
        return stats
        
    except Exception as e:
        print(f" Error during orphaned data cleanup: {e}")
        return {
            'scanned': 0,
            'embeddings_deleted': 0,
            'images_deleted': 0,
            'documents_deleted': 0,
            'errors': 1
        }