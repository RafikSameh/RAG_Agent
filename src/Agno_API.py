# api/agno_api.py
import os, shutil, sqlite3, time, uuid, re
from pathlib import Path
import torch
from agno.agent import Agent, AgentSession
from agno.db.sqlite import SqliteDb
from agno.db.base import SessionType
from agno.models.huggingface import HuggingFace
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.guardrails import PIIDetectionGuardrail, PromptInjectionGuardrail
from agno.session import SessionSummaryManager
from pdf2image import convert_from_path
import fitz
import constants as c




class AgnoAPI:
    def __init__(self, pdf_path, db_path="agno.db"):
        self.agent = None
        self.PDF_PATH = pdf_path
        self.TABLE_NAME = os.path.basename(self.PDF_PATH)
        self.URI_PATH = c.URI_PATH
        self.AGENT_DATABASE = SqliteDb(db_file=db_path)
        
        os.makedirs(f"{c.PDF_PAGES_FOLDER}", exist_ok=True)
        os.makedirs(f"{c.PDF_IMAGE_FOLDER}", exist_ok=True)

         # --- Dynamic folders for this specific document ---
        pdf_stem = Path(self.PDF_PATH).stem
        self.pdf_pages_dir = Path(f"./{c.PDF_PAGES_FOLDER}/{pdf_stem}")
        self.pdf_images_dir = Path(f"./{c.PDF_IMAGE_FOLDER}/{pdf_stem}")


    # ---------- PDF RENDERING ----------
    def render_pdf_pages(self, pdf_path, out_dir):
        
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = list(out_dir.glob("page*.jpg"))
        if existing:
            print(f"✅ PDF pages already rendered in {out_dir}")
            return
        pages = convert_from_path(pdf_path, dpi=150)
        for i, page in enumerate(pages, start=1):
            page.save(out_dir / f"page{i}.jpg", "JPEG", quality=80)
        print(f"✅ Rendered {len(pages)} pages")

    def extract_pdf_images(self, pdf_path, save_dir, min_w=200, min_h=200):
        doc = fitz.open(pdf_path)
        os.makedirs(save_dir, exist_ok=True)
        images = []
        seen = set()
        print(f"🖼️ Extracting images from {pdf_path}...")
        for pno in range(len(doc)):
            page = doc[pno]
            for i, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                if xref in seen:
                    continue
                seen.add(xref)
                pix = fitz.Pixmap(doc, xref)
                if pix.width < min_w or pix.height < min_h:
                    continue
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                path = os.path.join(save_dir, f"page{pno+1}_img{i}.png")
                pix.save(path)
                images.append((path, pno+1))
                pix = None
        print(f"📦 Total extracted: {len(images)} images.")
    
    # ---------- BACKEND: GET IMAGE PATHS FOR REFERENCED PAGES ----------
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


    # ---------- MODEL & KNOWLEDGE ----------
    def load_model(self):
        # Prevent meta tensor creation
        torch.set_float32_matmul_precision('high')
        model =  HuggingFace(
            id="Qwen/Qwen3-4B-Instruct-2507",
            provider="hf-inference",
            max_tokens=4096,
            api_key=os.getenv("HF_TOKEN"),
        )
        return model


    def safe_lancedb(self, table_name, uri, embedder, pdf_changed) -> LanceDb:
        try:
            reload = 0
            
            if pdf_changed:
                reload = 1
                shutil.rmtree(uri, ignore_errors=True)
                time.sleep(0.5)
            
            # Test embedder first
            try:
                test_embedding = embedder.get_embedding("test")
                print(f"✅ Embedder initialized successfully. Dimension: {len(test_embedding)}")
            except Exception as e:
                print(f"⚠️ Embedder initialization failed: {e}")
                raise
            
            db = LanceDb(
                table_name=table_name,
                uri=uri,
                search_type=SearchType.vector,
                embedder=embedder,
                on_bad_vectors="drop",
            )
            
            return db, reload
            
        except Exception as e:
            if "meta tensor" in str(e).lower() or "dim" in str(e) or "FixedSizeListType" in str(e):
                print("⚠️ Detected embedding/schema mismatch — resetting database...")
                shutil.rmtree(uri, ignore_errors=True)
                time.sleep(0.5)
                
                db = LanceDb(
                    table_name=table_name,
                    uri=uri,
                    search_type=SearchType.vector,
                    embedder=embedder,
                    on_bad_vectors="drop",
                )
                return db, 1
            else:
                print(f"❌ Failed to initialize LanceDB: {e}")
                raise

    def initialize_agent(self,session_id,pdf_changed: bool = False):
        embedder = SentenceTransformerEmbedder()
        if pdf_changed:
            try:
                vector_db.table.delete(where="true")
                print("🧹 Cleared previous document embeddings for new file.", icon="🧠")
            except Exception:
                print("⚠️ Could not clear old vectors; table may be empty already.")
        
        vector_db,reload = self.safe_lancedb(self.TABLE_NAME, self.URI_PATH, embedder,pdf_changed)
        time.sleep(0.5)
        pdf_pages_dir = Path(f"./{c.PDF_PAGES_FOLDER}/{Path(self.PDF_PATH).stem}")
        pdf_images_dir = Path(f"./{c.PDF_IMAGE_FOLDER}/{Path(self.PDF_PATH).stem}")
        shutil.rmtree(pdf_pages_dir, ignore_errors=True)
        shutil.rmtree(pdf_images_dir, ignore_errors=True)
        self.render_pdf_pages(self.PDF_PATH, pdf_pages_dir)
        self.extract_pdf_images(self.PDF_PATH, pdf_images_dir)

        pdf_reader = PDFReader(name="Page Chunking Reader", chunk_by="page")
        knowledge_base = Knowledge(vector_db=vector_db)
        knowledge_base.add_content(path=self.PDF_PATH, skip_if_exists=True, reader=pdf_reader)

        prompt_injection_guardrail = PromptInjectionGuardrail()
        pii_detection_guardrail = PIIDetectionGuardrail(mask_pii=True)
        model = self.load_model()

        # 🤖 Create agent
        self.agent = Agent(
            name="UPM Document RAG Agent",
            session_id=session_id,
            model= model,
            knowledge=knowledge_base,
            add_knowledge_to_context=True,
            add_history_to_context=True,
            db=self.AGENT_DATABASE,
            search_knowledge=True,
            markdown=True,
            description="An agent that can answer questions about UPM from the PDF document.",
            instructions=(
                ["You are a helpful assistant that answers questions from the provided document excerpts. ",
                "Always include the page numbers where you found the information at the end of your answer in the exact format: 'Pages: 1, 5-8' or 'Pages: 1'. ",
                "If the answer is not in the document, you must state 'Not provided in the document' and provide no page numbers."]
            ),
            add_memories_to_context=True,
            enable_agentic_memory=True,
            enable_session_summaries=True,
            read_chat_history=True,
            pre_hooks=[prompt_injection_guardrail, pii_detection_guardrail],
            num_history_runs=2,
            num_history_sessions=2,
            store_media=True,
            update_knowledge=True
        )

        return self.agent,reload

    # ---------- SESSION HANDLERS ----------
    def create_new_session(self):
        sid = str(uuid.uuid4())
        new_session = AgentSession(
            session_id=sid, 
            agent_id="rag-agent", 
            created_at=round(time.time()), 
            user_id="user@example.com",
            runs=[],
            summary="",)
        self.AGENT_DATABASE.upsert_session(session=new_session)
        return sid

    def get_session_data(self, session_id):
        conn = sqlite3.connect(self.AGENT_DATABASE.db_file)
        cur = conn.cursor()
        cur.execute("SELECT * FROM agno_sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        conn.close()
        return row

    def get_session_messages(self, session_id):
        msgs = self.AGENT_DATABASE.get_session(session_id=session_id,session_type="agent").get_messages_for_session()
        db_messages = []
        for msg in msgs:
            if msg.role == 'user':
                content = msg.content.split('\n', 1)[0].strip()
            elif msg.role == 'assistant':
                content = msg.content
            db_messages.append({"role": msg.role, "content": content})
        return db_messages
    
    def Sessions_manager(self,):
        self.session_records = self.AGENT_DATABASE.get_sessions(session_type=SessionType.AGENT,deserialize=True)
        self.existing_sessions = [session.session_id for session in self.session_records]
        # Map session IDs to labels for display
        self.session_labels = [
            f"{session.get_session_summary().summary.split(',')[0][:60] if session.get_session_summary() and session.get_session_summary().summary else 'No summary available'}"
            for session in self.session_records
        ]
        self.label_to_session_id = {
            label: session.session_id
            for label, session in zip(self.session_labels, self.session_records)
        }
        return self.existing_sessions,self.session_labels,self.label_to_session_id
    
    def update_session(self,result,session_id):
        current_session = self.AGENT_DATABASE.get_session(session_id=session_id,session_type="agent",deserialize=True)
        current_session.upsert_run(run=result)
        current_session.summary = SessionSummaryManager(self.agent.model).create_session_summary(current_session)
        self.agent._upsert_session(current_session)
        self.label_to_session_id.update()
