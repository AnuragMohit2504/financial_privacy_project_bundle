# backend/ingestion.py
import os
import sys
import faiss
import pickle
import pdfplumber
from sentence_transformers import SentenceTransformer

# --- Fix path so "privacy" can be imported ---
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from privacy.masker import mask_text

# -------------------------
# Config
# -------------------------
DATA_DIR = os.path.join(proj_root, "backend", "data")   # PDF upload folder
INDEX_FILE = os.path.join(proj_root, "backend", "index.faiss")
STORE_FILE = os.path.join(proj_root, "backend", "vector_store.pkl")

# -------------------------
# Vector Store
# -------------------------
class FinanceVectorStore:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_documents(self, docs):
        """docs: list of strings (bank statement rows converted to text)"""
        masked_docs = [mask_text(d) for d in docs]
        embeddings = self.model.encode(masked_docs, convert_to_numpy=True)
        self.index.add(embeddings)
        self.texts.extend(masked_docs)
        self.save()

    def retrieve(self, query, top_k=3):
        masked_q = mask_text(query)
        q_vec = self.model.encode([masked_q], convert_to_numpy=True)
        D, I = self.index.search(q_vec, top_k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

    def save(self):
        faiss.write_index(self.index, INDEX_FILE)
        with open(STORE_FILE, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(STORE_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(STORE_FILE, "rb") as f:
                self.texts = pickle.load(f)

# -------------------------
# Utils: Load statements from PDFs
# -------------------------
def load_statements(data_dir=DATA_DIR, file: str = None):
    """
    Load transactions from PDF statements.
    If file is provided, only parse that file.
    """
    docs = []
    files = [file] if file else os.listdir(data_dir)

    for fname in files:
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(data_dir, fname)

        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            continue

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    table = page.extract_table()
                    if not table:
                        continue
                    headers = [h.strip().lower() for h in table[0] if h]
                    for row in table[1:]:
                        if not any(row):  # skip empty rows
                            continue
                        parts = []
                        if "date" in headers and row[headers.index("date")]:
                            parts.append(f"Date: {row[headers.index('date')]}")
                        if "narration" in headers and row[headers.index("narration")]:
                            parts.append(f"Narration: {row[headers.index('narration')]}")
                        if "withdrawal amt." in headers and row[headers.index("withdrawal amt.")]:
                            parts.append(f"Debit: {row[headers.index('withdrawal amt.')]}") 
                        if "deposit amt." in headers and row[headers.index("deposit amt.")]:
                            parts.append(f"Credit: {row[headers.index('deposit amt.')]}") 
                        if "closing balance" in headers and row[headers.index("closing balance")]:
                            parts.append(f"Balance: {row[headers.index('closing balance')]}") 
                        
                        if parts:
                            docs.append(" | ".join(parts))
                except Exception as e:
                    print(f"⚠️ Error parsing {fname}: {e}")
    return docs

# -------------------------
# Main pipeline (CLI usage)
# -------------------------
if __name__ == "__main__":
    vs = FinanceVectorStore()
    vs.load()  # load existing index if available

    print("📂 Loading statements from PDFs...")
    docs = load_statements(DATA_DIR)
    print(f"✅ Loaded {len(docs)} transactions from PDFs")

    if docs:
        vs.add_documents(docs)
        print("✅ Documents added to vector store")
    else:
        print("⚠️ No statements found in data/")
