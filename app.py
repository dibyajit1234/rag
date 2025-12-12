# from src.data_loader import load_all_documents
# from src.vector_store import FaissVectorStore
# from src.search import RAGSearch
# if __name__ =="__main__":
#     data = load_all_documents("data")
#     store = FaissVectorStore()
#     store.build_from_documents(data)
#     search = RAGSearch()
#     print("Enter exit to quit")
#     instruction = input("Enter your prompt here:\n")
#     while(instruction!='exit'):
#         print(search.search_and_summarize(instruction))
#         instruction = input("Enter your prompt here:\n")

# app.py
import streamlit as st
import os
import shutil
from pathlib import Path
from typing import List

# import your project modules (make sure src is in PYTHONPATH or use relative imports)
from src.data_loader import load_all_documents
from src.vector_store import FaissVectorStore
from src.search import RAGSearch

DATA_DIR = Path("data")

st.set_page_config(page_title="RAG Search (Streamlit)", layout="wide")

def clear_data_folder():
    """Delete everything inside data/ and recreate empty folder."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    """Save one uploaded file to data/ and return path."""
    dest = DATA_DIR / uploaded_file.name
    # avoid path traversal
    dest = dest.resolve()
    DATA_DIR_RESOLVED = DATA_DIR.resolve()
    if not str(dest).startswith(str(DATA_DIR_RESOLVED)):
        raise ValueError("Invalid file name")
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest

def build_faiss_from_data() -> FaissVectorStore:
    """Load all documents from data/ and build Faiss vector store."""
    docs = load_all_documents(str(DATA_DIR))
    store = FaissVectorStore()
    # If your FaissVectorStore has a different method name, adapt here.
    # In user's snippet they used: store.build_from_documents(data)
    try:
        store.build_from_documents(docs)
    except Exception:
        # fallback to a common alternative name
        if hasattr(store, "build"):
            store.build(docs)
        else:
            raise
    return store

def create_rag_search(store: FaissVectorStore):
    """Try to create RAGSearch instance by trying common constructor signatures."""
    # many projects differ in how RAGSearch accepts a vector store; try a few options
    try:
        return RAGSearch(vector_store=store)
    except Exception:
        pass
    try:
        return RAGSearch(store)
    except Exception:
        pass
    try:
        return RAGSearch()
    except Exception as e:
        raise RuntimeError(
            "Couldn't instantiate RAGSearch with any known signature. "
            "Check RAGSearch constructor; caught error: " + str(e)
        )


# --- On every app start/rerun we clear data/ so "reload/close" will remove previous uploads ---
# This satisfies: uploaded files removed when page reloads or app restarts.
if "initialized" not in st.session_state:
    clear_data_folder()
    st.session_state.initialized = True
    st.session_state.store_built = False
    st.session_state.last_uploaded_files = []

st.title("RAG Search — Upload context files and ask questions")
st.markdown(
    """
Upload files (PDF, TXT, CSV etc.). Files are saved to `data/` and used to build a FAISS vector store.
Press **End session** to remove uploaded files and the built index.
"""
)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload context documents (multiple allowed)",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv"]
    )

    if uploaded:
        # whenever new files uploaded, clear folder before saving so stale files removed
        clear_data_folder()
        saved_paths = []
        with st.spinner("Saving uploaded files..."):
            for f in uploaded:
                p = save_uploaded_file(f)
                saved_paths.append(str(p.name))
        st.session_state.last_uploaded_files = saved_paths
        st.success(f"Saved {len(saved_paths)} files to `data/`.")
        st.session_state.store_built = False  # need to (re)build store for new files

    # Build / rebuild index button
    if st.button("Upload documents"):
        if not any(DATA_DIR.iterdir()):
            st.warning("No files found in data/. Upload files first.")
        else:
            with st.spinner("Loading documents and building FAISS vector store..."):
                try:
                    store = build_faiss_from_data()
                    st.session_state.faiss_store_exists = True
                    st.session_state.store_built = True
                    st.session_state.store_info = "built"
                    # keep store object in session_state if it's picklable
                    st.session_state.faiss_store = store
                    st.success("FAISS vector store built successfully.")
                except Exception as e:
                    st.error(f"Failed to build FAISS vector store: {e}")

    # Quick status & list files
    st.markdown("**Current uploaded files**")
    if st.session_state.get("last_uploaded_files"):
        for name in st.session_state.last_uploaded_files:
            st.write("- " + name)
    else:
        st.write("_No uploaded files in data/_")

    # End session button (clear uploaded files and any index)
    if st.button("Refresh (clear uploaded files & index)", key="end_session"):
        with st.spinner("Clearing data folder and removing any index artifacts..."):
            clear_data_folder()
            # clear session state store references
            for k in ["faiss_store", "store_built", "faiss_store_exists", "store_info", "last_uploaded_files"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.success("Session ended — data/ cleared.")

with col2:
    st.markdown("### Ask a question")
    prompt = st.text_area("Enter your prompt:", height=150)
    run_query = st.button("Ask AI")

    # If user requests query but store not built yet, try auto-building
    if run_query:
        # ensure there is data and store built
        if not any(DATA_DIR.iterdir()):
            st.error("No context documents available. Upload documents first.")
        else:
            # build store if not already built in session state
            store = st.session_state.get("faiss_store", None)
            if store is None:
                with st.spinner("Building FAISS store (this may take a moment)..."):
                    try:
                        store = build_faiss_from_data()
                        st.session_state.faiss_store = store
                        st.session_state.store_built = True
                        st.success("FAISS store built.")
                    except Exception as e:
                        st.error(f"Failed to build FAISS store: {e}")
                        store = None

            if store is not None:
                try:
                    with st.spinner("Creating RAG searcher..."):
                        rag = create_rag_search(store)
                except Exception as e:
                    st.error(str(e))
                    rag = None

                if rag is not None:
                    if not prompt:
                        st.warning("Please type a prompt before searching.")
                    else:
                        with st.spinner("Searching and generating response..."):
                            try:
                                # Many RAGSearch implementations expose a method like search_and_summarize
                                if hasattr(rag, "search_and_summarize"):
                                    result = rag.search_and_summarize(prompt)
                                elif hasattr(rag, "run"):
                                    result = rag.run(prompt)
                                elif hasattr(rag, "query"):
                                    result = rag.query(prompt)
                                else:
                                    # fallback to using call if it's callable
                                    result = rag(prompt)
                                st.subheader("Result")
                                st.write(result)
                            except Exception as e:
                                st.error("Error while running RAG search: " + str(e))

# show lightweight footer info
st.markdown("---")
st.caption("Files are saved to `data/` on the server. This app will clear `data/` on reload/start and when you press End session.")
