import os 
import faiss
import numpy as np
import pickle
from typing import List,Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self,persist_dict :str='faiss_store',embedding_model: str='all-MiniLM-L6-v2',chunk_size=500,chunk_overlap=100):
        self.persist_dict = persist_dict
        os.makedirs(self.persist_dict,exist_ok=True)
        self.embedding_model = embedding_model
        self.index = None
        self.metadata=[]
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[Info] Loaded embedding model :{embedding_model}")

    def build_from_documents(self,documents : List[Any]):
        print(f"Building vector store from {len(documents)} raw documents")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model,chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text":chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'),metadatas)
        self.save()
        print(f"[Info] vector store created and daved to {self.persist_dict}")

    def add_embeddings(self, embeddings :np.ndarray,metadatas : List[Any]=None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[Info] added {embeddings.shape} vectors to Faiss index")

    def save(self):
        faiss_path = os.path.join(self.persist_dict,"faiss.index")
        meta_path = os.path.join(self.persist_dict,'metadata.pkl')
        faiss.write_index(self.index,faiss_path)
        with open(meta_path ,'wb') as f:
            pickle.dump(self.metadata,f)
        print(f"[Info] Saved Faiss index and metadata to persist path {self.persist_dict}")

    def load(self):
        faiss_path = os.path.join(self.persist_dict,"faiss.index")
        meta_path = os.path.join(self.persist_dict,'metadata.pkl')
        self.index =faiss.read_index(faiss_path)
        with open(meta_path ,'rb') as f:
            self.metadata =pickle.load(f)
        print(f"[Info] Loaded Faiss index and metadata from persist path {self.persist_dict}")

    def search(self,query_embedding:np.ndarray,top_k:int=5):
        D,I = self.index.search(query_embedding,top_k)
        results =[]
        for idx,dist in zip(I[0],D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index":idx,"distance":dist,"metadata":meta})
        return results
    
    def query(self,query_text:str,top_k:int=5):
        print(f"[Info] quering vecgtor store for {query_text}")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb,top_k=top_k)