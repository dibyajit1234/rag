#simple rag pipeline with groq llm
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from src.vector_store import FaissVectorStore
load_dotenv()


class RAGSearch():
    def __init__(self,persist_dir:str='faiss_store',embedding_model:str="all-MiniLM-L6-v2",llm_model:str="openai/gpt-oss-120b"):
        self.vectorstore = FaissVectorStore(persist_dir,embedding_model)
        #Load and build vector store
        faiss_path = os.path.join(persist_dir,'faiss.index')
        meta_path = os.path.join(persist_dir,'metadata.pkl')

        if not(os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            data = load_all_documents("data")
            self.vectorstore.build_from_documents(data)
        else:
            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API")
        self.llm = ChatGroq(api_key = groq_api_key,model=llm_model,temperature=0.1,max_tokens=1024)
        print(f"[Info] LLM initialized :{llm_model}")

    def search_and_summarize(self,query:str,top_k:int=5)->str:
        results = self.vectorstore.query(query,top_k)
        texts = [r['metadata'].get('text',"") for r in results if r['metadata']]
        context = "\n\n".join(texts)
        if not context:
            print("No relavent context found\n")
        self.instrucion = f"""
            Use the following context to answer the questions concisely\n\n
            context:
            {context}
            question={query}
            answer: 
            """
        response = self.llm.invoke([self.instrucion])
        return response.content;
