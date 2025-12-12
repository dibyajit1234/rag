from pathlib import Path
from typing import List,Any
from langchain_community.document_loaders import PyMuPDFLoader,TextLoader,CSVLoader,Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir:str)->List[Any]:
    """ Load all supported files from the data directory and convert to langchain document structure"""
    #use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[Debug] datapath : {data_path}")
    documents =[]

    #pdf files 
    pdf_files = list(data_path.glob("**/*.pdf"))

    for pdf_file in pdf_files:
        print(f"Processing pdf : {pdf_file.name}\n")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            document = loader.load()
            print(f"[Debug] loaded {len(document)} documents from {len(pdf_file.name)}")
            documents.extend(document)
        except Exception as e:
            print(f"[Error] Problem occured loading pdf {e}")
    
    #text files
    text_files = list(data_path.glob("**/*.txt"))

    for text_file in text_files:
        print(f"Processing text : {text_file.name}\n")
        try:
            loader = TextLoader(str(text_file))
            document = loader.load()
            print(f"[Debug] loaded {len(document)} documents from {len(text_file.name)}")
            documents.extend(document)
        except Exception as e:
            print(f"[Error] Problem occured loading text {e}")

    #csv files
    csv_files = list(data_path.glob("**/*.csv"))

    for csv_file in csv_files:
        print(f"Processing csv : {csv_file.name}\n")
        try:
            loader = CSVLoader(str(csv_file))
            document = loader.load()
            print(f"[Debug] loaded {len(document)} documents from {len(csv_file.name)}")
            documents.extend(document)
        except Exception as e:
            print(f"[Error] Problem occured loading csv {e}")

    #SQL files
    sql_files = list(data_path.glob("**/*.sql"))

    for sql_file in sql_files:
        print(f"Processing sql : {sql_file.name}\n")
        try:
            loader = TextLoader(str(sql_file))
            document = loader.load()
            print(f"[Debug] loaded {len(document)} documents from {len(sql_file.name)}")
            documents.extend(document)
        except Exception as e:
            print(f"[Error] Problem occured loading sql {e}")

    return documents