import os,time
from glob import glob
import pandas as pd
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
import uuid
from natsort import natsorted
from dotenv import load_dotenv
if os.path.isfile('.env'):
    load_dotenv()
    
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" if os.getenv("EMBEDDING_MODEL") is None else os.getenv("EMBEDDING_MODEL")
LIMIT = 5 if os.getenv("LIMIT") is None else int(os.getenv("LIMIT"))  #for processing csv split to embeding
PATH_DATA_CSV = "./data/SPOTIFY_REVIEWS.csv" if os.getenv("PATH_DATA_CSV") is None else os.getenv("PATH_DATA_CSV")
CSV_SPLIT_LOC = "./data/splitcsv_smallercolumn" if os.getenv("CSV_SPLIT_LOC") is None else os.getenv("CSV_SPLIT_LOC")
PERSIST_DIR_CHROMA = "./chroma_persist" if os.getenv("PERSIST_DIR_CHROMA") is None else os.getenv("PERSIST_DIR_CHROMA")

underlying_embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs = {'device': 'cpu'})

if not os.path.exists(f"{CSV_SPLIT_LOC}"): 
    os.makedirs(f"{CSV_SPLIT_LOC}") 

for i,chunk in enumerate(pd.read_csv(F'{PATH_DATA_CSV}', chunksize=5000)):
    chunk.drop(chunk.columns[chunk.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    chunk.drop(chunk.columns[chunk.columns.str.contains('review_id', case=False)], axis=1, inplace=True)
    chunk.drop(chunk.columns[chunk.columns.str.contains('pseudo_author_id', case=False)], axis=1, inplace=True)
    chunk.drop(chunk.columns[chunk.columns.str.contains('author_name', case=False)], axis=1, inplace=True)
    chunk.drop(chunk.columns[chunk.columns.str.contains('author_app_version', case=False)], axis=1, inplace=True)
    chunk.to_csv(f'{CSV_SPLIT_LOC}/chunk{i}.csv', index=False)

list_csv_split=glob(f"{CSV_SPLIT_LOC}/*.csv")
sorted_list_csv_split=natsorted(list_csv_split)

total_start=time.time()
for i,v in enumerate(sorted_list_csv_split):
    start=time.time()
    print(i , v)

    start_split=time.time()
    loader = CSVLoader(f"{v}",encoding="utf8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,separators=["\n"])
    splits = text_splitter.split_documents(docs)
    end_split=time.time()
    print("split time", end_split-start_split)

    start_emb=time.time()
    uuids = [str(uuid.uuid4()) for x in range(len(docs))]
    vectorstore=Chroma.from_documents(docs,ids=uuids,persist_directory=f"{PERSIST_DIR_CHROMA}",embedding=underlying_embeddings)
    vectorstore.persist()
    end_emb=time.time()
    print("embedding time", end_emb-start_emb)

    end=time.time()
    print(f"{v} waktu total {end-start}")

    if i == LIMIT:
        break

total_end=time.time()
print(f"selesai preprocess {total_end-total_start}")