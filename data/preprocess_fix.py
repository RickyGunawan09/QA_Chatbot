import os,time
from glob import glob
import pandas as pd
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
import uuid
from natsort import natsorted
limit=5 #for processing csv split to embeding

embedding_model="BAAI/bge-small-en-v1.5"
underlying_embeddings=HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs = {'device': 'cpu'})

dir_split_csv="./splitcsv_smallercolumn"
if not os.path.exists(f"{dir_split_csv}"): 
    os.makedirs(f"{dir_split_csv}") 

for i,chunk in enumerate(pd.read_csv('SPOTIFY_REVIEWS.csv', chunksize=5000)):
    chunk.drop(chunk.columns[chunk.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    chunk.drop(chunk.columns[chunk.columns.str.contains('review_id', case=False)], axis=1, inplace=True)
    chunk.drop(chunk.columns[chunk.columns.str.contains('pseudo_author_id', case=False)], axis=1, inplace=True)
    chunk.drop(chunk.columns[chunk.columns.str.contains('author_name', case=False)], axis=1, inplace=True)
    chunk.drop(chunk.columns[chunk.columns.str.contains('author_app_version', case=False)], axis=1, inplace=True)
    chunk.to_csv(f'{dir_split_csv}/chunk{i}.csv', index=False)

list_csv_split=glob(f"{dir_split_csv}/*.csv")
sorted_list_csv_split=natsorted(list_csv_split)

total_start=time.time()
for i,v in enumerate(sorted_list_csv_split):
    start=time.time()
    print(i , v)

    start_split=time.time()
    loader = CSVLoader(f"{v}")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,separators=["\n"])
    splits = text_splitter.split_documents(docs)
    end_split=time.time()
    print("split time", end_split-start_split)

    start_emb=time.time()
    uuids = [str(uuid.uuid4()) for x in range(len(docs))]
    vectorstore=Chroma.from_documents(docs,ids=uuids,persist_directory="./chroma_persist2/",embedding=underlying_embeddings)
    vectorstore.persist()
    end_emb=time.time()
    print("embedding time", end_emb-start_emb)

    end=time.time()
    print(f"{v} waktu total {end-start}")

    if i == limit:
        break

total_end=time.time()
print(f"selesai preprocess {total_end-total_start}")