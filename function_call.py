from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI,HarmBlockThreshold,HarmCategory
from langchain_core.prompts import ChatPromptTemplate
import os,time
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.globals import set_debug
from dotenv import load_dotenv
if os.path.isfile('.env'):
    load_dotenv()

DEBUG = (os.getenv("DEBUG").lower() == 'true') if os.getenv("DEBUG") is not None else False
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" if os.getenv("EMBEDDING_MODEL") is None else os.getenv("EMBEDDING_MODEL")
PERSIST_DIR_CHROMA = "./chroma_persist" if os.getenv("PERSIST_DIR_CHROMA") is None else os.getenv("PERSIST_DIR_CHROMA")
MODEL_GOOGLE = "gemini-pro" if os.getenv("MODEL_GOOGLE") is None else os.getenv("MODEL_GOOGLE")
GOOGLE_API_KEY = "AIzaSyBclCA_SkFy05ghSh_zwEcTpvZ6UJETHZ8" if os.getenv("GOOGLE_API_KEY") is None else os.getenv("GOOGLE_API_KEY")
TEMPERATURE = 0.5 if os.getenv("TEMPERATURE") is None else float(os.getenv("TEMPERATURE"))
TOKENS = 4096 if os.getenv("TOKENS") is None else int(os.getenv("TOKENS"))

SEARCH_TYPE = "similarity_score_threshold" if os.getenv("SEARCH_TYPE") is None else os.getenv("SEARCH_TYPE")
K_RETRIEVER = 5 if os.getenv("K_RETRIEVER") is None else int(os.getenv("K_RETRIEVER"))
SCORE_THRESH = 0.7 if os.getenv("SCORE_THRESH") is None else float(os.getenv("SCORE_THRESH"))


set_debug(DEBUG)
llm = ChatGoogleGenerativeAI(model=MODEL_GOOGLE,
                            google_api_key=GOOGLE_API_KEY, # api key insert here, 
                            temperature=TEMPERATURE, 
                            convert_system_message_to_human=True,
                            max_output_tokens=TOKENS,
                            safety_settings={
                                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                }
                                )

start=time.time()
embedding_model=EMBEDDING_MODEL
underlying_embeddings=HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs = {'device': 'cpu'})
vectorstore=Chroma(persist_directory=PERSIST_DIR_CHROMA,embedding_function=underlying_embeddings)
if SEARCH_TYPE=="similarity_score_threshold":
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold":SCORE_THRESH,"k": K_RETRIEVER})
elif SEARCH_TYPE=="similarity":
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVER})
else:
    print("search type not found, right now search type only support 'similarity' and 'similarity_score_threshold'")
    print("running default search type 'similarity'")
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVER})

end=time.time()
print("waktu",end-start)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def function_call(question):
    prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an AI assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. the context is related to google review of spotify platform music. please be honest and critical for answering the qeustion."),
            ("user", "{context} , {question}")
        ])

    output_parser = StrOutputParser()

    chain = {"context": retriever | format_docs , "question": RunnablePassthrough()} | prompt | llm | output_parser
    
    result=chain.invoke(question)

    return result
