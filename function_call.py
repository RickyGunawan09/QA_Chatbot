from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI,HarmBlockThreshold,HarmCategory
from langchain_core.prompts import ChatPromptTemplate
import time
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.globals import set_debug
set_debug(True)

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             google_api_key="AIzaSyAuXdtAUuDqmKR5eXn6e5zFA6Sz9C0knxg", 
                             temperature=0.5, 
                             convert_system_message_to_human=True,
                             max_output_tokens=4096,
                             safety_settings={
                                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                }
                                )

start=time.time()
embedding_model="BAAI/bge-small-en-v1.5"
underlying_embeddings=HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs = {'device': 'cpu'})
vectorstore=Chroma(persist_directory="./data/chroma_persist2/",embedding_function=underlying_embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
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