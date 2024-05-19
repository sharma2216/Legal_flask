from flask import Flask, request, jsonify
import boto3
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.bedrock import Bedrock
from botocore.config import Config
import warnings
import logging
import sys
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
app = Flask(__name__) 
session = boto3.session.Session(profile_name='default')
retry_config = Config(
    region_name='us-east-1',
    retries={'max_attempts': 10, 'mode': 'standard'}
)
boto3_bedrock = session.client("bedrock", config=retry_config)
boto3_bedrock_runtime = session.client("bedrock-runtime", config=retry_config)

llm = Bedrock(
   model_id="mistral.mixtral-8x7b-instruct-v0:1",
    client=boto3_bedrock_runtime,
    model_kwargs={
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 200,
    }
)
loader = PyPDFDirectoryLoader("pdfs")
docs = loader.load()
embed_model = HuggingFaceBgeEmbeddings(model_name ="BAAI/bge-base-en")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)

db = Chroma.from_documents(texts, embedding=embed_model, persist_directory="dbase")

DEFAULT_SYSTEM_PROMPT = """      
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Give a long answer if he asks about summary or what happened.
""".strip()
SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know; don't try to make up an answer."

def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f""" 
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
""".strip()
   
template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_invoke= RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=True
)
@app.route('/')
def index():
    return open('templates/main.html').read()

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json['question']
    result = qa_invoke(question)
    
    # Print the result for debugging
    print("Result:", result)
    
    
    # Extract answer text from the result
    answer_text = result['result']
    
    # Return the answer text as JSON response
    return jsonify({'answer': answer_text})


if __name__ == '__main__':
    app.run(debug=True)
