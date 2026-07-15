from google import genai
import numpy as np
from pypdf import PdfReader
import os
from dotenv import load_dotenv

''' dotenv initializer '''
load_dotenv()

''' Loading Environmental Variables '''
api_key = os.getenv('API_KEY')
gen_model = os.getenv('GENERATIVE_MODEL')
embbed_model = os.getenv('EMBEDDING_MODEL')

''' Google Genai Setup '''
client = genai.Client(api_key=api_key)

''' Codes to extract Files '''
def extract_raw_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text_elements = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_elements.append(page_text)

    return "\n".join(text_elements)

''' Chunking to load into vector db '''
def chunk_raw_text_data(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size-overlap):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break

    return chunks


''' Generating embeddings of chunks '''
def get_embedding_of_text(text:str):
    embedding = client.models.embed_content(
        model=embbed_model,
        contents=text
    )

    return embedding.embeddings[0].values

''' Cosine Similiarity to extract Common elements '''
def get_cosin_similiarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

''' Function to store elements in vector db '''
def store_data_into_vector_db(chunks: list[str]):
    store = []
    for chunk in chunks:
        if len(chunk.strip()) < 10:
            continue

        vector = get_embedding_of_text(chunk)
        store.append({'text' : chunk, 'vector' : vector})

    return store


''' Implementing core RAG engine '''
def core_rag_engine(user_query:str, vector_store:list[dict], top_k:int = 5):
    query_vector = get_embedding_of_text(user_query)

    scored_chunks = []
    for i in vector_store:
        score = get_cosin_similiarity(query_vector, i['vector'])
        scored_chunks.append((score, i['text']))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    context = "\n\n---\n\n".join([text for _,  text in scored_chunks[:top_k]])

    prompt = f"""
        you are an internal system expert. Answer the questions using only the 
        verified facts provided in the context blocks below.
        if the context lacks information, then reply "information not available."

        context : {context}

        Question : {user_query}

        Answer :
    """

    response = client.models.generate_content(
        model=gen_model,
        contents=prompt
    )

    return response.text



''' Execute Functions '''
path = os.path.join('Docs/nwaf050.pdf')
obj = extract_raw_text_from_pdf(path)
obj2 = chunk_raw_text_data(obj)
obj3 = store_data_into_vector_db(obj2)
# print(core_rag_engine('Give me information about this Document ', obj3))


while True:
    prompt = input("User : ")

    if prompt.lower() in ['exit', 'leave']:
        break

    try:
        response = core_rag_engine(prompt, obj3)
        print("System Generated - ", response)
    except Exception as e:
        print("Error - ", e )