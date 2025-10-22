# Chat.py
import streamlit as st
import google.generativeai as genai
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------
# 1️⃣ API Key
# ---------------------------
API = st.secrets["API_KEY"]
genai.configure(api_key=API)

# Gemini 2.0 Flash modeli
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat(history=[])

# ---------------------------
# 2️⃣ Dataset yükleme
# ---------------------------
@st.cache_data
def load_local_dataset():
    files = {
        "atlas": "atlas-00000-of-00001.parquet",
        "baskentistanbul": "baskentistanbul-00000-of-00001.parquet",
        "bayindir": "bayindir-00000-of-00001.parquet",
        "medipol": "medipol-00000-of-00001.parquet",
        "yeditepe": "yeditepe-00000-of-00001.parquet"
    }
    dataset_raw = load_dataset("parquet", data_files=files)
    
    dataset = []
    titles = []
    for split in dataset_raw:
        dataset += dataset_raw[split][:3]["text"]
        titles += ["Makale " + str(i+1) for i in range(len(dataset_raw[split][:3]["text"]))]
    
    return dataset, titles

dataset, titles = load_local_dataset()

# ---------------------------
# 3️⃣ Embedding ve ChromaDB
# ---------------------------
@st.cache_resource
def init_chroma():
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.Client()
    collection = client.get_or_create_collection("medical_articles")
    
    if len(collection.get()["ids"]) == 0:
        embeddings = embedding_model.encode(dataset, batch_size=32, show_progress_bar=True)
        for i, (text, title, emb) in enumerate(zip(dataset, titles, embeddings)):
            collection.add(
                ids=[str(i)],
                embeddings=[emb],
                documents=[text],
                metadatas=[{"title": title}]
            )
    return collection, embedding_model

collection, embedding_model = init_chroma()

# ---------------------------
# 4️⃣ Retrieval fonksiyonu
# ---------------------------
def retrieve_context(query, top_k=5):
    all_docs = collection.get()["documents"]
    all_titles = [meta["title"].lower() for meta in collection.get()["metadatas"]]

    topic = query.lower()
    filtered_docs = [doc for doc, title in zip(all_docs, all_titles) if topic in title or topic in doc.lower()]
    if len(filtered_docs) == 0:
        filtered_docs = all_docs
    filtered_docs = filtered_docs[:top_k]
    summarized_docs = [doc[:300] + "..." if len(doc) > 300 else doc for doc in filtered_docs]
    return summarized_docs

# ---------------------------
# 5️⃣ Streamlit arayüzü
# ---------------------------
st.title("💬 DOA Medical Chat")

user_input = st.text_input("Sorunuzu yazın:", "")

if user_input:
    context_docs = retrieve_context(user_input)
    context_text = "\n\n".join(context_docs)

    prompt = f"""
Aşağıdaki tıbbi makalelerden alınan bilgiler ışığında soruyu yanıtla.
Eğer yeterli bilgi yoksa "Bu konuda yeterli veri bulunamadı." de.

**Soru:** {user_input}

**Bağlam:**
{context_text}
"""
    response = chat.send_message(prompt)
    st.markdown(f"**🤖 Gemini:** {response.text}")
