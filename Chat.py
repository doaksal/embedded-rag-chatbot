# chat_streamlit.py
# Akbank Generative AI Bootcamp – Embedded RAG + Gemini 2.0 Flash
# Geliştiren: Doga

import os
import streamlit as st
import google.generativeai as genai
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------
# 1️⃣ API Key ve model kurulumu
# ---------------------------
# Streamlit Secrets kullanımı
API = os.environ.get("API_KEY")
if not API:
    st.error("API_KEY bulunamadı! Streamlit Secrets kısmını kontrol et.")
    st.stop()

# Gemini 2.0 Flash yapılandırması
genai.configure(api_key=API)
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat(history=[])

# ---------------------------
# 2️⃣ Lokal dataset yükleme
# ---------------------------
st.info("📥 Dataset yükleniyor (lokal dosyalardan)...")
files = {
    "atlas": "data/atlas-00000-of-00001.parquet",
    "baskentistanbul": "data/baskentistanbul-00000-of-00001.parquet",
    "bayindir": "data/bayindir-00000-of-00001.parquet",
    "medipol": "data/medipol-00000-of-00001.parquet",
    "yeditepe": "data/yeditepe-00000-of-00001.parquet"
}

dataset_raw = load_dataset("parquet", data_files=files)
dataset = []
titles = []
for split in dataset_raw:
    dataset += dataset_raw[split][:3]["text"]
    titles += ["Makale " + str(i+1) for i in range(len(dataset_raw[split][:3]["text"]))]

st.success(f"✅ Dataset yüklendi. Toplam {len(dataset)} makale var.")

# ---------------------------
# 3️⃣ Embedding ve ChromaDB
# ---------------------------
st.info("📦 Embedding modeli yükleniyor...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

st.info("💾 ChromaDB kuruluyor ve koleksiyon oluşturuluyor...")
client = chromadb.Client()
collection = client.get_or_create_collection("medical_articles")

if len(collection.get()["ids"]) == 0:
    st.info("🔗 Makaleler embedding’e çevriliyor ve veritabanına ekleniyor...")
    embeddings = embedding_model.encode(dataset, batch_size=32, show_progress_bar=True)
    for i, (text, title, emb) in enumerate(zip(dataset, titles, embeddings)):
        collection.add(
            ids=[str(i)],
            embeddings=[emb],
            documents=[text],
            metadatas=[{"title": title}]
        )
    st.success("✅ Tüm makaleler veritabanına eklendi.")
else:
    st.success("✅ Koleksiyon zaten dolu, embedding’ler tekrar hesaplanmadı.")

# ---------------------------
# 4️⃣ Retrieval fonksiyonu
# ---------------------------
def retrieve_context(query, top_k=42):
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
# 5️⃣ Streamlit UI
# ---------------------------
st.title("💬 DOA Medical Chat")
st.write("Lütfen sorunu gir, tıbbi makaleler ışığında yanıt al. Çıkmak için tarayıcıyı kapatabilirsin.")

user_input = st.text_input("👤 Sen:")

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
    with st.spinner("🤖 Gemini cevap üretiyor..."):
        response = chat.send_message(prompt)
        st.markdown(f"**Gemini:** {response.text}")
