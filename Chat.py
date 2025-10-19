# chat.py
# Akbank Generative AI Bootcamp – Embedded RAG + Gemini 2.0 Flash
# Geliştiren: Doga
# Açıklama:
# Bu proje, lokal tıbbi makaleleri kullanarak embedding tabanlı retrieval (RAG) yapar
# ve Gemini 2.0 Flash modelini kullanarak akıllı yanıtlar üretir.
# Veri kaynağı lokal olduğu için internet bağlantısı gerekmez.
# ChromaDB ile vektör veritabanı kurulmuş ve embedding’ler saklanmıştır.

import os
from dotenv import load_dotenv
import google.generativeai as genai
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------
# 1️⃣ API Key ve model kurulumu
# ---------------------------
# .env dosyasındaki API_KEY kullanılır. Yoksa hata verir.
load_dotenv()
API = os.getenv("API_KEY")
if not API:
    raise ValueError("API_KEY bulunamadı! .env dosyanı ve konumunu kontrol et.")

# Gemini 2.0 Flash yapılandırması
genai.configure(api_key=API)
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat(history=[])

# ---------------------------
# 2️⃣ Lokal dataset yükleme
# ---------------------------
# Data klasöründeki tüm parquet dosyaları yüklenir
print("📥 Dataset yükleniyor (lokal dosyalardan)...")
files = {
    "acibadem": "data/acibadem-00000-of-00001.parquet",
    "anadolusaglik": "data/anadolusaglik-00000-of-00001.parquet",
    "atlas": "data/atlas-00000-of-00001.parquet",
    "baskentistanbul": "data/baskentistanbul-00000-of-00001.parquet",
    "bayindir": "data/bayindir-00000-of-00001.parquet",
    "florence": "data/florence-00000-of-00001.parquet",
    "guven": "data/guven-00000-of-00001.parquet",
    "liv": "data/liv-00000-of-00001.parquet",
    "medicalpark": "data/medicalpark-00000-of-00001.parquet",
    "medicalpoint": "data/medicalpoint-00000-of-00001.parquet",
    "medicana": "data/medicana-00000-of-00001.parquet",
    "medipol": "data/medipol-00000-of-00001.parquet",
    "memorial": "data/memorial-00000-of-00001.parquet",
    "yeditepe": "data/yeditepe-00000-of-00001.parquet"
}

# Datasets kütüphanesi ile parquet dosyaları okunur
dataset_raw = load_dataset("parquet", data_files=files)

# Sadece text alanları ve başlıkları ayıklıyoruz (test için ilk 3 kayıt alınıyor)
dataset = []
titles = []
for split in dataset_raw:
    dataset += dataset_raw[split][:3]["text"]
    titles += ["Makale " + str(i+1) for i in range(len(dataset_raw[split][:3]["text"]))]

print(f"✅ Dataset yüklendi. Toplam {len(dataset)} makale var.")

# ---------------------------
# 3️⃣ Embedding ve ChromaDB
# ---------------------------
# SentenceTransformer ile embedding oluşturulur
print("📦 Embedding modeli yükleniyor...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ChromaDB client oluşturulur ve koleksiyon kurulur
print("💾 ChromaDB kuruluyor ve koleksiyon oluşturuluyor...")
client = chromadb.Client()
collection = client.get_or_create_collection("medical_articles")

# Eğer koleksiyon boşsa embedding’leri hesapla ve ekle
if len(collection.get()["ids"]) == 0:
    print("🔗 Makaleler embedding’e çevriliyor ve veritabanına ekleniyor...")
    embeddings = embedding_model.encode(dataset, batch_size=32, show_progress_bar=True)

    for i, (text, title, emb) in enumerate(zip(dataset, titles, embeddings)):
        collection.add(
            ids=[str(i)],
            embeddings=[emb],
            documents=[text],
            metadatas=[{"title": title}]
        )
    print("✅ Tüm makaleler veritabanına eklendi.")
else:
    print("✅ Koleksiyon zaten dolu, embedding’ler tekrar hesaplanmadı.")

# ---------------------------
# 4️⃣ Gelişmiş retrieval fonksiyonu
# ---------------------------
def retrieve_context(query, top_k=42):
    """
    - Embedding tabanlı en yakın makaleleri getirir.
    - Eğer query belirli bir konu içeriyorsa (ör. 'grip'), title bazlı filtre uygular.
    - Döndürülen metinler özetlenir (ilk 300 karakter) prompt’u şişirmemek için.
    """
    all_docs = collection.get()["documents"]
    all_titles = [meta["title"].lower() for meta in collection.get()["metadatas"]]

    topic = query.lower()
    filtered_docs = [doc for doc, title in zip(all_docs, all_titles) if topic in title or topic in doc.lower()]
    
    if len(filtered_docs) == 0:
        filtered_docs = all_docs  # fallback: tüm dokümanlar

    filtered_docs = filtered_docs[:top_k]

    summarized_docs = [doc[:300] + "..." if len(doc) > 300 else doc for doc in filtered_docs]
    
    return summarized_docs

# ---------------------------
# 5️⃣ Chat loop
# ---------------------------
print("💬 DOA Medical Chat'e hoş geldin! (çıkmak için 'q' yaz)\n")

while True:
    user_input = input("👤 Sen: ")
    if user_input.lower() in ["q", "quit", "exit"]:
        print("🔚 Sohbet sonlandırıldı.")
        break

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
    print("🤖 Gemini:", response.text, "\n")
