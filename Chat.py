# Chat.py

# Streamlit: web arayüzü oluşturmak için kullanılır
import streamlit as st

# Google Generative AI: Gemini 2.0 Flash modelini kullanmak için
import google.generativeai as genai

# HuggingFace datasets: parquet dosyalarını kolayca yüklemek için
from datasets import load_dataset

# SentenceTransformers: metinleri vektörlere çevirip embedding oluşturmak için
from sentence_transformers import SentenceTransformer

# ChromaDB: embedding tabanlı veritabanı, hızlı retrieval için
import chromadb

# ---------------------------
# 1️⃣ API Key
# ---------------------------

# Streamlit secrets.toml'dan API Key çekiyoruz
API = st.secrets["API_KEY"]

# Google Generative AI kütüphanesini API Key ile yapılandırıyoruz
genai.configure(api_key=API)

# Gemini 2.0 Flash modelini başlatıyoruz
model = genai.GenerativeModel("gemini-2.0-flash")

# Yeni bir chat oturumu başlatıyoruz
chat = model.start_chat(history=[])

# ---------------------------
# 2️⃣ Dataset yükleme
# ---------------------------

# @st.cache_data: Fonksiyon sonucu cache'lenir. Tekrar tekrar yüklenmez, performans artar
@st.cache_data
def load_local_dataset():
    # Localdeki parquet dosyalarının isimleri
    files = {
        "atlas": "atlas-00000-of-00001.parquet",
        "baskentistanbul": "baskentistanbul-00000-of-00001.parquet",
        "bayindir": "bayindir-00000-of-00001.parquet",
        "medipol": "medipol-00000-of-00001.parquet",
        "yeditepe": "yeditepe-00000-of-00001.parquet"
    }
    
    # HuggingFace datasets ile parquet dosyalarını yükler
    dataset_raw = load_dataset("parquet", data_files=files)
    
    dataset = []  # Metinler buraya eklenecek
    titles = []   # Başlıklar buraya eklenecek
    
    # Dataset içindeki her split'i dolaş
    for split in dataset_raw:
        # Sadece ilk 3 metni alıyoruz (test amaçlı)
        dataset += dataset_raw[split][:3]["text"]
        # Başlıkları oluşturuyoruz
        titles += ["Makale " + str(i+1) for i in range(len(dataset_raw[split][:3]["text"]))]
    
    return dataset, titles  # İki listeyi döndürüyoruz

# Fonksiyonu çağır ve dataset ile titles değişkenlerini oluştur
dataset, titles = load_local_dataset()

# ---------------------------
# 3️⃣ Embedding ve ChromaDB
# ---------------------------

# @st.cache_resource: Fonksiyon sonucu cache'lenir ve uzun süreli kaynakları (model, veritabanı) saklar
@st.cache_resource
def init_chroma():
    # SentenceTransformer embedding modelini yükle
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # ChromaDB client oluştur
    client = chromadb.Client()
    
    # "medical_articles" koleksiyonunu oluştur veya varsa al
    collection = client.get_or_create_collection("medical_articles")
    
    # Koleksiyon boşsa embedding oluştur ve ekle
    if len(collection.get()["ids"]) == 0:
        embeddings = embedding_model.encode(dataset, batch_size=32, show_progress_bar=True)
        for i, (text, title, emb) in enumerate(zip(dataset, titles, embeddings)):
            collection.add(
                ids=[str(i)],              # Her makaleye benzersiz ID
                embeddings=[emb],          # Embedding vektörü
                documents=[text],          # Asıl metin
                metadatas=[{"title": title}]  # Başlık metadata olarak
            )
    return collection, embedding_model

# Koleksiyon ve embedding modelini başlat
collection, embedding_model = init_chroma()

# ---------------------------
# 4️⃣ Retrieval fonksiyonu
# ---------------------------

def retrieve_context(query, top_k=5):
    # Tüm metinler ve başlıklar
    all_docs = collection.get()["documents"]
    all_titles = [meta["title"].lower() for meta in collection.get()["metadatas"]]

    topic = query.lower()  # Kullanıcı sorgusunu küçük harfe çevir
    
    # Sorguya uygun metinleri filtrele (başlık veya içerik eşleşirse)
    filtered_docs = [doc for doc, title in zip(all_docs, all_titles) if topic in title or topic in doc.lower()]
    
    # Eğer eşleşen yoksa tüm dokümanları kullan
    if len(filtered_docs) == 0:
        filtered_docs = all_docs
    
    # Sadece top_k kadar döndür
    filtered_docs = filtered_docs[:top_k]
    
    # Çok uzun dokümanları kısalt (ilk 300 karakter)
    summarized_docs = [doc[:300] + "..." if len(doc) > 300 else doc for doc in filtered_docs]
    
    return summarized_docs

# ---------------------------
# 5️⃣ Streamlit arayüzü
# ---------------------------

# Başlık ekle
st.title("💬 DOA Medical Chat")

# Kullanıcıdan metin al
user_input = st.text_input("Sorunuzu yazın:", "")

# Eğer kullanıcı yazdıysa
if user_input:
    # Bağlamı retrieve et
    context_docs = retrieve_context(user_input)
    context_text = "\n\n".join(context_docs)  # Dokümanları birleştir

    # Model için prompt hazırla
    prompt = f"""
Aşağıdaki tıbbi makalelerden alınan bilgiler ışığında soruyu yanıtla.
Eğer yeterli bilgi yoksa "Bu konuda yeterli veri bulunamadı." de.

**Soru:** {user_input}

**Bağlam:**
{context_text}
"""
    # Gemini 2.0 Flash modeline prompt gönder
    response = chat.send_message(prompt)
    
    # Model cevabını Streamlit üzerinde göster
    st.markdown(f"**🤖 Gemini:** {response.text}")
