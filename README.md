# RAG Tabanlı Tıbbi Chatbot

Bu proje, tıbbi makaleleri kullanarak sorulara cevap veren bir RAG (Retrieval-Augmented Generation) tabanlı chatbot uygulamasıdır.

CHAT_BOT/
│
├─ data/                       
│   ├─ atlas-00000-of-00001.parquet
│   ├─ baskentistanbul-00000-of-00001.parquet
│   ├─ bayindir-00000-of-00001.parquet
│   ├─ medipol-00000-of-00001.parquet
│   └─ yeditepe-00000-of-00001.parquet
│
├─ .env                  
├─ chat.py                 
├─ requirement.txt 
├─ atlas-00000-of-00001.parquet
├─ baskentistanbul-00000-of-00001.parquet
├─ bayindir-00000-of-00001.parquet
├─ medipol-00000-of-00001.parquet
├─ yeditepe-00000-of-00001.parquet
├─ LICENSE
└─ README.md                 

# DOA Medical Chat - Embedded RAG + Gemini 2.0 Flash

**Geliştirici:** Doga Koksal 
**Bootcamp:** Akbank Generative AI Bootcamp  

## Proje Açıklaması
Bu proje, lokal tıbbi makaleleri kullanarak embedding tabanlı retrieval yapar ve Google Gemini 2.0 Flash modeli ile akıllı yanıtlar üretir.  
Sistem tamamen lokal veri üzerinde çalışır; internet bağlantısı sadece Gemini API çağrısı için gereklidir.  

## Özellikler
- Tıbbi makaleler `data/` klasöründen okunur.
- SentenceTransformer ile embedding oluşturulur.
- ChromaDB ile embedding’ler saklanır.
- Kullanıcı sorusu ile ilgili en alakalı makaleler retrieval edilir.
- Gemini 2.0 Flash modeli ile yanıt üretilir.
- Terminal tabanlı basit chat arayüzü.

## Kurulum
1. Repo klonlanır:
```bash

git clone <repo-url>
cd CHAT_BOT

python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows

Create Data Folder in your Python Project

pip install -r requirements.txt


API_KEY=YOUR_GOOGLE_API_KEY



python3 chat.py
## Veritabanı ve Kullanıcıya Sağladığı Kolaylıklar

Projede kullanılan veritabanı, Türkiye'deki farklı hastanelere ait tıbbi makaleleri içermektedir (Acıbadem, Anadolu Sağlık, Liv, Medicana vb.).  

### Neden bu veritabanı?  
- Gerçek ve çeşitli tıbbi içeriklerle modelin yanıtları güçlendiriliyor.  
- Model yalnızca hayali ya da genel bilgilerle cevap vermiyor; lokal ve doğrulanabilir içerik üzerinden yanıt üretiyor.  
- Kullanıcı, spesifik sağlık konularında hızlı ve doğru bilgiye ulaşabiliyor.  

### Kullanıcıya sağladığı kolaylıklar  
- **Hızlı erişim:** Sorulan soruya en ilgili makaleler retrieval yöntemiyle hemen çekiliyor.  
- **Doğruluk:** Model, lokal veritabanından beslenerek daha güvenilir yanıtlar veriyor.  
- **Offline kullanım:** İnternet bağlantısına ihtiyaç duymadan sistem çalışıyor; veriler lokal olarak saklanıyor.  
- **Geniş kapsam:** Farklı hastanelerin verilerini kapsadığı için tek bir kaynağa bağlı kalmadan kapsamlı yanıt alabiliyorsunuz.
- ** Parquet dosyalarını Data dosyası içerisine yerleştirililmesi gerekmektedir.


**Dataset:** [umutertugrul/turkish-hospital-medical-articles](https://huggingface.co/datasets/umutertugrul/turkish-hospital-medical-articles)

