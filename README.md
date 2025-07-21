🧠 Laffoon AI — Smart Web-Based Assistant
Laffoon AI is a Flask-powered conversational assistant that combines AI summarization, web scraping, speech recognition, and classic ML techniques to answer questions, summarize articles, and interact with users in natural language.

🚀 Features
🔹 Web Search Summarization (DuckDuckGo & Wikipedia)

🔹 Article Content Extraction & Cleaning

🔹 AI-Powered Text Summarization (Facebook BART Model)

🔹 Custom Chatbot with TF-IDF & BERT Embeddings

🔹 Speech Recognition & Text-to-Speech Output

🔹 Simple Web UI via Flask

🔹 Memory-based Response Training

🛠️ Tech Stack
Backend: Python, Flask

NLP Models: Hugging Face Transformers, SentenceTransformers

ML: scikit-learn (KNN Classifier)

Web Scraping: BeautifulSoup, DuckDuckGo Search API

Speech: speech_recognition, gTTS

Frontend: HTML (Flask Templates)

💻 How It Works
User sends a message via Web UI (or voice).

System checks prebuilt responses or tries to answer from memory.

If needed, it fetches and summarizes web content.

Generates a response using ML (TF-IDF/BERT) or fallback search results.

Supports text-to-speech playback for responses.

🔧 Installation:
git clone https://github.com/adam77461/laffoon3.git
cd laffoon3
pip install -r requirements.txt
python main.py
