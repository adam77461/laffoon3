import json
import random
import time
import string
import wikipedia
import requests
import os
import re
import speech_recognition as sr
from gtts import gTTS
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline
import re

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)

# Load and save memory.json
def load_memory():
    try:
        with open("memory.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_memory(data):
    with open("memory.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# Get current time
def get_time():
    return time.strftime("%I:%M %p")

# Preprocess text
def preprocess_text(text):
    return text.lower().strip()

# Train chatbot using TF-IDF
def train_chatbot_tfidf(data):
    if not data:
        return None, None
    inputs = [preprocess_text(pair[0]) for pair in data]
    outputs = [pair[1] for pair in data]
    
    vectorizer = TfidfVectorizer()
    input_vectors = vectorizer.fit_transform(inputs)
    
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(input_vectors, outputs)
    
    return vectorizer, classifier

# Train chatbot using BERT embeddings
def train_chatbot_bert(data):
    if not data:
        return None, None
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    inputs = [preprocess_text(pair[0]) for pair in data]
    outputs = [pair[1] for pair in data]
    
    input_vectors = model.encode(inputs)
    
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(input_vectors, outputs)
    
    return model, classifier

# Generate response using TF-IDF
def generate_response_tfidf(input_text, vectorizer, classifier):
    input_vector = vectorizer.transform([preprocess_text(input_text)])
    return classifier.predict(input_vector)[0]

# Generate response using BERT
def generate_response_bert(input_text, model, classifier):
    input_vector = model.encode([preprocess_text(input_text)])[0].reshape(1, -1)
    return classifier.predict(input_vector)[0]



def clean_text(html_text):
    """Cleans HTML content and formats it properly."""
    # Remove all HTML tags
    clean = re.sub(r'<.*?>', '', html_text)

    # Improve formatting
    clean = re.sub(r'🔹 \*\*Summary:\*\*', '\n\n🔹 Summary:', clean)  # Fix bullet points
    clean = re.sub(r'🔗', '\n\n🔗 Read more:', clean)  # Format links

    # Fix duplicated links
    clean = re.sub(r'Read more: Read more:', 'Read more:', clean)

    # Add spacing for better readability
    clean = re.sub(r'📌', '\n📌', clean)

    return clean.strip()



def summarize_text(text, query, sentence_count=12):
    # Split the text into paragraphs, focusing on meaningful content blocks
    sections = text.split('\n\n')
    
    # Initialize an empty list for the summaries
    section_summaries = []
    
    # Process each section
    for section in sections:
        if section.strip():
            try:
                # Inject the query for better context
                summary = summarizer(f"{query}. Focus on {query}. {section}", 
                                     max_length=40, min_length=8, do_sample=False)
                
                # Filter irrelevant or repetitive summaries
                if summary and isinstance(summary, list) and 'summary_text' in summary[0]:
                    content = summary[0]['summary_text'].strip()
                    
                    # Skip unwanted "Focus on {query}" entries
                    print(content.lower().strip())
                    print(f"{query}. Focus on {query}. {section}".lower().strip())
                    if content == f"{query}. Focus on {query}. {section}".lower():
                        print(True)
                        break
                    if content.lower() == f"{query}. Focus on {query}. {section}".lower():
                        print(f"blocked {content}\n")
                        break
                    elif 'CNN.com' not in content:  # Filter out unrelated content
                        section_summaries.append(content)
            except Exception as e:
                print(f"Error summarizing section: {e}")
    
    # Combine the summaries
    full_summary = "<br><br>• ".join(section_summaries)
    
    return full_summary + "<br>What do you think of this 😊? Do you want to learn more?<br>"

def clean_article(article_text):
    article_text = re.sub(r"(This article was co-authored by.*?|wikiHow staff writer.*?|Follow Us[\s\S]*?newsletter|\d+ references cited in this article|Editor’s Note:.*?social account|CC BY-SA [\d.]+, via Wikimedia Commons)", "", article_text, flags=re.DOTALL)
    article_text = re.sub(r"(Affiliate Disclosure:.*?|This article contains affiliate links.*?|Register to read more.*?|Sign up for our newsletter.*?\n)", "", article_text, flags=re.IGNORECASE)
    article_text = re.sub(r"\s+", " ", article_text).strip()
    return article_text

def generate_rich_response(query):
    summary = fetch_duckduckgo_summary(query)
    wiki_summary = fetch_wikipedia_summary(query)
    # summary = "In testin mode for wiki summary"

    return f"{summary}\n\n📘 Additional Info from Wikipedia:\n{wiki_summary}"

def fetch_article(url):
    """Fetches and extracts the full content of a given URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Error fetching article: {e}")
        return None, None

    soup = BeautifulSoup(response.text, "html.parser")

    # Try extracting the main content
    content = ""
    
    # Check for common content tags in articles
    for tag in ["article", "div", "section"]:
        main_content = soup.find(tag)
        if main_content:
            paragraphs = main_content.find_all("p")
            content = "\n".join([p.get_text() for p in paragraphs])
            break  # Stop after finding the main content

    # Fallback to extracting all `<p>` tags if structured content isn't found
    if not content:
        paragraphs = soup.find_all("p")
        content = "\n".join([p.get_text() for p in paragraphs])

    # Clean the text for improved readability
    cleaned_content = clean_article(content)
    
    # Extract an image URL if available
    image_tag = soup.find("img")
    image_url = image_tag["src"] if image_tag else None

    return cleaned_content, image_url






# Fetch DuckDuckGo summary
def fetch_duckduckgo_summary(query):
    ddgs = DDGS()
    results = ddgs.text(query)
    summaries = []

    if results:
        for i, result in enumerate(results):
            title = result['title']
            link = result['href']
            summary = result.get('body')  # Prefer existing summary if available
            image_url = None
            # Fetch full article only if necessary
            if not summary:
                article_content = fetch_article(link)
                if article_content:
                    summary = summarize_text(article_content, query, sentence_count=3)

            # Skip if the summary is invalid
            if not summary or len(summary) < 20 or "register" in summary.lower() or "affiliate" in summary.lower():
                continue  

            print(f"Image URL: {image_url}")
            # Format summary
            formatted_summary = (
                f"<h3>📌According to {title}, here’s what I found:</h3><br><br>"
                f"🔹 **Summary:** {summary}<br><br>"
                f"🔗 [Read more]({link})<br>"
                f"<br>"
            )

            summaries.append(formatted_summary)

            # Stop after 5 summaries
            if i >= 1:
                break  

    # Fallback to Wikipedia if DuckDuckGo has no useful results
    if not summaries:
        return fetch_wikipedia_summary(query)

    return "<br>".join(summaries)  # Return as clean text (UI can format to HTML)


# Fetch Wikipedia summary
def fetch_wikipedia_summary(query):
    try:
        page = wikipedia.page(query)
        title = page.title
        content = page.content
        
        # Optionally limit the content length for readability
        max_length = 7000
        if len(content) > max_length:
            content = content[:max_length] + "...\n\n🔗 [Read more on Wikipedia](" + page.url + ")"
        
        return f"📘 **{title}**\n\n{summarize_text(content,query)}"
    
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Can you be more specific? Multiple results found: {', '.join(e.options[:3])}"
    except wikipedia.exceptions.PageError:
        return "I couldn't find any information on that."
    except Exception as e:
        return f"An error occurred: {e}"

# Check if user input is a general knowledge question
def is_general_knowledge_question(user_input):
    question_starters = []
    return any(user_input.lower().startswith(q) for q in question_starters)

# Load memory
memory_data = load_memory()
vectorizer_tfidf, classifier_tfidf = train_chatbot_tfidf(memory_data)
model_bert, classifier_bert = train_chatbot_bert(memory_data)

# Prebuilt responses
prebuilt_responses = {}


def HandOver(user_input,send_input):

    return generate_rich_response(user_input);

# Text-to-Speech
def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")

# Speech-to-Text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Could not request results, please check your internet connection."

@app.route("/")
# def SID(SID):
#     history = []
#     pass
def index():
    return render_template("index.html")


@app.route("/settingChat", methods=["POST"])
def setting_chat():
    data = request.json
    print("Received data:", data)  # Debug print to check what is received in the terminal

    # Use the "message" field as the chat summary if available
    chat_summary = data.get("message", data.get("summary", "New Chat"))
    return jsonify({"summary": chat_summary})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    chat_history = data.get("history", [])
    prompt = data.get("prompt", "Laffoon is a helpful AI assistant.")

    # Combine prompt + chat history for improved context
    full_input = user_input
    send_input = f"{prompt}\n{' '.join(chat_history)}\nUser: {user_input}"
    print(data)
    chatbot_response = ""

    # Prebuilt responses
    for keyword, response_func in prebuilt_responses.items():
        if keyword in user_input.lower():
            chatbot_response = response_func()
            return jsonify({"response": chatbot_response})

    if is_general_knowledge_question(user_input):
        chatbot_response = fetch_duckduckgo_summary(user_input)
        return jsonify({"response": chatbot_response})

    # Generate response using TF-IDF or BERT
    if vectorizer_tfidf and classifier_tfidf:
        try:
            response = generate_response_tfidf(full_input, vectorizer_tfidf, classifier_tfidf)
            if response == "HandOver()":
                response = HandOver(user_input,full_input)
        except:
            response = generate_response_bert(full_input, model_bert, classifier_bert)
            if response == "HandOver()":
                response = HandOver(user_input,full_input)

        chatbot_response = response

    # Append conversation to chat history
    chat_history.append(f"User: {user_input}")
    chat_history.append(f"Laffoon: {chatbot_response}")
    
    # Return both response and updated history
    return jsonify({"response": chatbot_response, "history": chat_history})


if __name__ == "__main__":
    app.run(debug=True,port=80,host="127.0.0.1")
