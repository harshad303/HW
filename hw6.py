import streamlit as st
from openai import OpenAI
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import os
import sys

# SQLite workaround for Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings

# App title and description
st.title("Legal Insights Bot")
st.write("This bot provides insights on recent legal news, leveraging AI capabilities.")

# Initialize session state if needed
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []
if 'memory' not in st.session_state:
    st.session_state.memory = deque(maxlen=5)
if 'is_system_ready' not in st.session_state:
    st.session_state.is_system_ready = False
if 'openai_session' not in st.session_state:
    api_key = st.secrets["OPENAI_KEY"]
    st.session_state.openai_session = OpenAI(api_key=api_key)
if 'last_query_type' not in st.session_state:
    st.session_state.last_query_type = None

# Setup ChromaDB client
if 'chroma_session' not in st.session_state:
    try:
        db_directory = os.path.join(os.getcwd(), "chroma_db")
        st.session_state.chroma_session = chromadb.PersistentClient(path=db_directory)
        st.session_state.collection = st.session_state.chroma_session.get_or_create_collection(name="legal_news_data")
        st.success("ChromaDB is ready to use!")
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        st.stop()

# Function to load news data into ChromaDB
def load_news_data(csv_file, collection):
    if not os.path.exists(csv_file):
        st.error("Unable to locate CSV file.")
        st.stop()
    
    data = pd.read_csv(csv_file)
    for idx, record in data.iterrows():
        try:
            date = (datetime(2000, 1, 1) + timedelta(days=int(record['days_since_2000']))).strftime('%Y-%m-%d')
            content = f"{record['company_name']} on {date}: {record['Document']} ({record['URL']})"
            
            # Generate OpenAI embedding
            embedding = create_openai_embedding(content)
            collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[{
                    "company": record['company_name'],
                    "date": date,
                    "url": record['URL']
                }],
                ids=[f"doc_{idx}"]
            )
        except Exception as e:
            st.error(f"Error processing record {idx}: {e}")

# Helper function to create embeddings
def create_openai_embedding(text):
    st.info("📡 Contacting OpenAI to generate an embedding for your question...")
    response = st.session_state.openai_session.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Function to classify the query as conversational or news-related
def classify_query(query):
    # Automatically treat follow-ups as "news-related" if the last query was "news-related"
    if st.session_state.last_query_type == "news-related" and "expand" in query.lower():
        return "news-related"

    classification_prompt = [
        {"role": "system", "content": "You are an assistant that determines the type of question being asked."},
        {"role": "user", "content": f"Classify the following query as either 'conversational' or 'news-related': '{query}'"}
    ]
    response = st.session_state.openai_session.chat.completions.create(
        model="gpt-4o",
        messages=classification_prompt,
        temperature=0
    )
    classification = response.choices[0].message.content.strip().lower()
    st.write(f"Debug: Full classification response = '{classification}'")
    
    # Check if "news-related" or "conversational" is in the response
    if "news-related" in classification:
        return "news-related"
    elif "conversational" in classification:
        return "conversational"
    else:
        return "unknown"

# Initialize data if the collection is empty
csv_file_path = os.path.join(os.getcwd(), "Example_news_info_for_testing.csv")
if st.session_state.collection.count() == 0:
    st.info("Loading news articles into the system...")
    load_news_data(csv_file_path, st.session_state.collection)
    st.success("News data loaded successfully!")
else:
    st.info("Using existing data collection.")

st.session_state.is_system_ready = True

# Display chat interface
st.subheader("Chat with the Legal Insights Bot")

# Show chat history
for message in st.session_state.chat_log:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input
user_query = st.chat_input("Enter your query about legal news (e.g., 'latest updates on regulations'):")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        # Classify the query type using the LLM
        query_type = classify_query(user_query)

        if query_type == "conversational":
            # Respond with a conversational message
            response_text = "I'm here to help with legal news insights! How can I assist you with recent updates?"
            st.write("Debug: Entered conversational response")
            with st.chat_message("assistant"):
                st.markdown(response_text)

            # Log conversational response to chat history
            st.session_state.chat_log.append({"role": "user", "content": user_query})
            st.session_state.chat_log.append({"role": "assistant", "content": response_text})
            st.session_state.last_query_type = "conversational"
        
        elif query_type == "news-related":
            # Inform about embedding creation
            st.info("📡 Requesting embedding for your query from OpenAI...")

            # Create query embedding
            query_embedding = create_openai_embedding(user_query)
            
            # Query ChromaDB
            st.subheader("Database Search")
            st.info("🔍 Searching the database for related articles...")
            results = st.session_state.collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            relevant_articles = results['documents'][0]
            article_details = [f"{meta['company']} - {meta['date']}" for meta in results['metadatas'][0]]
            
            # Display retrieved articles
            st.subheader("Related Articles")
            if len(article_details) > 0:
                st.info(f"📰 Identified {len(article_details)} related news articles:")
                for detail in article_details:
                    st.write(f"📄 {detail}")
            else:
                st.warning("No relevant articles found.")
            
            # Prepare LLM input
            system_instruction = """You are an AI specialized in analyzing news stories with legal relevance for a global law firm. Your role is to evaluate and summarize legal impacts, regulatory changes, and business implications.

            When ranking the most significant stories, prioritize:
            1. Legal and regulatory implications
            2. Influence on business practices
            3. Precedent-setting legal cases or decisions
            4. Compliance factors
            5. Sector-wide impact

            For each article, include:
            - Summary of the event
            - Key legal points
            - Business relevance
            - Importance to a law firm."""
            
            # Build conversation history
            chat_history = "\n".join([
                f"User: {entry['question']}\nBot: {entry['answer']}" 
                for entry in st.session_state.memory
            ])
            
            # Structure message for LLM
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"""Context: {' '.join(relevant_articles)}

Previous conversation:
{chat_history}

Query: {user_query}"""
                }
            ]
            
            # Stream LLM response
            st.write("🤔 Processing your input and preparing a response...")
            response = st.session_state.openai_session.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                stream=True
            )
            
            # Stream response in real time
            with st.chat_message("assistant"):
                response_holder = st.empty()
                complete_response = ""
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        complete_response += chunk.choices[0].delta.content
                        response_holder.markdown(complete_response)
            
            # Update chat history and memory
            st.session_state.chat_log.append({"role": "user", "content": user_query})
            st.session_state.chat_log.append({"role": "assistant", "content": complete_response})
            st.session_state.memory.append({
                "question": user_query,
                "answer": complete_response
            })
            st.session_state.last_query_type = "news-related"
            
            # Expandable section for article sources
            with st.expander("📄 View Articles"):
                for detail in article_details:
                    st.write(f"- {detail}")
        
        else:
            st.write("Debug: Unhandled classification type")
            response_text = "I'm not sure how to handle your query. Please try asking in a different way."
            with st.chat_message("assistant"):
                st.markdown(response_text)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
