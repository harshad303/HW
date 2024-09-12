import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from anthropic import Anthropic
from anthropic.types.message import Message
import google.generativeai as genai


st.title("This is HW 2")


#Function to read URL Content.
def read_url_content(url):
	try:
		response = requests.get(url)
		response.raise_for_status() # Raise an exception for HTTP errors
		soup = BeautifulSoup(response.content, 'html.parser')
		return soup.get_text()
	except requests.RequestException as e:
		print(f"Error reading {url}: {e}")
		return None


openai_api_key = st.secrets["openai_api_key"] 
claude_api_key = st.secrets["claude_api_key"] 
google_api_key = st.secrets["google_api_key"] 

# Create an OpenAI client.
clientopenai = OpenAI(api_key=openai_api_key)

# Create an Claude client.
clientclaude = Anthropic(api_key = claude_api_key)

# Let the user upload a URL ⁠.
url = st.text_area(
        "Upload a URL here:",
        placeholder="Website URL for eg: www.google.com"
    )

	
# Ask the user for a question type via radibutton ⁠.
#question = "Can you please Summarise this for me:"

# Sidebar for selecting summary type (similar to your previous Lab2)
summary_type = st.selectbox("Select Summary Type", ["Summarize this document in 100 words", "Summarize this document in 2 connecting paragraphs", "Summarize this document in 5 bullet points"])

# Step 8: Dropdown menu to select output language
language = st.selectbox("Select Output Language", ["English", "French", "Spanish","Hindi"])

# Step 10: Option to select LLM models
llm_model = st.sidebar.selectbox("Select LLM", ["OpenAI", "Claude", "Google"])




if url: 
	content = read_url_content(url)
	
	if content and summary_type and language:
		question = summary_type
		messages_openai = [
            {
                "role": "user",
                "content": f"Here's a document: {content} \n\n---\n\n {question} in {language}",
            }
        ]
		messages_claude = [{'role': 'user', 
		"content": f"Here's a document: {content} \n\n---\n\n {question} in {language}"}]
		messages_google = f"Here's a document: {content} \n\n---\n\n {question} in {language}"
		

		if llm_model == "OpenAI":
			stream = clientopenai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_openai,
            stream=True,)
			st.write("Open AI's Response:")
			st.write_stream(stream)
		elif llm_model =="Claude":
			#Enter code for Claude using Claude Syntax.
			st.write("Claude's  Response:")
			response: Message = clientclaude.messages.create(
				max_tokens=256,
				messages= messages_claude,
				model="claude-3-haiku-20240307",
				temperature=0.5,)
			answer = response.content[0].text
			st.write(answer)
			
		elif llm_model == "Google":
			#Enter code for Cohere using Cohere Syntax.
			st.write("Google's Gemini  Response:")
			genai.configure(api_key=google_api_key)
			model = genai.GenerativeModel('gemini-1.5-flash')
			response = model.generate_content(messages_google)
			st.write(response.text)
else:
	
	st.write("Enter a valid URL")