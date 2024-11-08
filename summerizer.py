import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

# Initialize the summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to split text into chunks that fit within the model's token limit
def split_text(text, max_words=500):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield ' '.join(words[i:i + max_words])

# Streamlit UI setup
st.title("Webpage Summarizer")
st.write("Enter a URL to summarize the webpage content.")

# URL input
url = st.text_input("Insert the URL:")

# Button to trigger summarization
if st.button("Summarize"):
    if url:
        try:
            # Fetch and parse the text from the webpage
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            
            # Summarize each chunk and handle large text
            summaries = []
            for chunk in split_text(text):
                try:
                    summary = summarizer(chunk, do_sample=False)[0]['summary_text']
                    summaries.append(summary)
                except IndexError:
                    st.error("Error: Chunk exceeded model token limit.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

            # Combine all summaries and display
            final_summary = ' '.join(summaries)
            st.subheader("Summary")
            st.write(final_summary)
        
        except Exception as e:
            st.error(f"Error fetching or summarizing content: {e}")
    else:
        st.warning("Please enter a valid URL.")
