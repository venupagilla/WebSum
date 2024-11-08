from transformers import pipeline
import requests
from bs4 import BeautifulSoup

# Initialize the summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to split text into chunks that fit within the model's token limit
def split_text(text, max_words=500):  # Reducing max_words to ensure token limit
    words = text.split()
    for i in range(0, len(words), max_words):
        yield ' '.join(words[i:i + max_words])

# Fetch and parse the text from the webpage
url = input("Insert the URL: ")
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
text = ' '.join(p.get_text() for p in soup.find_all('p'))

# Summarize each chunk and handle large text
summaries = []
for chunk in split_text(text):
    try:
        # Adjust max_length and min_length based on desired summary size
        summary = summarizer(chunk,do_sample=False)[0]['summary_text']
        summaries.append(summary)
    except IndexError:
        print("Error: Chunk exceeded model token limit.")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Combine all summaries
final_summary = ' '.join(summaries)
print("Summary:", final_summary)
