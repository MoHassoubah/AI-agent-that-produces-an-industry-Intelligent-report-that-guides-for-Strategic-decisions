import requests
import json
from bs4 import BeautifulSoup
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from typing import List
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import torch


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: Analyze trends, competitors, and emerging insights in the target market.

Generate a detailed, structured report that includes insights, competitor analysis, and strategic recommendations.
Ignore the irrlevant information from the context
"""


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
        # region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings



folder_path="test resources"


docx_DATA_PATH = "D:/scripts/seitech_gpt/data/docx"
pdf_DATA_PATH = "D:/scripts/seitech_gpt/data/pdfs"
   
      

def generate_response(prompt, model='llama2'):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    print(prompt)
    print('LLM response')
    print(response)
    return json.loads(response.text)['response']


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print("CUDA optimizations enabled...")
else:
    print("CUDA is not available")
    




def search_web(query="tell me the highlights from Apple's 2024 ESG report"):
    url = "https://api.langsearch.com/v1/web-search"

    payload = json.dumps({
      "query": query,
      "freshness": "noLimit",
      "summary": True,
      "count": 10
    })
    headers = {
      'Authorization': 'Bearer sk-1e5f685302714af0bef6d5b2883db56a',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response)
    # p
    # print(response.text)

    # response = requests.get(url, headers=headers)
    print(response.status_code)

    if response.status_code == 200:
        results = response.json()
        print(results["data"]["webPages"]["value"][0].keys())
        links = [item["url"] for item in results["data"]["webPages"]["value"]]
        
        return links
    else:
        print("Error fetching search results.")
        return []


def extract_content(url):
    try:
        headers = {
            "User-Agent": "Chrome/91.0.4472.124"
        }

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs[:5]])  # Fetch first 5 paragraphs
        return content
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


def parse_text(text):
    """Parse the provided text into a structured format."""
    sections = {}
    lines = text.split("\n")
    current_section = None

    for line in lines:
        line = line.strip()  # Remove extra whitespace
        if ":" in line:  # Check if the line contains a colon
            # Split the line into heading (before the colon) and content (after the colon)
            parts = line.split(":", 1)  # Split only on the first colon
            current_section = parts[0].strip()  # Heading before the colon
            sections[current_section] = parts[1].strip()  # Content after the colon
        elif current_section:  # Add additional lines to the current section content
            sections[current_section] += " " + line.strip()

    return sections
    
def generate_pdf(output_filename, structured_content):
    """Generate a PDF report with the structured content."""
    doc = SimpleDocTemplate(output_filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Add Title to the PDF
    title = Paragraph("<b>Analysis of Trends, Competitors, and Emerging Insights in the Target Market</b>", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 20))  # Add space after the title

    # Add sections
    for section_title, section_content in structured_content.items():
        # Add section heading
        story.append(Paragraph(f"<b>{section_title}</b>", styles["Heading2"]))
        story.append(Spacer(1, 10))

        # Add section content
        story.append(Paragraph(section_content, styles["Normal"]))
        story.append(Spacer(1, 20))  # Add space after each section

    # Build the PDF document
    doc.build(story)

        
while(True):
    query_text = input(">Ask: ")
    if query_text == 'exit':
        print("thank you :)")
        break
        
    
        
    ret_links = search_web(query_text)

    
    context_text_parts = []  # Create an empty list to collect content from each link
    for link in ret_links:
        context_text_parts.append(extract_content(link))  # Call extract_content and add result to the list
        time.sleep(2)  # Wait for 2 seconds after processing each link

    context_text = "\n\n---\n\n".join(context_text_parts)  # Join the collected content with separators
    
    start_time = time.time()
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text)
    end_time = time.time()
    response = generate_response(prompt)

    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f">>Response: {response}"
    print(formatted_response)
    print(" ")
    print(" ")
    
    

    # Parse text into structured content
    structured_content = parse_text(response)

    # Generate the PDF
    generate_pdf("Detailed_Report.pdf", structured_content)

    print("PDF report generated successfully!")

