import os
import re
!pip install PyMuPDF
import fitz  # PyMuPDF for PDF handling
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt_tab')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords, lemmatizer, and unnecessary words
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
unnecessary_words = {"etc", "e.t.c", "eg", "i.e", "viz"}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    try:
        doc = fitz.open(pdf_path)  # Open the PDF file
        for page in doc:  # Extract text from each page
            pdf_text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return pdf_text

# Function to clean the extracted text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9.%\s]", "", text)  # Preserve numbers, percentages, and words
    words = word_tokenize(text)  # Tokenize the text
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in unnecessary_words]
    return " ".join(words)

# Path to your PDF
pdf_path = "/content/18-2013-Inland Revenue (Amendment) (E).pdf"

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Check if text was extracted
if not pdf_text:
    print("No text extracted from the PDF. Check the file path and content.")
    exit()

# Clean the extracted text
cleaned_text = clean_text(pdf_text)

# Save the cleaned text to a file
output_path = "/content/cleaned_text.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print(f"Processed text saved to {output_path}")


import os
import re
!pip install PyMuPDF
import fitz  # PyMuPDF for PDF handling
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt_tab')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords, lemmatizer, and unnecessary words
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
unnecessary_words = {"etc", "e.t.c", "eg", "i.e", "viz"}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    try:
        doc = fitz.open(pdf_path)  # Open the PDF file
        for page in doc:  # Extract text from each page
            pdf_text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return pdf_text

# Function to clean the extracted text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9.%\s]", "", text)  # Preserve numbers, percentages, and words
    words = word_tokenize(text)  # Tokenize the text
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in unnecessary_words]
    return " ".join(words)

# Path to your PDF
pdf_path = "/content/18-2013-Inland Revenue (Amendment) (E).pdf"

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Check if text was extracted
if not pdf_text:
    print("No text extracted from the PDF. Check the file path and content.")
    exit()

# Clean the extracted text
cleaned_text = clean_text(pdf_text)

# Save the cleaned text to a file
output_path = "/content/cleaned_text.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print(f"Processed text saved to {output_path}")


import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCqqEEP95KkL-HYBwm783BvM8hF0tqmRAg")
assert API_KEY, "ERROR: Gemini API Key is missing"

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Use the correct model (you can change to "gemini-1.5-flash-latest" for a faster response)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Get user inputs
policy = input("HTML ref of the policy (e.g., act, ordinance, law, regulation, circular, gazette): ")
scenario = input("Provide the scenario (e.g., Create a renewable energy policy for a developing country to reduce emissions by 20% by 2030.): ")

# Construct the prompt dynamically
prompt = f"Generate an economic policy from {policy} for the following scenario: {scenario}"

# Generate completion
response = model.generate_content(prompt)

# Print response
print(response.text)
