from opensearchpy import OpenSearch
import os
import requests
from pypdf import PdfReader
from transformers import AutoModel, AutoTokenizer
import torch

# Load the BGE model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')

def get_embedding(text):
    # Tokenize the text
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
    # Normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.tolist()[0]

def download_pdf(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status() # Raise an exception for HTTP errors
    with open(save_path, 'wb') as pdf_file:
        for chunk in response.iter_content(chunk_size=8192):
            pdf_file.write(chunk)
    print(f"Downloaded {url} to {save_path}")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def main():
    # Environment variables for OpenSearch connection
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    auth_username = os.getenv("OPENSEARCH_AUTH_USERNAME", "admin")
    auth_password = os.getenv("OPENSEARCH_AUTH_PASSWORD", "my_strong_password")

    # Create the client with SSL/TLS configuration
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True, # enables gzip compression for request bodies
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )

    # Test the connection
    try:
        info = client.info()
        print(f"Connected to OpenSearch: {info}")
    except Exception as e:
        print(f"Could not connect to OpenSearch: {e}")

    # Define OpenSearch index name and mapping
    index_name = "documents"
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "mapping": {
                    "total_fields": {
                        "limit": 10000
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "text_content": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384, # BGE-small-en-v1.5 has 384 dimensions
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib"
                    }
                }
            }
        }
    }

    # Create the index if not existing already
    if not client.indices.exists(index_name):
        client.indices.create(index=index_name, body=index_body)
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")

    # Create a directory for PDFs
    pdf_dir = "./pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    # Example PDF to download
    pdf_url = "https://arxiv.org/pdf/1706.03762"
    # pdf_url = "https://cdn.openai.com/gpt-5-system-card.pdf"
    pdf_filename = os.path.join(pdf_dir, "attention_is_all_you_need.pdf")

    try:
        download_pdf(pdf_url, pdf_filename)
        extracted_text = extract_text_from_pdf(pdf_filename)
        print(f"Extracted text length: {len(extracted_text)} characters")

        if extracted_text:
            embeddings = get_embedding(extracted_text)
            print(f"Generated embedding with dimension: {len(embeddings)}")

            # Index the document
            document = {
                "title": os.path.basename(pdf_filename),
                "text_content": extracted_text,
                "embedding": embeddings
            }
            response = client.index(index=index_name, body=document)
            print(f"Document indexed: {response['result']}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")

if __name__ == "__main__":
    main()
