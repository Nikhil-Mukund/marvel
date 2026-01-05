from multiprocessing import Pool
import os
import json
import re
import subprocess
from datetime import datetime, timedelta
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader
from config import config
configs = config.load_config()


def inject_metadata_to_content(text, metadata):
    metadata_str = " ".join([f"{k}:{v}" for k, v in metadata.items()])
    return f"[METADATA] {metadata_str} [/METADATA]\n{text}"


def run_ocr_in_surya_env(pdf_path):
    # Define the command to activate the conda environment and run the script
    command = f"conda run -n surya-ocr python run_ocr_script.py '{pdf_path}'"
    # Execute the command
    print("Executing {}".format(command))
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    output, errors = process.communicate()
    if process.returncode == 0:
        return output.strip()
    else:
        raise Exception(f"Error in OCR processing: {errors}")


def convert_pdf_date(date_string):
    # Extract date and time components from PDF dateime format like:  "D:20170520092133+05'30'"
    match = re.match(
        r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})([+-])(\d{2})'(\d{2})'", date_string)
    if not match:
        raise ValueError("Invalid PDF date format")
    year, month, day, hour, minute, second, sign, tz_hour, tz_minute = match.groups()
    # Create a datetime object
    dt = datetime(int(year), int(month), int(day),
                  int(hour), int(minute), int(second))
    # Apply timezone offset
    tz_offset = timedelta(hours=int(tz_hour), minutes=int(tz_minute))
    if sign == '-':
        tz_offset = -tz_offset
    dt = dt - tz_offset
    return dt

def process_pdf(pdf_path, chunk_size=1500, chunk_overlap=300):
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Convert PDF pages to images and extract text (including equations) using OCR
    images = convert_from_path(pdf_path)
    combined_text = []
    for image in images:
        text = pytesseract.image_to_string(image, lang='eng')
        combined_text.append(text)
    # Split text into chunks
    return text_splitter.create_documents(combined_text)


def extract_text_data(text_directory, chunk_size=1500, chunk_overlap=500):
    print(
        f"Extracting data from text files in {text_directory} and its subfolders")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = []
    chunk_index = 0
    for root, dirs, files in os.walk(text_directory):
        for filename in files:
            if filename.endswith(".txt"):
                text_path = os.path.join(root, filename)
                try:
                    with open(text_path, 'r', encoding='utf-8') as file:
                        text_data = file.read()
                    # Get the relative path starting from RAG_DataSets
                    relative_path = os.path.relpath(
                        text_path, start=os.path.dirname(text_directory))
                    relative_path = os.path.join("RAG_DataSets", relative_path)
                    chunks = text_splitter.split_text(text_data)
                    for chunk in chunks:
                        metadata = {
                            "chunk_index": chunk_index,
                            "filename": relative_path
                        }
                        content_with_metadata = inject_metadata_to_content(
                            chunk, metadata)
                        all_splits.append(
                            Document(page_content=content_with_metadata, metadata={}))
                        chunk_index += 1
                    print(f"Processed: {text_path}")
                except Exception as e:
                    print(f"Error processing {text_path}: {str(e)}")
    print(f"Split text data into {len(all_splits)} chunks")
    return all_splits


def extract_JSONL_data(jsonl_directory, chunk_size=1500):
    print(f"Extracting data from JSONL files in {jsonl_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    all_documents = []
    # Walk through the directory to process all JSONL files in subdirectories
    for root, dirs, files in os.walk(jsonl_directory):
        for filename in files:
            if filename.endswith(".jsonl"):
                jsonl_path = os.path.join(root, filename)
                with open(jsonl_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        try:
                            json_obj = json.loads(line)
                            messages = json_obj.get("messages", [])
                            combined_content = ""
                            for message in messages:
                                role = message.get("role", "")
                                content = message.get("content", "")
                                combined_content += f"{role.capitalize()}: {content}\n\n"
                            # Split the content into chunks
                            chunks = text_splitter.split_text(combined_content)
                            # Create Document objects with metadata
                            for chunk in chunks:
                                doc = Document(page_content=chunk, metadata={
                                               "source_file": filename})
                                all_documents.append(doc)
                        except json.JSONDecodeError as e:
                            print(
                                f"Error decoding JSON in file {filename}: {e}")
                        except Exception as e:
                            print(f"Unexpected error in file {filename}: {e}")
    print(f"Extracted {len(all_documents)} documents from JSONL files")
    return all_documents


def extract_heyligo_ProFreports_csv_data(heyligo_csv_directory, chunk_size=1500, chunk_overlap=0):
    print(f"extracting data from heyligo-style ProFreports_csv files")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # List to hold all text splits
    all_splits = []
    # Iterate through each CSV file in the directory
    for filename in os.listdir(heyligo_csv_directory):
        if filename.endswith(".csv"):
            csv_path = os.path.join(heyligo_csv_directory, filename)
            # Read the CSV file and select specific columns
            df = pd.read_csv(csv_path, usecols=[
                             'Unnamed: 0', 'author_id', 'comments', 'content', 'file_urls', 'files', 'report_time', 'section', 'title'])
            # Create the combined_text column
            df['combined_text'] = (
                "Title:" + df['title'] + "," +
                "Logbook entry: " + df['Unnamed: 0'].astype(str) + "," +
                "Author name :" + df['author_id'].astype(str) + "," +
                "section:" + df['section'].astype(str) + "," +
                "report_time:" + df['report_time'].astype(str) + "," +
                "content:" + df['content'].astype(str) + "," +
                "comments:" + df['comments'].astype(str) + "," +
                "files:" + df['files'].astype(str)
            )
            # Process each item in df['combined_text']
            for combined_text in df['combined_text']:
                if configs['retrieval']['enable_verbose']:
                    print("###################")
                    print(combined_text)
                    print("###################")
                if not pd.isna(combined_text):
                    all_splits += text_splitter.create_documents(
                        [combined_text])
    print(
        f"Splitting Heyligo-style ProFreports_csv data into {len(all_splits)} chunks")
    return all_splits


def extract_alpaca_json_data(RAG_DataSet_directory, chunk_size=1500, chunk_overlap=0):
    print(f"extracting data from alpaca-style json files")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # List to hold all text splits
    all_splits = []
    # Iterate through each JSON file in the directory
    for filename in os.listdir(RAG_DataSet_directory):
        if filename.endswith(".json"):
            json_path = os.path.join(RAG_DataSet_directory, filename)
            # Load JSON file
            with open(json_path, 'r') as file:
                json_data = json.load(file)
            # Process each item in the JSON array
            for item in json_data:
                combined_text = item['instruction'] + " " + item['output']
                all_splits += text_splitter.create_documents([combined_text])
    print(f"Splitting json data into {len(all_splits)} chunks")
    return all_splits


def extract_ligo_roster_json_data(RAG_DataSet_directory, chunk_size=1500, chunk_overlap=0):
    print(f"extracting data from alpaca-style json files")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # List to hold all text splits
    all_splits = []
    # Iterate through each JSON file in the directory
    for filename in os.listdir(RAG_DataSet_directory):
        if filename.endswith(".json"):
            json_path = os.path.join(RAG_DataSet_directory, filename)
            # Load JSON file
            with open(json_path, 'r') as file:
                json_data = json.load(file)
            # Process each item in the JSON array
            for item in json_data:
                combined_text = item['name'] + " is from " + item['group'] + \
                    " has the email " + \
                    item['email'] + " and phone number " + item["phone_number"]
                all_splits += text_splitter.create_documents([combined_text])
    print(f"Splitting json data into {len(all_splits)} chunks")
    return all_splits

# [OFFICIAL METHOD]IMPROVED Processing PDF files using PDF2IMAGE


def extract_pdf_data_V2(pdf_directory, chunk_size=configs['retrieval']['chunk_size'], chunk_overlap=configs['retrieval']['chunk_overlap'], extraction_method="tesseract"):
    print("extracting data from pdf files using method: {}".format(extraction_method))
    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # List to hold all text splits
    all_splits = []
    # Iterate through each PDF in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            # Load PDF and get text content
            print(f"Loading: {pdf_path}")
            # Convert PDF pages to images and extract text (including equations) using OCR
            if extraction_method == "tesseract":
                images = convert_from_path(pdf_path)
                combined_text = []
                for image in images:
                    text = pytesseract.image_to_string(image, lang='eng')
                    combined_text.append(text)
            # Split text into chunks and add to the all_splits list
            elif extraction_method == "surya":
                combined_text = []
                text = run_ocr_in_surya_env(pdf_path)
                combined_text.append(text)
            all_splits += text_splitter.create_documents(combined_text)
            print(combined_text)
            print(all_splits)
            print(f"Splitting pdf docs data into {len(all_splits)} chunks")
    return all_splits

def extract_pdf_data_V2_parallel(pdf_directory, num_processes, chunk_size=1500, chunk_overlap=300):
    print("Extracting data from pdf files in parallel")
    # Get all PDF file paths
    pdf_paths = [os.path.join(pdf_directory, f)
                 for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    # Create a pool of worker processes
    with Pool(num_processes) as pool:
        # Use starmap to pass multiple arguments
        results = pool.starmap(
            process_pdf, [(pdf_path, chunk_size, chunk_overlap) for pdf_path in pdf_paths])

def extract_pdf_data_V3(pdf_directory, chunk_size=configs['retrieval']['chunk_size'], chunk_overlap=configs['retrieval']['chunk_overlap'], extraction_method="tesseract"):
    print(f"Extracting data from pdf files using method: {extraction_method}")
    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # List to hold all text splits
    all_splits = []
    # Iterate through each PDF in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"Loading: {pdf_path}")
            # Extract metadata
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                metadata = pdf_reader.metadata
                try:
                    publish_date = convert_pdf_date(
                        metadata.get('/CreationDate'))
                except:
                    publish_date = 2024

            # Convert PDF pages to images and extract text (including equations) using OCR
            if extraction_method == "tesseract":
                images = convert_from_path(pdf_path)
                for page_num, image in enumerate(images, start=1):
                    text = pytesseract.image_to_string(image, lang='eng')
                    # Create Document with metadata
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "publish_date": publish_date,
                            "page": page_num
                        }
                    )
                    all_splits.extend(text_splitter.split_documents([doc]))
            elif extraction_method == "surya":
                text = run_ocr_in_surya_env(pdf_path)
                # Create Document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "publish_date": publish_date,
                        "page": "all"  # Since Surya method doesn't provide page-level information
                    }
                )
                all_splits.extend(text_splitter.split_documents([doc]))
    print(f"Splitting pdf docs data into {len(all_splits)} chunks")
    return all_splits


# extract chunks from all the latex .tex files from sub-directories within directory
def extract_latex_data(latex_directory, chunk_size=512, chunk_overlap=256):
    def parse_latex(file_path):
        print(f"Parsing LaTeX file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # Define patterns for sections, subsections, equations, tables, and figures
        section_pattern = r'\\section\{[^}]+\}.*?(?=\\section\{|\\subsection\{|\\subsubsection\{|\\begin\{equation\}|\\begin\{table\}|\\begin\{figure\}|\\begin\{itemize\}|\\begin\{enumerate\}|$)'
        subsection_pattern = r'\\subsection\{[^}]+\}.*?(?=\\section\{|\\subsection\{|\\subsubsection\{|\\begin\{equation\}|\\begin\{table\}|\\begin\{figure\}|\\begin\{itemize\}|\\begin\{enumerate\}|$)'
        subsubsection_pattern = r'\\subsubsection\{[^}]+\}.*?(?=\\section\{|\\subsection\{|\\subsubsection\{|\\begin\{equation\}|\\begin\{table\}|\\begin\{figure\}|\\begin\{itemize\}|\\begin\{enumerate\}|$)'
        equation_pattern = r'\\begin\{equation\}.*?\\end\{equation\}'
        table_pattern = r'\\begin\{table\}.*?\\end\{table\}'
        figure_pattern = r'\\begin\{figure\}.*?\\end\{figure\}'
        itemize_pattern = r'\\begin\{itemize\}.*?\\end\{itemize\}'
        enumerate_pattern = r'\\begin\{enumerate\}.*?\\end\{enumerate\}'
        # Combine all patterns
        all_patterns = f'({section_pattern}|{subsection_pattern}|{subsubsection_pattern}|{equation_pattern}|{table_pattern}|{figure_pattern}|{itemize_pattern}|{enumerate_pattern})'
        # Split the content
        splits = re.split(all_patterns, content, flags=re.DOTALL)
        # Optional: Remove empty strings if they are not needed
        splits = [split for split in splits if split.strip()]
        print(f"Parsed {len(splits)} chunks from LaTeX file.")
        return splits

    def create_chunked_documents(chunks, file_path, chunk_size, chunk_overlap):
        documents = []
        doc_id = 0
        for chunk in chunks:
            # Split the chunk into smaller pieces
            start = 0
            while start < len(chunk):
                end = start + chunk_size
                # Create a new document
                doc_content = chunk[start:end].strip()
                if doc_content:  # Only add non-empty documents
                    documents.append(Document(
                        page_content=doc_content,
                        metadata={'doc_id': f'chunk_{doc_id}',
                                  'source': file_path}
                    ))
                    doc_id += 1
                # Move the start position for the next chunk, considering overlap
                start = end - chunk_overlap
        return documents
    print("Processing LaTeX documents...")
    all_documents = []
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(latex_directory):
        for filename in files:
            if filename.endswith(".tex"):
                file_path = os.path.join(root, filename)
                try:
                    chunks = parse_latex(file_path)
                    documents = create_chunked_documents(
                        chunks, file_path, chunk_size, chunk_overlap)
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    print(f"Created {len(all_documents)} documents from LaTeX chunks.")
    return all_documents

# code to vectorstore embed in batches
