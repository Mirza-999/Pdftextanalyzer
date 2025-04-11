import os
import gradio as gr
import google.generativeai as genai
import chardet
import fitz  # PyMuPDF for PDF support
from langdetect import detect  # For language detection

# Load API Key from Hugging Face Secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key is missing. Set it in Hugging Face Secrets.")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

def read_file(file_path):
    if file_path.endswith(".pdf"):
        return read_pdf(file_path)
    else:
        return read_text(file_path)

def read_text(file_path):
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
        encoding = chardet.detect(raw_data)["encoding"] or "utf-8"
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"Error reading text file: {str(e)}"

def read_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"

def analyze_input(text, file):
    try:
        if file is not None:
            text = read_file(file)
        elif not text.strip():
            return "Please enter text or upload a file.", "", "", ""

        text = text[:3000]  # Limit input size for Gemini
        word_count = len(text.split())
        char_count = len(text)
        language = detect(text)

        prompt = f"""
        Analyze and summarize the following document. Highlight key information and named entities such as people, organizations, and locations.

        Text:
        {text}
        """
        response = model.generate_content([prompt], stream=True)
        result = "".join(chunk.text for chunk in response)

        return result, f"Word Count: {word_count}", f"Character Count: {char_count}", f"Language Detected: {language.upper()}"
    except Exception as e:
        return f"Error: {str(e)}", "", "", ""

def clear_inputs():
    return "", None, "", "", "", ""

def generate_downloadable_file(text):
    if text.strip():
        file_path = "analysis_result.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        return file_path
    return None

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("""
    # AI-Powered Text and PDF Analyzer
    Upload a .txt or .pdf file, or paste some text to get an instant AI-generated analysis, summary, and entity highlights.
    """)

    with gr.Row():
        text_input = gr.Textbox(label="Enter Text", placeholder="Paste text here or upload a file...", lines=6)
        file_input = gr.File(label="Upload File (.txt or .pdf)", type="filepath")  

    output_text = gr.Textbox(label="Analysis Result", lines=12, interactive=False)
    word_count_display = gr.Textbox(label="Word Count", interactive=False)
    char_count_display = gr.Textbox(label="Character Count", interactive=False)
    language_display = gr.Textbox(label="Language Detected", interactive=False)

    with gr.Row():
        analyze_button = gr.Button("Analyze", variant="primary")
        clear_button = gr.Button("Clear", variant="secondary")

    with gr.Column():
        gr.Markdown("Download Analysis Result")
        with gr.Row():
            download_button = gr.Button("Download Result", variant="success", size="sm")
            download_file = gr.File(label="Download File", interactive=False)

    analyze_button.click(analyze_input, inputs=[text_input, file_input],
                         outputs=[output_text, word_count_display, char_count_display, language_display])
    clear_button.click(clear_inputs, inputs=[], outputs=[text_input, file_input, output_text, word_count_display, char_count_display, language_display, download_file])
    download_button.click(generate_downloadable_file, inputs=output_text, outputs=download_file)

demo.launch()
