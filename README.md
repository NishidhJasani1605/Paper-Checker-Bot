# Paper Checker Bot - Streamlit App

This is a Streamlit web application version of the Paper Checker Bot, designed to automate the process of checking and evaluating student answer sheets against official answer keys using AI.

## Features

1. **All-in-One Process**: Upload all required PDFs at once and process them in a single workflow
2. **PDF Processing**: Convert PDF answer sheets to images and preprocess them for better text recognition
3. **Student Answer Extraction**: Extract student answers from handwritten answer sheets
4. **Official Answer Extraction**: Extract official answers from answer key PDFs
5. **Automated Evaluation**: Compare student answers with official answers and generate evaluation reports

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. For PDF processing, you'll need to install poppler:
   - **Windows**: Download from [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/) and add the bin folder to your PATH
   - **Mac**: `brew install poppler`
   - **Linux**: `apt-get install poppler-utils`

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Enter your Gemini API key in the sidebar

4. Follow the steps in each tab:
   - **All-in-One Process**: Upload all three PDFs (student answer, question paper, and official answer key) at once and process them in a single workflow
   - **PDF Processing**: Upload and process the student's answer sheet PDF
   - **Student Answer Extraction**: Extract student answers from the processed images
   - **Official Answer Extraction**: Extract official answers from the answer key PDF
   - **Evaluation**: Compare student answers with official answers and generate an evaluation report

## API Key

This application requires a Google Gemini API key. You can get one by:

1. Going to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Creating an account if you don't have one
3. Creating an API key

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Required Python packages
- `pdf_pages/`: Folder containing extracted PDF pages as images
- `preprocessed_pages/`: Folder containing preprocessed images for better text recognition
- `Student_QNA/`: Folder containing extracted student answers in JSON format
- `Original_Answer/`: Folder containing extracted official answers in JSON format
- `Final_Evaluation/`: Folder containing evaluation reports

## Notes

- The application uses Google's Gemini AI for text extraction and evaluation
- For best results, ensure the PDF scans are clear and legible
- The evaluation is based on semantic understanding, not just keyword matching