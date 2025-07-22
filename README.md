# Paper Checker Bot - Streamlit App

This is a Streamlit web application version of the Paper Checker Bot, designed to automate the process of checking and evaluating student answer sheets against official answer keys using AI.

## Features

1. **All-in-One Process**: Upload all required PDFs at once and process them in a single workflow
2. **PDF Processing**: Convert PDF answer sheets to images and preprocess them for better text recognition
3. **Student Answer Extraction**: Extract student answers from handwritten answer sheets
4. **Official Answer Extraction**: Extract official answers from answer key PDFs
5. **Automated Evaluation**: Compare student answers with official answers and generate evaluation reports

## Streamlit App Workflow (Tabs)

The app is organized into five main tabs, each representing a step in the workflow:

1. **All-in-One Process**: Upload all three PDFs (student answer, question paper, and official answer key) and process them in a single workflow.
2. **PDF Processing**: Upload and process the student's answer sheet PDF to extract and preprocess images.
3. **Student Answer Extraction**: Extract student answers from the processed images using AI.
4. **Official Answer Extraction**: Extract official answers from the answer key PDF using AI.
5. **Evaluation**: Compare student answers with official answers and generate an evaluation report.

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

4. **Environment Variables**: (Optional, for advanced configuration)
   - Copy `.env.example` to `.env` and fill in any required values (e.g., API keys). The app primarily uses the Gemini API key entered via the sidebar, but you may store other configuration here if needed.

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Enter your Gemini API key in the sidebar

4. Follow the steps in each tab as described above.

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
- `.env.example` / `.env`: Example and actual environment variable files

## Output Files

- `Student_QNA/student_final_verified_qna.json`: Extracted student answers
- `Original_Answer/original_answer.json`: Extracted official answers
- `Final_Evaluation/evaluation_report.txt`: Detailed evaluation report

## Sample Output

**Sample evaluation report excerpt:**
```
--- Question 1 ---
Status: Answered
Score: 90%
Justification: The student correctly explained the concept but missed one minor detail mentioned in the official answer.
👤 Student's Answer:
[Student's answer text]
📚 Official Answer:
[Official answer text]
----------------------------------------------------------------------
```

**Sample student Q&A JSON object:**
```json
{
  "question_number": "12",
  "question_text": "What is Barter System?",
  "answer_text": "Bartering is the direct exchange of one goods with another goods without the use of money...",
  "status": "Answered"
}
```

## Troubleshooting

- **API errors**: Double-check your Gemini API key and internet connection.
- **File not found errors**: Make sure you have completed each step in order and that the required files are present in the correct folders.
- **Python version**: Use Python 3.8 or higher for best compatibility.

## Notes

- The application uses Google's Gemini AI for text extraction and evaluation
- For best results, ensure the PDF scans are clear and legible
- The evaluation is based on semantic understanding, not just keyword matching
- All dependencies are listed in `requirements.txt`