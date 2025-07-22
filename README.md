# Paper Checker Bot

An intelligent system for processing, analyzing, and evaluating student answer papers using AI. The system extracts text from handwritten answers, compares them with official answer keys, and provides automated evaluation with detailed scoring. This version features a complete pipeline from answer extraction to final evaluation.

## Features

- Complete evaluation pipeline (extraction → comparison → scoring)
- User-friendly Streamlit web interface
- PDF to Image conversion for processing
- Advanced image preprocessing for better OCR
- AI-powered text extraction using Google's Gemini Pro Vision
- Automatic answer detection and evaluation
- Multi-language support (English/Hindi answers)
- Exact text preservation (including spelling mistakes and formatting)
- Support for both handwritten and printed text
- Multiple JSON outputs (student answers, official answers, evaluation)
- Detailed evaluation report generation
- Real-time progress tracking

## Prerequisites

### Required Python Packages

```bash
streamlit>=1.32.0
google-generativeai>=0.3.0
pdf2image>=1.16.3
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
```

### Environment Setup

1. Google Gemini API Key
   - Get your API key from Google AI Studio
   - Set it as an environment variable:

     ```bash
     set GEMINI_API_KEY=your_api_key_here  # Windows
     export GEMINI_API_KEY=your_api_key_here  # Linux/Mac
     ```

2. Poppler Installation (for PDF processing)
   - Windows: Download and install poppler from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)
   - Linux: `sudo apt-get install poppler-utils`
   - Mac: `brew install poppler`

## Project Structure

```plaintext
#final_bot/
├── app.py                # Main Streamlit application
├── bot.ipynb            # Development notebook (for reference)
├── preprocessed_pages/  # Directory for preprocessed answer sheet images
├── pdf_pages/          # Directory for raw PDF pages as images
├── Student_QNA/        # Directory for student answers in JSON format
├── Original_Answer/    # Directory for official answer key in JSON format
└── Final_Evaluation/   # Directory for evaluation reports
```

## Usage

1. **Initial Setup**
   - Install required packages:

     ```bash
     pip install -r requirements.txt
     ```

   - Set up your Gemini API key in environment variables
   - Ensure Poppler is installed for PDF processing

2. **Running the Application**

   ```bash
   streamlit run app.py
   ```

3. **Using the Web Interface**
   - Open the provided URL in your browser
   - Upload a student's answer paper PDF
   - Wait for automatic processing

   The application will automatically:
   
   a. **PDF Processing**
   - Convert PDF to images
   - Save individual pages
   
   b. **Image Preprocessing**
   - Apply advanced preprocessing for better text extraction
   - Handle handwritten text optimization
   
   c. **Text Extraction & Analysis**
   - Use Gemini AI for accurate text extraction
   - Process official answer key
   - Compare student answers with official answers
   
   d. **Evaluation**
   - Score each answer automatically
   - Generate detailed justifications
   - Calculate overall performance
   
   e. **Results Display**
   - Show answers and scores in the web interface
   - Provide downloadable JSON outputs
   - Generate evaluation reports
   
   f. **Progress Tracking**
   - Display real-time processing status
   - Show progress bars for each step

## Output Formats

The system generates multiple JSON files for different purposes:

### 1. Student Answers (`student_final_verified_qna.json`)

```json
{
    "question_number": "1",
    "question_text": "What is...",
    "answer_text": "Student's exact answer...",
    "status": "Answered"
}
```

### 2. Official Answers (`original_answer.json`)

```json
{
    "question_number": "1",
    "question_text": "What is Barter System?",
    "official_answer_text": "Official answer in original language (English/Hindi)"
}
```

### 3. Evaluation Report (`evaluation_report.txt`)

The evaluation report includes:
- Detailed analysis of each answer
- Comparison with official answers
- Scores and justifications
- Overall performance summary

## Error Handling

- The system preserves original text exactly as written
- Unclear handwriting is marked with [?]
- Illegible text is marked as [illegible]
- Missing answers are marked as "Not Answered"
- Crossed-out text is ignored

## Best Practices

1. Use high-quality scans of answer sheets
2. Ensure proper PDF formatting
3. Test with a small sample first
4. Verify preprocessed images before full processing
5. Keep original files backed up

## Known Limitations

- Very poor handwriting may affect accuracy
- Heavy document noise can impact preprocessing
- Large PDFs may require more processing time
- Requires stable internet connection for AI processing
- Multi-language answers (English/Hindi) may need manual verification
- Complex mathematical equations or diagrams may need human review
- Performance may vary based on answer complexity

## Troubleshooting

If you encounter issues:

1. Check environment variables are set correctly
2. Verify all required packages are installed
3. Ensure PDF files are readable and not corrupted
4. Check preprocessed images for quality
5. Verify internet connection for AI API access

## License

This project is for educational purposes. Please ensure you have appropriate permissions for using AI services and processing student data.
