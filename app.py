import os
import re
import json
import time
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import google.generativeai as genai

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

# --- Folder and File Paths ---
# Folder containing the preprocessed images of the answer sheets
PREPROCESSED_ANSWERS_FOLDER = "preprocessed_pages"
# Folder for the final output
FINAL_OUTPUT_FOLDER = "Student_QNA"
# The final structured output file
FINAL_JSON_OUTPUT_FILE = os.path.join(FINAL_OUTPUT_FOLDER, "student_final_verified_qna.json")
# Path to the JSON file with official answers
ORIGINAL_ANSWERS_PATH = os.path.join("Original_Answer", "original_answer.json")
# Output folder for the final report
EVALUATION_OUTPUT_FOLDER = "Final_Evaluation"
# Path to the final text report file
EVALUATION_REPORT_FILE = os.path.join(EVALUATION_OUTPUT_FOLDER, "evaluation_report.txt")

# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================

def preprocess_image(image, is_handwritten=True):
    """
    Improved adaptive preprocessing for handwritten images
    with uneven shadows or lighting.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply light Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding (block size and C adjusted)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,  # Invert to make text white on black (for optional cleaning)
        35, 15  # Larger block size + higher constant to adapt to shadows
    )

    # Invert back to black text on white
    adaptive = cv2.bitwise_not(adaptive)

    # Light morphological close to connect letters (optional)
    if is_handwritten:
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        return processed

    return adaptive

# ==============================================================================
# SECTION 3: CORE LOGIC - STUDENT ANSWER EXTRACTION
# ==============================================================================

# The "SINGLE-PASS" MASTER PROMPT
MASTER_QNA_PROMPT = """
You are an expert examination evaluator AI. Your task is to accurately match questions from a provided question paper with their corresponding handwritten answers from a set of images and generate a single, structured JSON output.

You will be given:
1.  A PDF file containing all the exam questions (`question_paper.pdf`).
2.  A series of images (`answer_page_X.png`) containing the student's handwritten answers.

**Your process must be:**

1.  **Iterate Through Questions:** Systematically go through the question paper, question by question, from start to finish.
2.  **Locate the Answer:** For each question (e.g., "Question 6"), locate the student's corresponding handwritten answer in the provided images. Use question numbers (e.g., "Ans-6", "Question-6") as your primary guide.
3.  **Transcribe Accurately:** Transcribe the handwritten answer text *exactly* as it appears, preserving all original wording, spelling, and grammar. Do not correct anything.
4.  **Handle Unanswered Questions:** If you cannot find a corresponding answer for a question, you MUST explicitly mark it as "Not Answered". Do not skip it.
5.  **Ignore Crossed-Out Text:** If any text is visibly struck through or crossed out, ignore it completely and do not include it in the transcription.

**MANDATORY OUTPUT FORMAT:**

-   Your final output must be a single, well-formed JSON array `[...]`.
-   Each element in the array must be a JSON object `{...}` representing one question.
-   Each object must contain these four keys:
    -   `"question_number"`: The specific number of the question (e.g., "1 (i)", "6", "12").
    -   `"question_text"`: The full text of the question.
    -   `"answer_text"`: The student's transcribed answer. If not answered, this must be the string "Not Answered".
    -   `"status"`: A status string, either "Answered" or "Not Answered".

**Example JSON Object:**
{
    "question_number": "12",
    "question_text": "What is Barter System?",
    "answer_text": "Bartering is the direct exchange of one goods with another goods without the use of money For eg for the services of Carpenter or blacksmith of he is given quintal of wheat then it is bartering.",
    "status": "Answered"
}

Do not add any text, notes, or explanations outside of the final JSON array.
"""

def generate_structured_qna(question_pdf_path, answer_image_paths, model):
    """
    Performs the main Q&A matching and verification in a single AI call.
    """
    with st.spinner("Preparing files for AI processing..."):
        try:
            # Prepare the list of files to send to the AI
            # The first item is the master prompt
            prompt_parts = [MASTER_QNA_PROMPT]

            # Add the question paper PDF
            st.write(f"Uploading question paper: {os.path.basename(question_pdf_path)}")
            prompt_parts.append(genai.upload_file(path=question_pdf_path, display_name="question_paper.pdf"))

            # Add all the answer page images
            for i, img_path in enumerate(answer_image_paths):
                st.write(f"Uploading answer page: {os.path.basename(img_path)}")
                prompt_parts.append(genai.upload_file(path=img_path, display_name=f"answer_page_{i+1}.png"))

        except Exception as e:
            st.error(f"Failed to upload files for processing. Error: {e}")
            return None

    with st.spinner("Executing single-pass AI evaluation... This may take a few moments as the AI processes all documents at once..."):
        try:
            # Make the single, powerful API call
            response = model.generate_content(prompt_parts)

            # Extract the JSON from the response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.text)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                st.warning("Could not find a JSON block in the AI's response. Trying to parse the whole text.")
                # Fallback for when the AI forgets the markdown block
                return json.loads(response.text)

        except json.JSONDecodeError:
            st.error("Failed to parse JSON from the AI response. The response may be malformed.")
            st.text("--- AI Response Text ---")
            st.text(response.text)
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during AI generation: {e}")
            return None

# ==============================================================================
# SECTION 4: CORE LOGIC - OFFICIAL ANSWER EXTRACTION
# ==============================================================================

# THE MASTER PROMPT (***RE-ENGINEERED FOR COMPLETENESS***)
MASTER_PROMPT = """
You are a meticulous and highly precise data extraction AI. Your primary directive is to create a **COMPLETE** and **VERBATIM** JSON representation of the provided question paper and its official answers. You *must* process every question from start to finish without any omissions.

You will be given:
1.  `question_paper.pdf`: Contains exam questions.
2.  `official_answer_key.pdf`: Contains the official answers.

**CRITICAL RULES FOR EXECUTION:**

1.  ***ABSOLUTE COMPLETENESS & VERIFICATION***:
    -   You **MUST** process **ALL 23 questions** from the `question_paper.pdf`, from Question 1 to Question 23.
    -   Do not stop early under any circumstances.
    -   **Before finishing, you must perform a final self-check to ensure all 23 questions and their sub-parts are present in your final JSON output.**

2.  ***VERBATIM (EXACT) EXTRACTION***:
    -   All extracted text must be a *character-for-character copy*.
    -   Do not translate, summarize, rephrase, or alter any text.
    -   The original languages in the source documents (e.g., English and Hindi) must be preserved perfectly.

3.  ***QUESTION TEXT SPECIFICATIONS***:
    -   For each question number, extract the full **English version** of the question text.
    -   If a question has multiple-choice options (e.g., A, B, C, D), you **MUST** include those options as part of the `question_text`.

4.  ***"OR" (अथवा) QUESTION HANDLING***:
    -   Many questions have an 'OR' (अथवा) option. For these, your `question_text` **MUST** include the text for **BOTH** the main question and the 'OR' question.
    -   The `official_answer_text` should be the single answer provided in the answer key, which may correspond to either the main question or the 'OR' part. Extract whichever answer is present.

**MANDATORY OUTPUT FORMAT:**

-   Your final output must be a single, well-formed JSON array `[...]`.
-   Each object must contain these three keys:
    -   `"question_number"`: The specific number (e.g., "1 (i)", "6", "23").
    -   `"question_text"`: The full English text of the question (and its 'OR' part, if present), copied verbatim.
    -   `"official_answer_text"`: The full text of the corresponding answer from the answer key, copied verbatim in its original language.

**Example for a Multiple-Choice Question:**
{
    "question_number": "1 (i)",
    "question_text": "Study of a firm, an industries, price of a good is done under -\n(A) Macro economics\n(B) Micro economics\n(C) Both (A and B)\n(D) National income",
    "official_answer_text": "(ब) व्यष्टि अर्थशास्त्र"
}

**Example for a Question with an "OR" part:**
{
    "question_number": "6",
    "question_text": "What are the central problems of an economy?\nOR\nDefine Economic Problem.",
    "official_answer_text": "(i) किन वस्तुओं का कितनी मात्रा में उत्पादन किया जाये। (ii) वस्तुओं का उत्पादन किसके लिऐ किया जाये। (iii) वस्तुओं के उत्पादन का ढंग क्या हो?"
}

Do not add any text, notes, or explanations outside of the final JSON array.
"""

def generate_official_qna_json(question_pdf, answer_pdf, model):
    """
    Uses AI to extract and combine questions and official answers.
    """
    with st.spinner("Preparing files for AI processing..."):
        try:
            # List of prompt and files to send to the AI
            prompt_parts = [
                MASTER_PROMPT,
                genai.upload_file(path=question_pdf, display_name="question_paper.pdf"),
                genai.upload_file(path=answer_pdf, display_name="official_answer_key.pdf")
            ]
        except Exception as e:
            st.error(f"Failed to upload files. Error: {e}")
            return None

    with st.spinner("Starting AI extraction... This may take a moment..."):
        try:
            # Send the request to the AI
            response = model.generate_content(prompt_parts)

            # Extract the JSON from the response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.text)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                st.warning("No JSON block found in AI response. Attempting to parse the full text.")
                return json.loads(response.text)

        except json.JSONDecodeError:
            st.error("Failed to parse JSON from AI response. The response may be malformed.")
            st.text("--- AI Response Text ---")
            st.text(response.text)
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during AI generation: {e}")
            return None

# ==============================================================================
# SECTION 5: CORE LOGIC - EVALUATION
# ==============================================================================

# AI EVALUATION PROMPT
EVALUATION_PROMPT_TEMPLATE = """
You are an expert, impartial examiner. Your task is to evaluate a student's answer against the official model answer and provide a score based on semantic correctness.

**Context:**
- The student's answer and the official answer may be in different languages (e.g., English and Hindi).
- Your evaluation must be based on the *meaning and core concepts*, not just keyword matching.

**Official Answer:**
---
{official_answer}
---

**Student's Answer:**
---
{student_answer}
---

**Your Task:**
1.  Compare the student's answer to the official answer.
2.  Provide a numerical score from 0 to 100, where 100 means the student's answer perfectly conveys the same meaning as the official answer, and 0 means it is completely incorrect or irrelevant.
3.  Provide a brief, one-sentence justification for your score.

**Mandatory Output Format:**
You MUST return your response in a single line with the format: `score|justification`
**Example 1:** `90|The student correctly explained the concept but missed one minor detail mentioned in the official answer.`
**Example 2:** `0|The student's answer is factually incorrect and does not align with the official answer.`
**Example 3:** `100|The student's answer perfectly matches the core concepts of the official answer.`
"""

def load_and_merge_data(original_file, student_file):
    """Loads data from both JSON files and merges them for comparison."""
    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = {item['question_number']: item for item in json.load(f)}
    except FileNotFoundError:
        st.error(f"Official answer file not found at '{original_file}'")
        return None
        
    try:
        with open(student_file, 'r', encoding='utf-8') as f:
            student_data = {item['question_number']: item for item in json.load(f)}
    except FileNotFoundError:
        st.error(f"Student answer file not found at '{student_file}'")
        return None

    merged_data = []
    for q_num, original_q in original_data.items():
        student_q = student_data.get(q_num)
        if student_q:
            merged_data.append({
                "question_number": q_num,
                "question_text": original_q.get("question_text", "N/A"),
                "official_answer": original_q.get("official_answer_text", ""),
                "student_answer": student_q.get("answer_text", ""),
                "status": student_q.get("status", "Not Answered")
            })
    return merged_data

def evaluate_answers(data_to_evaluate, evaluation_model):
    """Iterates through data and uses AI to score each answered question."""
    evaluated_results = []
    total_questions = len(data_to_evaluate)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, item in enumerate(data_to_evaluate):
        status_text.write(f"Evaluating question {i+1}/{total_questions}: '{item['question_number']}'...")
        if item["status"] != "Answered" or not item["student_answer"] or item["student_answer"] == "Not Answered":
            item["score"] = 0
            item["justification"] = "Question was not answered by the student."
            status_text.write(f"Evaluating question {i+1}/{total_questions}: '{item['question_number']}'... SKIPPED (Not Answered)")
        else:
            prompt = EVALUATION_PROMPT_TEMPLATE.format(
                official_answer=item["official_answer"],
                student_answer=item["student_answer"]
            )
            try:
                response = evaluation_model.generate_content(prompt)
                # Parse the response: score|justification
                parts = response.text.strip().split('|')
                if len(parts) == 2:
                    item["score"] = int(re.sub(r'[^0-9]', '', parts[0])) # Clean non-numeric chars
                    item["justification"] = parts[1].strip()
                    status_text.write(f"Evaluating question {i+1}/{total_questions}: '{item['question_number']}'... SCORE: {item['score']}%")
                else:
                    raise ValueError("Response was not in the expected 'score|justification' format.")
            except Exception as e:
                item["score"] = -1  # Use -1 to indicate an evaluation error
                item["justification"] = f"AI evaluation failed: {e}"
                status_text.write(f"Evaluating question {i+1}/{total_questions}: '{item['question_number']}'... FAILED ({e})")

        evaluated_results.append(item)
        progress_bar.progress((i + 1) / total_questions)
    
    return evaluated_results

def generate_evaluation_report(results):
    """Generates a detailed text file report from the evaluation results."""
    os.makedirs(EVALUATION_OUTPUT_FOLDER, exist_ok=True)
    
    total_score = 0
    answered_count = 0
    
    with open(EVALUATION_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("======================================================================\n")
        f.write("                 AUTOMATED ANSWER EVALUATION REPORT\n")
        f.write("======================================================================\n\n")

        for item in results:
            f.write(f"--- Question {item['question_number']} ---\n")
            f.write(f"Status: {item['status']}\n")
            f.write(f"Score: {item['score']}%\n")
            f.write(f"Justification: {item['justification']}\n\n")
            f.write("👤 Student's Answer:\n")
            f.write(f"{item['student_answer']}\n\n")
            f.write("📚 Official Answer:\n")
            f.write(f"{item['official_answer']}\n")
            f.write("----------------------------------------------------------------------\n\n")
            
            if item['status'] == 'Answered' and item['score'] != -1:
                total_score += item['score']
                answered_count += 1
                
        # Final Summary
        average_score = (total_score / answered_count) if answered_count > 0 else 0
        f.write("======================================================================\n")
        f.write("                            FINAL SUMMARY\n")
        f.write("======================================================================\n")
        f.write(f"Total Questions: {len(results)}\n")
        f.write(f"Questions Answered by Student: {answered_count}\n")
        f.write(f"Average Score on Answered Questions: {average_score:.2f}%\n")
        f.write("======================================================================\n")
    
    st.success(f"Evaluation report saved to: {EVALUATION_REPORT_FILE}")
    return average_score, answered_count, len(results)

# ==============================================================================
# SECTION 6: STREAMLIT APP
# ==============================================================================

def main():
    st.set_page_config(page_title="Paper Checker Bot", page_icon="📝", layout="wide")
    
    st.title("📝 Paper Checker Bot")
    st.markdown("---")
    
    # Sidebar for API key input and configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your Gemini API Key", type="password")
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-pro-latest")
                st.success("✅ Gemini AI configured successfully")
            except Exception as e:
                st.error(f"❌ Failed to configure Gemini. Error: {e}")
                model = None
        else:
            st.warning("Please enter your Gemini API Key to proceed")
            model = None
    
    # Main app tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["All-in-One Process", "PDF Processing", "Student Answer Extraction", "Official Answer Extraction", "Evaluation"])
    
    # Tab 1: All-in-One Process
    with tab1:
        st.header("All-in-One Process")
        st.write("Upload all required PDFs at once and process them in a single workflow.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            student_answer_pdf = st.file_uploader("Upload Student Answer PDF", type="pdf", key="all_in_one_student_pdf")
        
        with col2:
            question_pdf = st.file_uploader("Upload Question Paper PDF", type="pdf", key="all_in_one_question_pdf")
        
        with col3:
            official_answer_pdf = st.file_uploader("Upload Official Answer Key PDF", type="pdf", key="all_in_one_answer_pdf")
        
        if student_answer_pdf is not None and question_pdf is not None and official_answer_pdf is not None:
            if st.button("Process All Files") and model is not None:
                # Step 1: Process Student Answer PDF
                with st.spinner("Step 1/4: Converting Student Answer PDF to images..."):
                    # Save the uploaded PDF temporarily
                    student_pdf_path = os.path.join(os.getcwd(), student_answer_pdf.name)
                    with open(student_pdf_path, "wb") as f:
                        f.write(student_answer_pdf.getbuffer())
                    
                    # Convert PDF to images
                    images = convert_from_path(student_pdf_path)
                    st.write(f"Extracted {len(images)} pages from the student answer PDF.")
                    
                    # Create output directories if they don't exist
                    os.makedirs("pdf_pages", exist_ok=True)
                    os.makedirs("preprocessed_pages", exist_ok=True)
                    
                    # Process each image
                    for i, img in enumerate(images):
                        # Save original image
                        img_path = os.path.join("pdf_pages", f"page_{i+1}.jpg")
                        img.save(img_path, "JPEG")
                        
                        # Preprocess image
                        cv_img = np.array(img)
                        cv_img = cv_img[:, :, ::-1].copy()  # RGB to BGR
                        processed = preprocess_image(cv_img, is_handwritten=True)
                        
                        # Save preprocessed image
                        pil_img = Image.fromarray(processed)
                        processed_path = os.path.join("preprocessed_pages", f"page_{i+1}.png")
                        pil_img.save(processed_path)
                    
                    st.success(f"Step 1/4 Complete: Successfully processed {len(images)} pages.")
                
                # Step 2: Save Question PDF
                with st.spinner("Step 2/4: Preparing question paper..."):
                    # Save the uploaded question PDF temporarily
                    question_pdf_path = os.path.join(os.getcwd(), question_pdf.name)
                    with open(question_pdf_path, "wb") as f:
                        f.write(question_pdf.getbuffer())
                    
                    # Save the uploaded official answer PDF temporarily
                    official_answer_pdf_path = os.path.join(os.getcwd(), official_answer_pdf.name)
                    with open(official_answer_pdf_path, "wb") as f:
                        f.write(official_answer_pdf.getbuffer())
                    
                    st.success("Step 2/4 Complete: Question paper and official answer key prepared.")
                
                # Step 3: Extract Student Answers
                with st.spinner("Step 3/4: Extracting student answers..."):
                    # Get a sorted list of all answer image files
                    answer_files = sorted(
                        [os.path.join(PREPROCESSED_ANSWERS_FOLDER, f) for f in os.listdir(PREPROCESSED_ANSWERS_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
                        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
                    )
                    
                    if not answer_files:
                        st.error(f"No image files found in '{PREPROCESSED_ANSWERS_FOLDER}'.")
                    else:
                        # Create output directory if it doesn't exist
                        os.makedirs(FINAL_OUTPUT_FOLDER, exist_ok=True)
                        
                        # Generate the final structured data
                        final_qna_data = generate_structured_qna(question_pdf_path, answer_files, model)
                        
                        if final_qna_data and isinstance(final_qna_data, list):
                            with open(FINAL_JSON_OUTPUT_FILE, "w", encoding="utf-8") as f:
                                # Save as a pretty-printed JSON file, which is ideal for machine processing
                                json.dump(final_qna_data, f, indent=4, ensure_ascii=False)
                            st.success(f"Step 3/4 Complete: Extracted {len(final_qna_data)} student question-answer pairs.")
                        else:
                            st.error("No valid student Q&A data was generated. The output file was not created.")
                            return
                
                # Step 4: Extract Official Answers
                with st.spinner("Step 4/4: Extracting official answers..."):
                    # Create output directory if it doesn't exist
                    os.makedirs("Original_Answer", exist_ok=True)
                    
                    # Generate the final structured data
                    final_official_qna_data = generate_official_qna_json(question_pdf_path, official_answer_pdf_path, model)
                    
                    if final_official_qna_data and isinstance(final_official_qna_data, list):
                        with open(ORIGINAL_ANSWERS_PATH, "w", encoding="utf-8") as f:
                            json.dump(final_official_qna_data, f, indent=4, ensure_ascii=False)
                        st.success(f"Step 4/4 Complete: Extracted {len(final_official_qna_data)} official question-answer pairs.")
                    else:
                        st.error("No valid official Q&A data was generated. The output file was not created.")
                        return
                
                # Step 5: Evaluate Answers
                with st.spinner("Final Step: Evaluating answers..."):
                    # Load and merge data from both JSON files
                    merged_data = load_and_merge_data(ORIGINAL_ANSWERS_PATH, FINAL_JSON_OUTPUT_FILE)
                    
                    if merged_data:
                        # Evaluate the answers using the AI model
                        st.subheader("Automated Evaluation Progress")
                        final_results = evaluate_answers(merged_data, model)
                        
                        # Generate a detailed report file
                        st.subheader("Generating Evaluation Report")
                        avg_score, answered_count, total_questions = generate_evaluation_report(final_results)
                        
                        # Display summary statistics
                        st.subheader("Evaluation Summary")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Questions", total_questions)
                        col2.metric("Questions Answered", answered_count)
                        col3.metric("Average Score", f"{avg_score:.2f}%")
                        
                        # Display the report content
                        st.subheader("Evaluation Report")
                        with open(EVALUATION_REPORT_FILE, 'r', encoding='utf-8') as f:
                            report_content = f.read()
                        st.text_area("Report Content", report_content, height=400)
                        
                        # Provide a download button for the report
                        with open(EVALUATION_REPORT_FILE, 'rb') as f:
                            st.download_button(
                                label="Download Evaluation Report",
                                data=f,
                                file_name="evaluation_report.txt",
                                mime="text/plain"
                            )
                    else:
                        st.error("Could not proceed due to errors in loading data.")
            else:
                if model is None:
                    st.warning("Please configure the Gemini API key in the sidebar first.")
        else:
            st.info("Please upload all three PDF files to proceed with the all-in-one process.")
    
    # Tab 2: PDF Processing
    with tab2:
        st.header("Step 2: Process PDF Answer Sheets")
        st.write("Upload a PDF file containing the student's answer sheets to process and convert to images.")
        
        uploaded_pdf = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")
        
        if uploaded_pdf is not None:
            # Save the uploaded PDF temporarily
            pdf_path = os.path.join(os.getcwd(), uploaded_pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            
            if st.button("Process PDF"):
                with st.spinner("Converting PDF to images..."):
                    # Convert PDF to images
                    images = convert_from_path(pdf_path)
                    st.write(f"Extracted {len(images)} pages from the PDF.")
                    
                    # Create output directories if they don't exist
                    os.makedirs("pdf_pages", exist_ok=True)
                    os.makedirs("preprocessed_pages", exist_ok=True)
                    
                    # Process each image
                    for i, img in enumerate(images):
                        # Save original image
                        img_path = os.path.join("pdf_pages", f"page_{i+1}.jpg")
                        img.save(img_path, "JPEG")
                        
                        # Preprocess image
                        cv_img = np.array(img)
                        cv_img = cv_img[:, :, ::-1].copy()  # RGB to BGR
                        processed = preprocess_image(cv_img, is_handwritten=True)
                        
                        # Save preprocessed image
                        pil_img = Image.fromarray(processed)
                        processed_path = os.path.join("preprocessed_pages", f"page_{i+1}.png")
                        pil_img.save(processed_path)
                    
                    st.success(f"Successfully processed {len(images)} pages. Original images saved to 'pdf_pages' folder and preprocessed images saved to 'preprocessed_pages' folder.")
                    
                    # Display a few sample images
                    st.subheader("Sample Processed Images")
                    cols = st.columns(3)
                    for i, col in enumerate(cols):
                        if i < len(images):
                            processed_path = os.path.join("preprocessed_pages", f"page_{i+1}.png")
                            col.image(processed_path, caption=f"Page {i+1}", use_column_width=True)
    
    # Tab 3: Student Answer Extraction
    with tab3:
        st.header("Step 3: Extract Student Answers")
        st.write("Extract student answers from the processed images using AI.")
        
        question_pdf = st.file_uploader("Upload Question Paper PDF", type="pdf", key="question_pdf_uploader")
        
        if question_pdf is not None:
            # Save the uploaded question PDF temporarily
            question_pdf_path = os.path.join(os.getcwd(), question_pdf.name)
            with open(question_pdf_path, "wb") as f:
                f.write(question_pdf.getbuffer())
            
            if os.path.exists(PREPROCESSED_ANSWERS_FOLDER) and os.listdir(PREPROCESSED_ANSWERS_FOLDER):
                st.write(f"Found {len(os.listdir(PREPROCESSED_ANSWERS_FOLDER))} preprocessed answer pages.")
                
                if st.button("Extract Student Answers") and model is not None:
                    # Get a sorted list of all answer image files
                    answer_files = sorted(
                        [os.path.join(PREPROCESSED_ANSWERS_FOLDER, f) for f in os.listdir(PREPROCESSED_ANSWERS_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
                        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
                    )
                    
                    if not answer_files:
                        st.error(f"No image files found in '{PREPROCESSED_ANSWERS_FOLDER}'.")
                    else:
                        # Create output directory if it doesn't exist
                        os.makedirs(FINAL_OUTPUT_FOLDER, exist_ok=True)
                        
                        # Generate the final structured data
                        final_qna_data = generate_structured_qna(question_pdf_path, answer_files, model)
                        
                        if final_qna_data and isinstance(final_qna_data, list):
                            with open(FINAL_JSON_OUTPUT_FILE, "w", encoding="utf-8") as f:
                                # Save as a pretty-printed JSON file, which is ideal for machine processing
                                json.dump(final_qna_data, f, indent=4, ensure_ascii=False)
                            st.success(f"Final file created with {len(final_qna_data)} question-answer pairs. Output saved to: {FINAL_JSON_OUTPUT_FILE}")
                            
                            # Display a sample of the extracted data
                            st.subheader("Sample Extracted Data")
                            st.json(final_qna_data[:3])
                        else:
                            st.error("No valid Q&A data was generated. The output file was not created.")
                else:
                    if model is None:
                        st.warning("Please configure the Gemini API key in the sidebar first.")
            else:
                st.warning(f"No preprocessed answer pages found in '{PREPROCESSED_ANSWERS_FOLDER}'. Please process the PDF first in the 'PDF Processing' tab.")
    
    # Tab 4: Official Answer Extraction
    with tab4:
        st.header("Step 4: Extract Official Answers")
        st.write("Extract official answers from the answer key PDF using AI.")
        
        question_pdf = st.file_uploader("Upload Question Paper PDF", type="pdf", key="official_question_pdf_uploader")
        answer_pdf = st.file_uploader("Upload Official Answer Key PDF", type="pdf", key="official_answer_pdf_uploader")
        
        if question_pdf is not None and answer_pdf is not None:
            # Save the uploaded PDFs temporarily
            question_pdf_path = os.path.join(os.getcwd(), question_pdf.name)
            with open(question_pdf_path, "wb") as f:
                f.write(question_pdf.getbuffer())
                
            answer_pdf_path = os.path.join(os.getcwd(), answer_pdf.name)
            with open(answer_pdf_path, "wb") as f:
                f.write(answer_pdf.getbuffer())
            
            if st.button("Extract Official Answers") and model is not None:
                # Create output directory if it doesn't exist
                os.makedirs("Original_Answer", exist_ok=True)
                
                # Generate the final structured data
                final_qna_data = generate_official_qna_json(question_pdf_path, answer_pdf_path, model)
                
                if final_qna_data and isinstance(final_qna_data, list):
                    with open(ORIGINAL_ANSWERS_PATH, "w", encoding="utf-8") as f:
                        json.dump(final_qna_data, f, indent=4, ensure_ascii=False)
                    st.success(f"Final file created with {len(final_qna_data)} question-answer pairs. Output saved to: {ORIGINAL_ANSWERS_PATH}")
                    
                    # Display a sample of the extracted data
                    st.subheader("Sample Extracted Data")
                    st.json(final_qna_data[:3])
                else:
                    st.error("No valid Q&A data was generated. The output file was not created.")
            else:
                if model is None:
                    st.warning("Please configure the Gemini API key in the sidebar first.")
    
    # Tab 5: Evaluation
    with tab5:
        st.header("Step 5: Evaluate Student Answers")
        st.write("Compare student answers with official answers and generate an evaluation report.")
        
        if os.path.exists(ORIGINAL_ANSWERS_PATH) and os.path.exists(FINAL_JSON_OUTPUT_FILE):
            st.write("✅ Both student answers and official answers files are available.")
            
            if st.button("Start Evaluation") and model is not None:
                # Create output directory if it doesn't exist
                os.makedirs(EVALUATION_OUTPUT_FOLDER, exist_ok=True)
                
                # Load and merge data from both JSON files
                with st.spinner("Loading and merging data..."):
                    merged_data = load_and_merge_data(ORIGINAL_ANSWERS_PATH, FINAL_JSON_OUTPUT_FILE)
                
                if merged_data:
                    # Evaluate the answers using the AI model
                    st.subheader("Automated Evaluation Progress")
                    final_results = evaluate_answers(merged_data, model)
                    
                    # Generate a detailed report file
                    st.subheader("Generating Evaluation Report")
                    avg_score, answered_count, total_questions = generate_evaluation_report(final_results)
                    
                    # Display summary statistics
                    st.subheader("Evaluation Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Questions", total_questions)
                    col2.metric("Questions Answered", answered_count)
                    col3.metric("Average Score", f"{avg_score:.2f}%")
                    
                    # Display the report content
                    st.subheader("Evaluation Report")
                    with open(EVALUATION_REPORT_FILE, 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    st.text_area("Report Content", report_content, height=400)
                    
                    # Provide a download button for the report
                    with open(EVALUATION_REPORT_FILE, 'rb') as f:
                        st.download_button(
                            label="Download Evaluation Report",
                            data=f,
                            file_name="evaluation_report.txt",
                            mime="text/plain"
                        )
                else:
                    st.error("Could not proceed due to errors in loading data.")
            else:
                if model is None:
                    st.warning("Please configure the Gemini API key in the sidebar first.")
        else:
            missing_files = []
            if not os.path.exists(ORIGINAL_ANSWERS_PATH):
                missing_files.append("Official answers file")
            if not os.path.exists(FINAL_JSON_OUTPUT_FILE):
                missing_files.append("Student answers file")
            
            st.warning(f"Missing required files: {', '.join(missing_files)}. Please complete the previous steps first.")

if __name__ == "__main__":
    main()