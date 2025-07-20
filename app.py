import streamlit as st
import os
import re
import json
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import shutil

# Load environment variables from .env file
load_dotenv()

# --- IMPROVED Prompts for higher accuracy ---
MASTER_QNA_PROMPT_STUDENT = """
You are an expert examination evaluator AI, specialized in accurately transcribing handwritten answers. Your critical task is to match questions from a provided question paper with their corresponding handwritten answers from a set of images and generate a single, structured JSON output.

You will be given:
1.  A PDF file containing all the exam questions (`question_paper.pdf`).
2.  A series of images (`answer_page_X.png`) containing the student's handwritten answers.

**CRITICAL RULES FOR TRANSCRIPTION:**

1.  **Systematic Processing:** You MUST process the question paper systematically, from the first question to the last. For each question (e.g., "Question 6"), locate the corresponding answer in the provided images using question numbers (e.g., "Ans-6", "Question-6") as your primary guide.

2.  **Verbatim Transcription with Self-Correction:** Transcribe the handwritten answer text *exactly* as it appears, preserving all original wording, spelling, and grammar. You MUST double-check your transcription for common character confusion (e.g., 'u' vs 'v', 'a' vs 'o'), especially for critical single-word answers where a small mistake changes the entire meaning. Do not correct any of the student's mistakes.

3.  **Handling Special Cases:**
    * **Unanswered Questions:** If you cannot find an answer for a question, you MUST set the `answer_text` to the string "Not Answered" and the `status` to "Not Answered".
    * **Crossed-Out Text:** If any text is visibly struck through or crossed out, you MUST ignore it completely.
    * **Matching Questions:** For "Match the columns" questions (e.g., Question 4), you MUST break down the answer into sub-parts. Create separate JSON objects for "4 (i)", "4 (ii)", etc. The `question_text` should be the item from the first column (e.g., "Demand schedule"), and the `answer_text` must be the full text of the matched item.

**MANDATORY OUTPUT FORMAT:**

-   Your final output must be a single, well-formed JSON array `[...]`.
-   Each element in the array must be a JSON object `{...}` representing one question.
-   Each object must contain these four keys:
    -   `"question_number"`: The specific number of the question (e.g., "1 (i)", "4 (i)", "12").
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

**Final Instruction:** Your entire response must ONLY be the single, raw JSON array. Do not include any introductory text, explanations, notes, or markdown formatting like ` ```json ` in your final output.
"""

MASTER_QNA_PROMPT_OFFICIAL = """
You are a meticulous and highly precise data extraction AI. Your primary directive is to create a **COMPLETE** and **VERBATIM** JSON representation of the provided question paper and its official answers.

You will be given:
1.  `question_paper.pdf`: Contains exam questions.
2.  `official_answer_key.pdf`: Contains the official answers.

**CRITICAL RULES FOR EXECUTION:**

1.  ***ABSOLUTE COMPLETENESS & VERIFICATION***:
    -   You **MUST** process **ALL 23 questions** from the `question_paper.pdf`.
    -   Before finishing, you must perform a final self-check to ensure all 23 questions and their sub-parts are present in your final JSON output.

2.  ***VERBATIM (EXACT) EXTRACTION***:
    -   All extracted text must be a *character-for-character copy*. Do not translate, summarize, or alter any text.

3.  ***"OR" (अथवा) QUESTION HANDLING***:
    -   For questions with an 'OR' option, the `question_text` **MUST** include the text for **BOTH** the main question and the 'OR' question.

4.  ***Special Instruction for Matching Questions***:
    -   For "Match the columns" questions (like Question 4), you **MUST** break it down into sub-parts. Create a separate JSON object for each matched pair, using question numbers like `4 (i)`, `4 (ii)`, etc. The `question_text` for each object should be the full item from the first column of the question paper (e.g., "(i) Demand schedule"), and the `official_answer_text` should be the corresponding matched pair's text from the answer key.

**MANDATORY OUTPUT FORMAT:**

-   Your final output must be a single, well-formed JSON array `[...]`.
-   Each object must contain these three keys:
    -   `"question_number"`: The specific number (e.g., "1 (i)", "4 (i)", "23").
    -   `"question_text"`: The full English text of the question (and its 'OR' part, if present), copied verbatim.
    -   `"official_answer_text"`: The full text of the corresponding answer from the answer key, copied verbatim.

Do not add any text, notes, or explanations outside of the final JSON array.
"""

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
2.  Provide a numerical score from 0 to 100.
3.  Provide a brief, one-sentence justification for your score.

**Mandatory Output Format:**
You MUST return your response in a single line with the format: `score|justification`
**Example:** `90|The student correctly explained the concept but missed one minor detail mentioned in the official answer.`
"""


# --- All Helper Functions ---
def preprocess_image(image, is_handwritten=True):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )
    adaptive = cv2.bitwise_not(adaptive)
    if is_handwritten:
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        return processed
    return adaptive

def safe_json_loads(text, step_name):
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    content_to_parse = match.group(1) if match else text
    try:
        start_index = content_to_parse.find('[')
        end_index = content_to_parse.rfind(']')
        if start_index != -1 and end_index != -1:
            json_str = content_to_parse[start_index : end_index + 1]
            return json.loads(json_str)
        else:
            raise json.JSONDecodeError("Could not find JSON array brackets.", content_to_parse, 0)
    except json.JSONDecodeError:
        st.error(f"Error during '{step_name}': The AI returned a response that could not be parsed as JSON, even after cleaning.")
        st.text_area("Malformed AI Response that caused the error:", value=text, height=200)
        return None

def generate_text_report(results, summary):
    report_lines = []
    report_lines.append("======================================================")
    report_lines.append("         AUTOMATED ANSWER EVALUATION REPORT")
    report_lines.append("======================================================")
    report_lines.append("\n--- FINAL SUMMARY ---\n")
    report_lines.append(f"Total Questions: {summary['total_questions']}")
    report_lines.append(f"Items Attempted: {summary['answered_count']}")
    report_lines.append(f"Average Score on Attempted Items: {summary['average_score']:.2f}%\n")
    report_lines.append("======================================================")

    for item in results:
        report_lines.append(f"\n\n--- Question {item.get('question_number', 'N/A')} ---")
        report_lines.append(f"Status: {item.get('status', 'N/A')}")
        report_lines.append(f"Score: {item.get('score', 'N/A')}%")
        report_lines.append(f"Justification: {item.get('justification', 'N/A')}\n")
        report_lines.append("👤 Student's Answer:")
        report_lines.append(f"{item.get('student_answer', 'Not Answered')}\n")
        report_lines.append("📚 Official Answer:")
        report_lines.append(f"{item.get('official_answer', 'N/A')}")
        report_lines.append("------------------------------------------------------")
    return "\n".join(report_lines)


# --- Main App Logic ---
st.set_page_config(layout="wide", page_title="AI Paper Grader")
st.title("📄 Automated Examination Paper Grader")
st.markdown("This app automates the entire paper grading workflow. Upload the required files to begin.")

with st.sidebar:
    st.header("⚙️ Configuration")
    if os.getenv("GEMINI_API_KEY"):
        st.success("✅ Gemini API Key loaded from .env file.")
    else:
        st.error("❗️ Gemini API Key not found. Please create a .env file.")
    st.markdown("---")
    st.info("Files are processed in memory and are not stored on any server.")

st.header("1. Upload Required PDFs")
col1, col2, col3 = st.columns(3)
student_pdf_file = col1.file_uploader("Upload Student's Answer Sheet", type="pdf")
question_pdf_file = col2.file_uploader("Upload Question Paper", type="pdf")
answer_key_pdf_file = col3.file_uploader("Upload Official Answer Key", type="pdf")

if st.button("🚀 Grade Paper", type="primary"):
    if not all([student_pdf_file, question_pdf_file, answer_key_pdf_file]):
        st.warning("Please upload all three PDF files.")
    elif not os.getenv("GEMINI_API_KEY"):
        st.error("Cannot proceed without a Gemini API Key in the .env file.")
    else:
        # Clear previous results before starting a new run
        for key in ['final_results', 'student_qna_data', 'official_qna_data']:
            if key in st.session_state:
                del st.session_state[key]

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        temp_dir = "temp_processing"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            with st.spinner("Grading in progress... This may take several minutes."):
                # Step 1: Preprocess Student's PDF
                st.info("Step 1/5: Converting and cleaning student's answer sheet...")
                student_images_pil = convert_from_bytes(student_pdf_file.getvalue())
                preprocessed_paths = [os.path.join(temp_dir, f"page_{i+1}.png") for i in range(len(student_images_pil))]
                for i, pil_image in enumerate(student_images_pil):
                    cv_img = np.array(pil_image)[:, :, ::-1].copy()
                    processed_cv_img = preprocess_image(cv_img)
                    Image.fromarray(processed_cv_img).save(preprocessed_paths[i])

                # Step 2: Extract Student's Answers
                st.info("Step 2/5: Reading student's answers using AI...")
                temp_question_path = os.path.join(temp_dir, "question.pdf")
                with open(temp_question_path, "wb") as f: f.write(question_pdf_file.getvalue())
                
                parts = [MASTER_QNA_PROMPT_STUDENT, genai.upload_file(path=temp_question_path)]
                for path in preprocessed_paths: parts.append(genai.upload_file(path=path))
                
                response = model.generate_content(parts)
                student_qna_data = safe_json_loads(response.text, "Student Answer Extraction")
                if not student_qna_data: st.stop()
                st.session_state.student_qna_data = student_qna_data

                # Step 3: Extract Official Answers
                st.info("Step 3/5: Reading official answers using AI...")
                temp_answer_key_path = os.path.join(temp_dir, "answer_key.pdf")
                with open(temp_answer_key_path, "wb") as f: f.write(answer_key_pdf_file.getvalue())

                parts = [MASTER_QNA_PROMPT_OFFICIAL, genai.upload_file(path=temp_question_path), genai.upload_file(path=temp_answer_key_path)]
                response = model.generate_content(parts)
                official_qna_data = safe_json_loads(response.text, "Official Answer Extraction")
                if not official_qna_data: st.stop()
                st.session_state.official_qna_data = official_qna_data

                # Step 4: Evaluate Answers
                st.info("Step 4/5: Evaluating answers one by one...")
                official_map = {item['question_number']: item for item in official_qna_data}
                merged_data = []
                for s_item in student_qna_data:
                    q_num = s_item['question_number']
                    if q_num in official_map:
                        merged_data.append({
                            "question_number": q_num,
                            "question_text": official_map[q_num].get("question_text", "N/A"),
                            "official_answer": official_map[q_num].get("official_answer_text", ""),
                            "student_answer": s_item.get("answer_text", ""),
                            "status": s_item.get("status", "Not Answered")
                        })

                evaluated_results = []
                progress_bar = st.progress(0, text="Starting evaluation...")
                for i, item in enumerate(merged_data):
                    progress_bar.progress((i + 1) / len(merged_data), text=f"Evaluating Question {item['question_number']}...")
                    if item["status"] != "Answered" or not item["student_answer"] or item["student_answer"] == "Not Answered":
                        item["score"], item["justification"] = 0, "Question was not answered by the student."
                    else:
                        prompt = EVALUATION_PROMPT_TEMPLATE.format(official_answer=item["official_answer"], student_answer=item["student_answer"])
                        try:
                            response = model.generate_content(prompt)
                            parts = response.text.strip().split('|')
                            if len(parts) == 2:
                                item["score"] = int(re.sub(r'[^0-9]', '', parts[0]))
                                item["justification"] = parts[1].strip()
                            else:
                                item["score"], item["justification"] = -1, "AI response was not in the expected 'score|justification' format."
                        except Exception as e:
                            item["score"], item["justification"] = -1, f"AI evaluation failed: {e}"
                    evaluated_results.append(item)
                
                # Step 5: Finalize and Store Results
                st.info("Step 5/5: Compiling the final report...")
                st.session_state.final_results = evaluated_results

            st.success("✅ Grading complete! View and download the report below.")
        except Exception as e:
            st.error(f"An unexpected error occurred during the workflow: {e}")
        finally:
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

# --- Display Results ---
if 'final_results' in st.session_state:
    results = st.session_state.final_results
    
    unique_question_numbers = set()
    for item in results:
        q_num = item.get('question_number', 'N/A')
        base_q_num = q_num.split(' ')[0].split('(')[0]
        if base_q_num:
            unique_question_numbers.add(base_q_num)
    total_unique_questions = len(unique_question_numbers)

    answered_count = sum(1 for item in results if item['status'] == 'Answered' and item.get('score', -1) >= 0)
    total_score = sum(item.get('score', 0) for item in results if item['status'] == 'Answered' and item.get('score', -1) >= 0)
    average_score = (total_score / answered_count) if answered_count > 0 else 0
    summary = {
        "total_questions": total_unique_questions,
        "answered_count": answered_count,
        "average_score": average_score
    }

    st.header("2. Evaluation Report")
    st.markdown("---")

    st.subheader("📊 Performance Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Questions", f"{summary['total_questions']}")
    m2.metric("Items Attempted", f"{summary['answered_count']}")
    m3.metric("Average Score (on attempted)", f"{summary['average_score']:.2f}%", delta_color="off")

     # --- NEW: Interactive Q&A Section ---
    st.markdown("---")
    st.header("3. 💬 Ask About a Specific Question")
    
    if results:
        question_numbers = [item.get('question_number', 'N/A') for item in results]
        
        selected_q_num = st.selectbox(
            "Select a question number to see the detailed comparison:",
            options=question_numbers,
            index=None,
            placeholder="Choose a question..."
        )

        if selected_q_num:
            # Find the selected question's data from the results
            selected_data = next((item for item in results if item['question_number'] == selected_q_num), None)
            
            if selected_data:
                st.markdown(f"#### Comparison for Question: `{selected_data['question_number']}`")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("👤 Student's Answer")
                    st.info(selected_data.get('student_answer', 'Not Answered'))
                
                with col2:
                    st.subheader("📚 Official Answer")
                    st.success(selected_data.get('official_answer', 'N/A'))

                st.markdown("---")
                st.subheader("⚖️ Evaluation")
                score = selected_data.get('score', 'N/A')
                st.metric(label="Score Awarded", value=f"{score}%")
                st.markdown(f"**Justification:** *{selected_data.get('justification', 'N/A')}*")

    st.subheader("🔍 View and Download Results")
    tab1, tab2, tab3 = st.tabs(["Full Evaluation Report", "Student Answers (JSON)", "Official Answers (JSON)"])

    with tab1:
        st.markdown("##### 📝 Evaluation Report")
        report_text = generate_text_report(results, summary)
        st.code(report_text, language="text")
        st.download_button(label="Download Full Report (.txt)", data=report_text, file_name="evaluation_report.txt", mime="text/plain")

    with tab2:
        st.markdown("##### 👤 Student's Answers")
        student_data = st.session_state.get('student_qna_data', {})
        st.json(student_data, expanded=False)
        st.download_button(label="Download Student Answers (.json)", data=json.dumps(student_data, indent=4, ensure_ascii=False), file_name="student_answers.json", mime="application/json")

    with tab3:
        st.markdown("##### 📚 Official Answers")
        official_data = st.session_state.get('official_qna_data', {})
        st.json(official_data, expanded=False)
        st.download_button(label="Download Official Answers (.json)", data=json.dumps(official_data, indent=4, ensure_ascii=False), file_name="official_answers.json", mime="application/json")

   