import streamlit as st
import fitz  # pymupdf
import openai
import os
from fpdf import FPDF
from io import BytesIO
import re

# ─── 1) Page Configuration ────────────────────────────────────────────────
st.set_page_config(page_title="RAIN-CHECK")

# ─── 2) Initialize session state for current movie only ───────────────────
if "current_movie" not in st.session_state:
    st.session_state["current_movie"] = None

# ─── 3) Set your OpenAI API key from Streamlit secrets ─────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ─── 4) Function to extract text from uploaded PDF ─────────────────────────
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ─── 5) Single‐call OpenAI helper ───────────────────────────────────────────
def call_openai_single(prompt: str, model="gpt-4o-mini", temperature=0.7, max_tokens=1500):
    """
    Calls OpenAI with a single prompt and returns the raw text response.
    """
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI chatbot automating script improvements and providing "
                    "data-driven insights (casting, budget, scheduling, marketing) to film producers."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

# ─── 6) Markdown‐cleaning helper ───────────────────────────────────────────
def clean_markdown(text):
    text = re.sub(r"(\*\*|__)", "", text)         # remove bold markers
    text = re.sub(r"(#+\s*)", "", text)            # remove headings
    text = re.sub(r"`", "", text)                  # remove backticks
    text = re.sub(r"\n{3,}", "\n\n", text)         # collapse excessive line breaks
    return text.strip()

# ─── 7) PDF generation function ──────────────────────────────────────────
def create_pdf_report(data: dict) -> BytesIO:
    pdf = FPDF()
    pdf.add_page()

    font_regular = "DejaVuSans.ttf"
    font_bold = "DejaVuSans-Bold.ttf"

    if not os.path.isfile(font_regular) or not os.path.isfile(font_bold):
        raise FileNotFoundError("Font files not found. Make sure DejaVu fonts are in the app folder.")

    pdf.add_font("DejaVu", "", font_regular, uni=True)
    pdf.add_font("DejaVu", "B", font_bold, uni=True)

    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 10, "Screenplay Analysis Report", ln=True, align="C")
    pdf.ln(10)

    for section, content in data.items():
        pdf.set_font("DejaVu", "B", 14)
        pdf.cell(0, 10, section, ln=True)
        pdf.ln(2)

        pdf.set_font("DejaVu", "", 12)
        cleaned_content = clean_markdown(content)
        pdf.multi_cell(0, 8, cleaned_content)
        pdf.ln(10)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# ─── 8) Build prompts dictionary and query OpenAI per section ─────────────
def get_all_analyses_single(screenplay_text: str) -> dict:
    """
    For each of the nine analysis sections, send a separate prompt to OpenAI
    and collect the raw string response under the corresponding key.
    """
    prompts = {
        "Logline": f"""Write a Hollywood-style logline for my screenplay. It should only contain the logline, making it engaging and high-concept.

Screenplay:
\"\"\"{screenplay_text}\"\"\"""",

        "Genre": f"""Suggest the genre for the provided screenplay. By genre, we mean a particular type or style of literature, art, film, or music recognizable by its special characteristics.

Screenplay:
\"\"\"{screenplay_text}\"\"\"""",

        "Top Keywords": f"""Give the top 10 keywords of the attached movie screenplay without any explanation.

Screenplay:
\"\"\"{screenplay_text}\"\"\"""",

        "Location Setting": f"""Give the location setting of the attached movie screenplay, considering only the primary location.

Screenplay:
\"\"\"{screenplay_text}\"\"\"""",

        "Synopsis": f"""Give only the synopsis of the attached screenplay.

Screenplay:
\"\"\"{screenplay_text}\"\"\"""",

        "Script Score": f"""Analyze the attached screenplay and give it a script score out of 10, including:
- Character development score (out of 10) with 1-2 lines explanation
- Plot construction (out of 10) with 1-2 lines explanation
- Dialogue (out of 10) with 1-2 lines explanation
- Originality (out of 10) with 1-2 lines explanation
- Emotional engagement (out of 10) with 1-2 lines explanation
- Theme and message (out of 10) with 1-2 lines explanation
- Overall rating out of 10 with explanation

Screenplay:
\"\"\"{screenplay_text}\"\"\"""",

        "Plot Assessment": f"""Analyze the attached screenplay and give the plot assessment and enhancement, including:
- 5 points of what is working well (positive aspects)
- 5 points where the screenplay lacks
- 5 points of improvements that may be made
- An overall review of the screenplay

Screenplay:
\"\"\"{screenplay_text}\"\"\"""",

        "Character Profiling": f"""Analyze the attached screenplay and return character profiling for the main characters, including:
- Brief description of each main character
- What is working well for each character
- Areas for improvement
- The archetype for each

Screenplay:
\"\"\"{screenplay_text}\"\"\"""",

        "Box Office Collection": f"""Analyze the attached screenplay and give its box office prediction with the following fields:
- Opening day (global and local)
- Opening week (global and local)
- Opening month (global and local)

Screenplay:
\"\"\"{screenplay_text}\"\"\"""",
    }

    results = {}
    for section_name, prompt_text in prompts.items():
        ai_response = call_openai_single(
            prompt_text, model="gpt-4o-mini", temperature=0.7, max_tokens=1200
        )
        results[section_name] = ai_response

    return results

# ─── 9) Main Streamlit UI ─────────────────────────────────────────────────
st.title("RAIN-CHECK")

uploaded_file = st.file_uploader("Upload a movie screenplay (PDF)", type=["pdf"])
if uploaded_file is not None:
    movie_name = os.path.splitext(uploaded_file.name)[0]

    # Extract text on first upload or when movie changes
    if "screenplay_text" not in st.session_state or st.session_state["current_movie"] != movie_name:
        with st.spinner("Extracting screenplay..."):
            st.session_state["screenplay_text"] = extract_text_from_pdf(uploaded_file)
        st.session_state["current_movie"] = movie_name
        st.success("Screenplay extracted and ready!")

    if st.button("Generate Report"):
        with st.spinner("Analyzing screenplay…"):
            all_results = get_all_analyses_single(st.session_state["screenplay_text"])

        if all_results:
            # Display each section in bold, then the raw text
            st.write("**Analysis Results**")
            st.write("")  # add a blank line

            for section, content in all_results.items():
                st.write(f"**{section}:**")
                st.write(clean_markdown(content))
                st.write("")  # blank line between sections

            # PDF download button
            pdf_file = create_pdf_report(all_results)
            st.success("Analysis complete!")
            st.download_button(
                label="Download Analysis Report as PDF",
                data=pdf_file,
                file_name=f"{movie_name}-report.pdf",
                mime="application/pdf"
            )
else:
    st.info("Please upload a PDF screenplay to begin analysis.")
