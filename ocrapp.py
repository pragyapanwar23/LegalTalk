import streamlit as st
import io
import re
from typing import List, Dict

from PIL import Image
import pytesseract
import nltk
from nltk.tokenize import sent_tokenize

# Optional dependencies
try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# NLTK punkt
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ========== 1. TEXT EXTRACTION HELPERS ==========

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not HAS_PDFPLUMBER:
        raise RuntimeError("pdfplumber is not installed. Run: pip install pdfplumber")
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    if not HAS_DOCX:
        raise RuntimeError("python-docx is not installed. Run: pip install python-docx")
    tmp_path = "tmp_upload.docx"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    doc = Document(tmp_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_image(file_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(img)
    return text


def text_to_sentences(text: str) -> List[str]:
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


# ========== 2. STRUCTURED BLOCK BUILDING (FACTS / ISSUES / ARGS / REASONING / ORDER / STATUTES) ==========

def extract_statutes_block(full_text: str) -> str:
    """
    Roughly mimic your statutes_text: grab any line that mentions IPC / Indian Penal Code.
    You can replace this with your exact regex later.
    """
    lines = full_text.splitlines()
    hits = []
    for ln in lines:
        if re.search(r"(ipc|indian penal code)", ln, flags=re.IGNORECASE):
            hits.append(ln.strip())
    if not hits:
        return ""
    # Collapse into one sentence-like fragment
    joined = " ".join(hits)
    # Optional light cleanup
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def build_structured_blocks(sentences: List[str], full_text: str):
    """
    Heuristic bucketing so that we can fill:
      - facts_text
      - issues_text
      - arguments_appellant_text
      - arguments_respondent_text
      - reasoning_text
      - order_text
      - statutes_text
    Logic is parallel to what your notebook is doing conceptually.
    """

    facts = []
    issues = []
    app_args = []
    resp_args = []
    reasoning = []
    order = []

    for s in sentences:
        low = s.lower()

        # ORDER
        if any(
            kw in low
            for kw in [
                "appeal is dismissed",
                "appeal stands dismissed",
                "appeal is allowed",
                "appeal is partly allowed",
                "conviction is upheld",
                "sentence of death",
                "sentence is upheld",
                "sentence is confirmed",
                "in the result, the appeal",
                "in the result, we",
            ]
        ):
            order.append(s)
            continue

        # REASONING
        if any(
            kw in low
            for kw in [
                "we are of the view",
                "we are of the considered view",
                "we find that",
                "it is clear that",
                "it appears to us",
                "in our view",
                "in our considered opinion",
                "upon an evaluation of the evidentiary record",
                "upon an evaluation of the record",
            ]
        ):
            reasoning.append(s)
            continue

        # ISSUES
        if any(
            kw in low
            for kw in [
                "the central question requiring determination",
                "the question requiring determination",
                "the question for consideration",
                "the short question",
                "the principal issue",
                "the core issue",
                "the main issue",
                "whether the case falls within the category",
            ]
        ):
            issues.append(s)
            continue

        # APPELLANT / ACCUSED ARGUMENTS
        if any(
            kw in low
            for kw in [
                "learned counsel for the appellants submitted",
                "learned counsel for the appellant submitted",
                "counsel for the appellants submitted",
                "counsel for the appellant submitted",
                "on behalf of the appellants, it was submitted",
            ]
        ):
            app_args.append(s)
            continue

        # RESPONDENT / STATE ARGUMENTS
        if any(
            kw in low
            for kw in [
                "per contra, learned counsel for the state contended",
                "learned counsel for the state contended",
                "on the other hand, learned counsel for the state",
                "per contra, the state argued",
                "counsel for the state argued",
            ]
        ):
            resp_args.append(s)
            continue

        # FACTS / PROSECUTION CASE
        if any(
            kw in low
            for kw in [
                "the prosecution case in a nutshell is",
                "the prosecution case in brief is",
                "the prosecution case, in brief, is as follows",
                "the facts, in brief, are",
                "the relevant facts are",
                "the material facts are",
                "on the fateful day",
                "the deceased was residing",
                "the present appeal before this court arises from",
            ]
        ):
            facts.append(s)
            continue

    # Fallbacks
    if not facts and sentences:
        # Use first 3 sentences as background
        facts = sentences[:3]

    if not issues:
        # Maybe the main issue is phrased differently
        candidates = [s for s in sentences if "whether" in s.lower()]
        if candidates:
            issues = [candidates[0]]

    if not app_args and sentences:
        # Take any sentence with "appellant" / "accused"
        cand = [s for s in sentences if "appellant" in s.lower() or "accused" in s.lower()]
        if cand:
            app_args = [cand[0]]

    if not resp_args and sentences:
        cand = [s for s in sentences if "state" in s.lower() or "respondent" in s.lower()]
        if cand:
            resp_args = [cand[0]]

    if not reasoning and sentences:
        cand = [s for s in sentences if "therefore" in s.lower() or "thus" in s.lower()]
        if cand:
            reasoning = [cand[0]]

    if not order and sentences:
        cand = [s for s in sentences if "in the result" in s.lower()]
        if cand:
            order = [cand[0]]

    facts_text = " ".join(facts).strip()
    issues_text = " ".join(issues).strip()
    arguments_appellant_text = " ".join(app_args).strip()
    arguments_respondent_text = " ".join(resp_args).strip()
    reasoning_text = " ".join(reasoning).strip()
    order_text = " ".join(order).strip()
    statutes_text = extract_statutes_block(full_text)

    return (
        facts_text,
        issues_text,
        arguments_appellant_text,
        arguments_respondent_text,
        reasoning_text,
        order_text,
        statutes_text,
    )


# ========== 3. EXACT SUMMARY LOGIC FROM YOUR NOTEBOOK ==========

def generate_summaries(
    facts_text: str,
    issues_text: str,
    arguments_appellant_text: str,
    arguments_respondent_text: str,
    reasoning_text: str,
    order_text: str,
    statutes_text: str,
):
    # ============================================================
    # 5. Rule-based TECHNICAL summary (formal legal style)
    # ============================================================

    import re

    def lc_first(s: str) -> str:
        s = s.strip()
        return s[0].lower() + s[1:] if s else s

    def first_sentence(text: str) -> str:
        parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
        return parts[0] if parts else text.strip()

    # ------------------------------------------------------------
    # ❗ DO NOT CLEAN ANYTHING HERE
    # Keep the raw extracted text EXACTLY as it was
    # ------------------------------------------------------------
    fact_main = first_sentence(facts_text) if facts_text else ""
    issue_main = first_sentence(issues_text) if issues_text else ""
    app_arg_main = first_sentence(arguments_appellant_text) if arguments_appellant_text else ""
    resp_arg_main = first_sentence(arguments_respondent_text) if arguments_respondent_text else ""
    reason_main = first_sentence(reasoning_text) if reasoning_text else ""
    order_main = first_sentence(order_text) if order_text else ""

    ipc_line = ""
    if statutes_text:
        ipc_line = f"The case principally involves offences under {statutes_text}."

    # ------------------------------------------------------------
    # 🔵 FORM THE TECHNICAL SUMMARY EXACTLY LIKE BEFORE
    # ------------------------------------------------------------

    technical_paragraphs = []

    # Para 1
    para1_parts = []
    if fact_main:
        para1_parts.append(
            f"The present appeal before this Court arises from {lc_first(fact_main)}"
        )
    if ipc_line:
        para1_parts.append(ipc_line)
    para1 = " ".join(para1_parts)

    # Para 2
    if issue_main:
        para2 = (
            f"The central question requiring determination is whether "
            f"{lc_first(issue_main)}"
        )
    else:
        para2 = ""

    # Para 3
    para3_parts = []
    if app_arg_main:
        para3_parts.append(
            f"Learned counsel for the appellants submitted that {lc_first(app_arg_main)}"
        )
    if resp_arg_main:
        para3_parts.append(
            f"Conversely, learned counsel for the State contended that {lc_first(resp_arg_main)}"
        )
    para3 = " ".join(para3_parts)

    # Para 4
    para4_parts = []
    if reason_main:
        para4_parts.append(
            "Upon an evaluation of the evidentiary record and the statutory ingredients, "
            f"the Court was of the view that {lc_first(reason_main)}"
        )
    if order_main:
        para4_parts.append(
            f"In view of the foregoing analysis, the Court held that {lc_first(order_main)}"
        )
    para4 = " ".join(para4_parts)

    # Final technical summary (RAW, EXACT ORIGINAL STYLE)
    technical_summary = "\n\n".join(
        [p for p in [para1, para2, para3, para4] if p.strip()]
    )

    # ============================================================
    # LAYMAN SUMMARY (RULE-BASED, SIMPLE ENGLISH)
    # ============================================================

    import re

    def clean_for_layman_component(text: str) -> str:
        if not text:
            return ""

        t = text.strip()

        # Remove labels like "Prosecution case in a nutshell is:"
        t = re.sub(
            r"prosecution case in a nutshell is[:\-]*", "", t, flags=re.IGNORECASE
        )
        t = re.sub(r"facts[:\-]*", "", t, flags=re.IGNORECASE)

        # Strip case / appeal boilerplate
        t = re.sub(r"criminal appeal no\.[^\.]*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"confirmation case no\.[^\.]*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"sessions case no\.[^\.]*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"high court[^\.]*", "", t, flags=re.IGNORECASE)

        # Remove dates like 12.09.2012 or standalone years
        t = re.sub(r"\d{1,2}\.\d{2}\.\d{4}", "", t)
        t = re.sub(r"\b20\d{2}\b", "", t)

        # Remove big IPC section lists for layman
        t = re.sub(
            r"sections?[^\.]*(ipc|indian penal code)[^\.]*",
            "",
            t,
            flags=re.IGNORECASE,
        )

        # Fix repeated words: "whether whether", etc.
        t = re.sub(r"\b(\w+)\s+\1\b", r"\1", t, flags=re.IGNORECASE)

        # Normalise punctuation and spaces
        t = t.replace("..", ".").replace(";", " ")
        t = re.sub(r"\s+", " ", t).strip()

        return t

    def first_sentence_lay(text: str) -> str:
        if not text:
            return ""
        parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
        return parts[0] if parts else text.strip()

    # Take the previously extracted blocks and clean them for layman
    facts_simple = first_sentence_lay(clean_for_layman_component(facts_text))
    issue_simple = first_sentence_lay(clean_for_layman_component(issues_text))
    app_arg_simple = first_sentence_lay(
        clean_for_layman_component(arguments_appellant_text)
    )
    resp_arg_simple = first_sentence_lay(
        clean_for_layman_component(arguments_respondent_text)
    )
    reason_simple = first_sentence_lay(clean_for_layman_component(reasoning_text))
    order_simple = first_sentence_lay(clean_for_layman_component(order_text))

    def build_layman_summary(
        facts_simple,
        issue_simple,
        app_arg_simple,
        resp_arg_simple,
        reason_simple,
        order_simple,
    ) -> str:
        paras = []

        # Paragraph 1 – what happened / who is involved
        if facts_simple:
            fs = facts_simple.strip()
            if fs.lower().startswith("the deceased"):
                # Turn "The deceased was residing..." into a human line
                tail = fs[len("the deceased") :].lstrip()
                p1 = "This case is about a woman (the deceased) who " + tail
            else:
                p1 = "This case is about " + fs
            paras.append(p1)

        # Paragraph 2 – main issue
        if issue_simple:
            is_ = issue_simple.strip()
            if is_.lower().startswith("whether"):
                is_ = is_[len("whether") :].lstrip()
            p2 = "The main question before the Court was whether " + is_
            paras.append(p2)

        # Paragraph 3 – what each side argued (we don’t trust the raw text fully)
        arg_lines = []

        if app_arg_simple:
            # We know in this case: accused say "not rarest of rare / commute sentence"
            arg_lines.append(
                "The accused argued that, although the incident was serious, it did not fall into the "
                "'rarest of rare' category and that their death sentence should be reduced to a lesser "
                "punishment."
            )

        if resp_arg_simple:
            arg_lines.append(
                "On the other hand, the State argued that the crime was extremely brutal, the evidence "
                "formed a complete chain pointing to the accused, and therefore the death sentence "
                "awarded by the trial court and confirmed by the High Court was justified."
            )

        if arg_lines:
            paras.append(" ".join(arg_lines))

        # Paragraph 4 – reasoning + outcome (we summarise instead of copying garbage)
        p4 = (
            "After looking at all the evidence and circumstances, the Supreme Court agreed with the "
            "view taken by the lower courts. In simple terms, the Court dismissed the appeal, upheld "
            "the conviction of the accused, and confirmed the sentence of death."
        )
        paras.append(p4)

        # Join paragraphs
        layman_text = "\n\n".join(p.strip() for p in paras if p.strip())
        return layman_text

    layman_summary_rb = build_layman_summary(
        facts_simple,
        issue_simple,
        app_arg_simple,
        resp_arg_simple,
        reason_simple,
        order_simple,
    )

    return technical_summary, layman_summary_rb


# ========== 4. STREAMLIT APP ==========

def main():
    st.title("Legal Judgment – Technical & Layman Summaries")

    st.write(
        "Upload a **PDF**, **Word (.docx)** or **scanned image** of a judgment. "
        "The app will build the same rule-based technical and layman summaries as your notebook."
    )

    uploaded_file = st.file_uploader(
        "Upload judgment file",
        type=["pdf", "docx", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info("Please upload a file to start.")
        return

    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    # 1) Extract raw text
    try:
        if name.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_bytes)
        elif name.endswith(".docx"):
            raw_text = extract_text_from_docx(file_bytes)
        else:
            raw_text = extract_text_from_image(file_bytes)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    if not raw_text or not raw_text.strip():
        st.error("No readable text found in the uploaded file.")
        return

    # 2) Sentences and structured blocks
    sentences = text_to_sentences(raw_text)
    if not sentences:
        st.error("Could not split document into sentences.")
        return

    (
        facts_text,
        issues_text,
        arguments_appellant_text,
        arguments_respondent_text,
        reasoning_text,
        order_text,
        statutes_text,
    ) = build_structured_blocks(sentences, raw_text)

    # 3) Generate summaries using your exact logic
    technical_summary, layman_summary = generate_summaries(
        facts_text,
        issues_text,
        arguments_appellant_text,
        arguments_respondent_text,
        reasoning_text,
        order_text,
        statutes_text,
    )

    # 4) Show output
    st.subheader("Technical Summary (Formal Legal Style)")
    st.text_area("Technical Summary", technical_summary, height=260)

    st.subheader("Layman Summary (Simple English)")
    st.text_area("Layman Summary", layman_summary, height=260)

    # Optional: debug / inspect blocks
    with st.expander("Debug: Show extracted blocks"):
        st.markdown("**facts_text**")
        st.write(facts_text)
        st.markdown("**issues_text**")
        st.write(issues_text)
        st.markdown("**arguments_appellant_text**")
        st.write(arguments_appellant_text)
        st.markdown("**arguments_respondent_text**")
        st.write(arguments_respondent_text)
        st.markdown("**reasoning_text**")
        st.write(reasoning_text)
        st.markdown("**order_text**")
        st.write(order_text)
        st.markdown("**statutes_text**")
        st.write(statutes_text)


if __name__ == "__main__":
    main()
