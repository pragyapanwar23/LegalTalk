import streamlit as st
from PIL import Image
from pdf2image import convert_from_path
from docx import Document
import pytesseract
import io
import os

# Page configuration
st.set_page_config(page_title="OCR Text Extractor", page_icon="📝", layout="wide")

st.title("🧾 OCR Text Extractor App")
st.write("Upload an image, PDF, or Word document to extract text using OCR.")

# --- Helper function for OCR ---
def ocr_image(img):
    return pytesseract.image_to_string(img)

# --- File uploader ---
uploaded_file = st.file_uploader("📤 Upload your file", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp", "pdf", "docx"])

if uploaded_file:
    file_name = uploaded_file.name
    extracted_text = ""

    # --- If image ---
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        with st.spinner("🔍 Performing OCR..."):
            extracted_text = ocr_image(img)

    # --- If PDF ---
    elif file_name.lower().endswith('.pdf'):
        with st.spinner("📄 Converting PDF pages and extracting text..."):
            pdf_bytes = uploaded_file.read()
            with open("temp.pdf", "wb") as f:
                f.write(pdf_bytes)
            pages = convert_from_path("temp.pdf")
            for i, page in enumerate(pages, start=1):
                st.image(page, caption=f"Page {i}", use_container_width=True)
                extracted_text += f"\n\n=== PAGE {i} ===\n"
                extracted_text += ocr_image(page)
            os.remove("temp.pdf")

    # --- If DOCX ---
    elif file_name.lower().endswith('.docx'):
        with st.spinner("📘 Reading Word document..."):
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"
            # Extract text from embedded images
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    image_data = rel.target_part.blob
                    img = Image.open(io.BytesIO(image_data))
                    st.image(img, caption="Embedded Image", use_container_width=True)
                    extracted_text += "\n[Image OCR]:\n" + ocr_image(img)

    else:
        st.warning("⚠️ Unsupported file type. Please upload an image, PDF, or DOCX file.")

    # --- Display extracted text ---
    st.subheader("🧠 Extracted Text")
    st.text_area("", extracted_text, height=400)

    # --- Option to download extracted text ---
    st.download_button(
        label="💾 Download Extracted Text",
        data=extracted_text,
        file_name=f"{os.path.splitext(file_name)[0]}_extracted.txt",
        mime="text/plain"
    )

else:
    st.info("👆 Please upload a file to start.")
