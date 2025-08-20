import streamlit as st
import json
import re
import os
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from docx import Document
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize

# --- Agent 1: API Type and Security Agent ---
def api_type_agent(tokens):
    """Identifies API type and security based on keywords."""
    spec = {
        "apiType": "REST",
        "security": "OAuth2 (placeholder)",
    }
    if "graphql" in tokens:
        spec["apiType"] = "GraphQL"
    if any(x in tokens for x in ["real-time", "real time", "grpc", "bidirectional"]):
        spec["apiType"] = "gRPC"
    if any(x in tokens for x in ["login", "token", "oauth", "api-key"]):
        spec["security"] = "token based"
    return spec

# --- Agent 2: Data Model and Endpoint Agent ---
def data_model_agent(tokens):
    """Identifies data models and endpoints based on keywords."""
    spec = {
        "endpoints": ["/sample"],
        "dataModel": {},
    }
    if "user" in tokens:
        spec["dataModel"]["user"] = {"id": "integer", "name": "string"}
    if "products" in tokens:
        spec["endpoints"].append("/products")
        spec["dataModel"]["products"] = {"id": "integer", "name": "string", "price": "number"}
    if "order" in tokens:
        spec["endpoints"].append("/orders")
        spec["dataModel"]["order"] = {"id": "integer", "user_id": "integer", "status": "string"}
    return spec

# --- Main App ---
st.title("📑 API Advisor from RFP")

uploaded_file = st.file_uploader("Upload RFP (txt/pdf)", type=["txt", "pdf"])

rfp_text = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            rfp_text += page.extract_text() or ""
    else:
        rfp_text = uploaded_file.read().decode("utf-8", errors="ignore")
    st.text_area("RFP Content", rfp_text, height=200)

if st.button("🚀 Generate API Spec"):
    if not rfp_text:
        st.error("Please upload an RFP file first.")
    else:
        with st.spinner("🧠 Analyzing RFP with dual agents..."):
            try:
                nltk.download('punkt', quiet=True)
                tokens = word_tokenize(rfp_text.lower())
            except Exception:
                tokens = rfp_text.lower().split()
                
            # Run both agents
            type_and_security_spec = api_type_agent(tokens)
            data_and_endpoint_spec = data_model_agent(tokens)
            
            # Combine the results
            final_spec = {**type_and_security_spec, **data_and_endpoint_spec}
            
            st.subheader("4️⃣ Draft API Spec")
            st.json(final_spec)
            st.session_state.spec = final_spec

if 'spec' in st.session_state and st.session_state.spec:
    spec_content = json.dumps(st.session_state.spec, indent=2)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("💾 Export as JSON", spec_content, file_name="api_spec.json", mime="application/json")
    with col2:
        output = BytesIO()
        pdf = SimpleDocTemplate(output)
        styles = getSampleStyleSheet()
        pdf.build([Paragraph(spec_content, styles['Normal'])])
        st.download_button("📄 Export as PDF", output.getvalue(), file_name="api_spec.pdf", mime="application/pdf")
    with col3:
        doc = Document()
        doc.add_heading("API Spec", 0)
        doc.add_paragraph(spec_content)
        doc_io = BytesIO()
        doc.save(doc_io)
        st.download_button("📝 Export as Word", doc_io.getvalue(), file_name="api_spec.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
