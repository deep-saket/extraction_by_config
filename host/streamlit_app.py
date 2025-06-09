import os
import json
import streamlit as st
import requests
from typing import List
from extraction_io.ExtractionItems import ExtractionItems, ExtractionItem

# Disable file watching to avoid torch error
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Paths
DE_CONFIG_DIR = "../de_config"
PDF_DATASET_DIR = "../dataset"
OUTPUT_DIR = "../output"
FASTAPI_URL = "http://localhost:8001/perform_de"  # Adjust if hosted differently

st.set_page_config(page_title="DE Config Dashboard", layout="wide")
st.title("🧠 Document Extraction Config Dashboard")

# --- Helper Functions ---
def load_config(filepath: str) -> ExtractionItems:
    st.write(f"[DEBUG] Loading config from {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return ExtractionItems.model_validate(data)

def save_config(filepath: str, data: ExtractionItems):
    st.write(f"[DEBUG] Saving config to {filepath}")
    with open(filepath, 'w') as f:
        json.dump([item.model_dump() for item in data], f, indent=2)

# --- Navigation Mode Handling ---
if "mode" not in st.session_state:
    st.session_state.mode = "view"

if st.button("📂 Show All DE Configs"):
    st.session_state.mode = "view"
if st.button("📝 Add New DE Config"):
    st.session_state.mode = "add"
if st.button("🚀 Perform DE"):
    st.session_state.mode = "perform"
if st.button("📄 View Outputs"):
    st.session_state.mode = "outputs"

# --- Show All DE Configs ---
if st.session_state.mode == "view":
    st.write("[DEBUG] Showing all DE configs")
    config_files = [f for f in os.listdir(DE_CONFIG_DIR) if f.endswith(".json")]
    selected_file = st.selectbox("Choose a config file", config_files)
    if selected_file:
        file_path = os.path.join(DE_CONFIG_DIR, selected_file)
        items = load_config(file_path)
        st.subheader(f"Editing: {selected_file}")

        for i, item in enumerate(items):
            with st.expander(f"✏️ Edit Entry #{i + 1} - `{item.field_name}`"):
                item.field_name = st.text_input("Field Name", item.field_name, key=f"fn_{i}")
                item.description = st.text_area("Description", item.description, key=f"desc_{i}")
                item.type = st.selectbox("Type", ["key-value", "bullet-points", "summarization", "checkbox"],
                                        index=["key-value", "bullet-points", "summarization", "checkbox"].index(item.type),
                                        key=f"type_{i}")
                pp_str = st.text_input("Probable Pages (comma-separated)",
                                       ", ".join(map(str, item.probable_pages or [])), key=f"pp_{i}")
                try:
                    item.probable_pages = list(map(int, filter(None, pp_str.split(","))))
                except:
                    st.error("Invalid probable page numbers")
                item.multipage_value = st.checkbox("Multipage Value", item.multipage_value, key=f"mpv_{i}")
                item.multiline_value = st.checkbox("Multiline Value", item.multiline_value, key=f"mlv_{i}")
                item.search_keys = st.text_area("Search Keys (one per line)",
                                               "\n".join(item.search_keys or []), key=f"sk_{i}").splitlines()
                raw_extra = st.text_area("Extra Rules (JSON format)", json.dumps(item.extra_rules, indent=2),
                                        key=f"extra_{i}")
                try:
                    item.extra_rules = json.loads(raw_extra)
                except:
                    st.error("Invalid JSON in Extra Rules")
                if item.type == "summarization":
                    item.scope = st.selectbox("Summarization Scope",
                                             ["whole", "section", "pages", "extracted_fields"],
                                             index=["whole", "section", "pages", "extracted_fields"]
                                             .index(item.scope or "whole"), key=f"scope_{i}")
                    if item.scope == "section":
                        item.section_name = st.text_input("Section Name", item.section_name or "",
                                                        key=f"sec_{i}")

        if st.button("💾 Save Changes"):
            save_config(file_path, items)
            st.success(f"Saved changes to {selected_file}")

# --- Perform DE ---
elif st.session_state.mode == "perform":
    st.subheader("🚀 Perform DE")
    config_files = [f for f in os.listdir(DE_CONFIG_DIR) if f.endswith(".json")]
    selected_file = st.selectbox("Select a DE Config", config_files, key="de_config_file")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_pdf:
        st.success("✅ PDF uploaded successfully.")
        pdf_save_path = os.path.join(PDF_DATASET_DIR, uploaded_pdf.name)
        with open(pdf_save_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        output_json_path = os.path.join(OUTPUT_DIR, uploaded_pdf.name.replace(".pdf", ".json"))
        extraction_config_path = os.path.join(DE_CONFIG_DIR, selected_file)

        st.session_state.pdf_path = pdf_save_path
        st.session_state.output_path = output_json_path
        st.session_state.config_path = extraction_config_path
        st.session_state.ready_to_extract = True
    else:
        st.warning("📭 Please upload a PDF to proceed.")

    if st.session_state.get("ready_to_extract", False):
        if st.button("🟢 Run Extraction"):
            st.write("[DEBUG] Starting extraction...")
            try:
                with st.spinner("Calling FastAPI..."):
                    print({
                            "pdf_path": st.session_state.pdf_path,
                            "extraction_config_path": st.session_state.config_path,
                            "output_json_path": st.session_state.output_path,
                        })
                    with open(st.session_state.pdf_path, 'rb') as f:
                        files = {
                            "pdf": (os.path.basename(st.session_state.pdf_path), f, "application/pdf")
                        }
                        data = {
                            "config_name": os.path.basename(st.session_state.config_path)
                        }

                        response = requests.post(
                            "http://localhost:8001/perform_de",
                            files=files,
                            data=data,
                            timeout=500
                        )

                if response.status_code == 200:
                    st.success("✅ Extraction completed!")
                    with open(st.session_state.output_path, 'r') as f:
                        st.json(json.load(f))
                else:
                    st.error(f"❌ FastAPI returned error {response.status_code}")
                    st.text(response.text)

            except requests.exceptions.Timeout:
                st.error("⏱️ Timeout from FastAPI.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to FastAPI.")
            except Exception as e:
                st.error(f"💥 Unexpected error: {e}")

elif st.session_state.mode == "add":
    st.subheader("📝 Add New DE Config")

    new_items: List[ExtractionItem] = []
    num_fields = st.number_input("How many fields to add?", min_value=1, max_value=50, value=1, step=1)

    for i in range(num_fields):
        with st.expander(f"➕ New Field #{i+1}"):
            field_name = st.text_input("Field Name", key=f"add_fn_{i}")
            description = st.text_area("Description", key=f"add_desc_{i}")
            type_val = st.selectbox("Type", ["key-value", "bullet-points", "summarization", "checkbox"], key=f"add_type_{i}")
            probable_pages_str = st.text_input("Probable Pages (comma-separated)", key=f"add_pp_{i}")
            probable_pages = list(map(int, filter(None, probable_pages_str.split(",")))) if probable_pages_str else []
            multipage = st.checkbox("Multipage Value", key=f"add_mpv_{i}")
            multiline = st.checkbox("Multiline Value", key=f"add_mlv_{i}")
            # scope one of 
            search_keys = st.text_area("Search Keys (one per line)", key=f"add_sk_{i}").splitlines()
            extra_rules_raw = st.text_area("Extra Rules (JSON)", "{}", key=f"add_extra_{i}")
            try:
                extra_rules = json.loads(extra_rules_raw)
            except:
                st.warning("⚠️ Invalid JSON in Extra Rules, using empty dict.")
                extra_rules = {}

            scope = section_name = None
            if type_val == "summarization":
                scope = st.selectbox("Summarization Scope", ["whole", "section", "pages", "extracted_fields"],
                                     key=f"add_scope_{i}")
                if scope == "section":
                    section_name = st.text_input("Section Name", key=f"add_section_{i}")
            if type_val == "checkbox":
                scope = st.selectbox("Summarization Scope", ["single_value", "multi_value"],
                                     key=f"add_scope_{i}")

            new_items.append(ExtractionItem(
                field_name=field_name,
                description=description,
                type=type_val,
                probable_pages=probable_pages,
                multipage_value=multipage,
                multiline_value=multiline,
                search_keys=search_keys,
                extra=extra_rules,
                scope=scope,
                section_name=section_name,
            ))

    new_filename = st.text_input("Config Filename (e.g., my_config.json)")
    if st.button("💾 Save Config"):
        if new_filename:
            save_path = os.path.join(DE_CONFIG_DIR, new_filename)
            print("new_items =", new_items)
            save_config(save_path, ExtractionItems(new_items))
            st.success(f"✅ Saved to {new_filename}")
        else:
            st.error("⚠️ Please provide a valid filename.")

elif st.session_state.mode == "outputs":
    st.subheader("📄 View Extracted Output Files")
    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]
    selected_output = st.selectbox("Choose an output file", output_files)

    if selected_output:
        output_path = os.path.join(OUTPUT_DIR, selected_output)
        st.write(f"📄 Showing: `{selected_output}`")
        with open(output_path, 'r') as f:
            st.json(json.load(f))