"""
ClassiDocs backend API.

This Flask application exposes endpoints for:
1. Checking whether the backend and models are available.
2. Predicting the research methodology label from raw text.
3. Predicting both methodology and domain labels from uploaded .txt or .pdf files.

"""


#Importing modules to be used 
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import sys
import fasttext
import fitz

from utils import (
    validate_input_text,
    readable_domain_label,
)

# Flask application setup
app = Flask(__name__)
CORS(app)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NLP_ROOT = PROJECT_ROOT / "NLP Pipeline"
SRC_PATH = NLP_ROOT / "src"
MODEL_PATH_METHOD = NLP_ROOT / "output" / "cfg_A_model_iter_5.ftz"
MODEL_PATH_DOMAIN = NLP_ROOT / "output" / "domain_model.ftz"

sys.path.append(str(SRC_PATH))

method_model = None
domain_model = None

#Load trained models once when the backend starts (with error handling)
if MODEL_PATH_METHOD.exists():
    method_model = fasttext.load_model(str(MODEL_PATH_METHOD))
else:
    print(f"Warning: Methodology model not found at {MODEL_PATH_METHOD}")

if MODEL_PATH_DOMAIN.exists():
    domain_model = fasttext.load_model(str(MODEL_PATH_DOMAIN))
else:
    print(f"Warning: Domain model not found at {MODEL_PATH_DOMAIN}")

#Function for text processing
def clean_text_for_fasttext(text: str):
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text


def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_bytes = uploaded_file.read()
    if not pdf_bytes:
        return ""

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"
    doc.close()
    return text.strip()

#Model function helpers
def predict_with_model(model, text, label_name="prediction", map_domain=False):
    """
    Run prediction with a FastText model and return label + confidence.
    Domain labels can optionally be mapped to human-readable names.
    """
    if model is None:
        return {
            f"{label_name}_error": f"{label_name} model is not loaded."
        }

    labels, probs = model.predict(text, k=1)
    pred_label = labels[0].replace("__label__", "")
    confidence = float(probs[0])

    # Only map domain labels to readable text
    if map_domain:
        return {
            f"{label_name}_prediction": readable_domain_label(pred_label),
            f"{label_name}_code": pred_label,
            f"{label_name}_confidence": round(confidence, 4)
        }

    return {
        f"{label_name}_prediction": pred_label,
        f"{label_name}_confidence": round(confidence, 4)
    }


def predict_text(text: str):
    """
    Clean and validate input text, then return methodology and domain predictions.
    """
    text = clean_text_for_fasttext(text)

    if not text:
        return {"error": "No text provided."}

    # Validate minimum text length before prediction
    is_valid, validated = validate_input_text(text)
    if not is_valid:
        return validated

    text = validated

    methodology_result = predict_with_model(method_model, text, "methodology")

    # Domain is mapped to readable output
    domain_result = predict_with_model(domain_model, text, "domain", map_domain=True)

    return {
        **methodology_result,
        **domain_result
    }


@app.route("/health", methods=["GET"])
def health():
    """
    Health-check endpoint.

    Used to confirm that backend is running and trained models are successfully loaded.
    """
    return jsonify({
        "status": "ok",
        "methodology_model_loaded": method_model is not None,
        "domain_model_loaded": domain_model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Used to predict labels from raw text 
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = predict_text(text)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


@app.route("/predict-file", methods=["POST"])
def predict_file():
    """
    Predict labels from an uploaded .txt or .pdf file.
   """
    #Validate that a file was uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename = file.filename.lower()

    try:
        
        if filename.endswith(".txt"):
            #Read text from file
            content = file.read().decode("utf-8", errors="ignore").strip()
        elif filename.endswith(".pdf"):
            #Extract from file
            content = extract_text_from_pdf(file)
        else:
            return jsonify({
                "error": "Unsupported file type. Please upload a .txt or .pdf file."
            }), 400
        #Clean extracted file
        content = clean_text_for_fasttext(content)

        if not content:
            return jsonify({"error": "No readable text was found in the uploaded file."}), 400
        
        result = predict_text(content)
        #Return predictions 
        return jsonify({
            "filename": file.filename,
            "extracted_text_preview": content[:1000],
            **result
        })

    except Exception as e:
        return jsonify({"error": f"File processing failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)