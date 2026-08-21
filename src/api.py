"""
api.py

FastAPI wrapper around the trained GNN model. This is what your
partner's frontend will call.

Contract (agree this with your partner before they build too far):

  POST /predict
    Request body:
      {
        "drug_a_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "drug_b_smiles": "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O"
      }
    Response body:
      {
        "risk_score": 0.622,
        "is_risky": true
      }

Run locally with:
    uvicorn api:app --reload

Then test in a browser at http://127.0.0.1:8000/docs -- FastAPI
auto-generates an interactive test page, useful for you AND your
partner to try requests without writing any frontend code yet.
"""

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from torch_geometric.data import Batch

from audio_convert import convert_to_wav
from data_processing import smiles_to_graph
from drug_lookup import extract_drug_name_from_ocr, name_to_smiles, search_drug_names
from gnn_model import DrugPairInteractionModel, molgraph_to_pyg_data
from ocr_reader import extract_text_lines
from speech_to_text import transcribe_audio

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="RxReveal API", version="0.1.0")

# Without this, browsers block requests to this API from any page not
# served from the exact same origin (127.0.0.1:8000) -- which would
# include your partner's laptop entirely. This opens it up to any
# origin, which is fine for local dev/testing but should be locked
# down to a specific origin if this ever gets deployed publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_WEIGHTS_PATH = "gnn_model_weights.pt"
FEATURE_SIZE = 14  # matches ATOM_TYPES + extra features in data_processing.py

# Load the model once at startup, not on every request (much faster).
_model = DrugPairInteractionModel(in_channels=FEATURE_SIZE)
_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
_model.eval()


class PredictionRequest(BaseModel):
    drug_a_smiles: str
    drug_b_smiles: str


class PredictionResponse(BaseModel):
    risk_score: float
    is_risky: bool


@app.get("/")
def root():
    """Basic health check -- confirms the API is running."""
    return {"status": "RxReveal API is running"}


@app.get("/drugs/search")
def search_drugs(q: str):
    """
    Live autocomplete search for drug names -- powers a search-as-you-type
    dropdown on the frontend. Returns a plain list of matching drug name
    strings for the given partial query.

    Example: GET /drugs/search?q=para -> ["Paracetamol", "Paracoumarin", ...]
    """
    suggestions = search_drug_names(q)
    return {"suggestions": suggestions}


def resolve_to_smiles(value: str) -> str:
    """
    Accepts EITHER a SMILES string OR a plain drug name, and returns
    a valid SMILES string either way.

    Tries the input as SMILES first (fast, no network call). If that
    fails to parse, falls back to treating it as a drug name and
    looking it up via PubChem -- the same lookup the photo/voice
    endpoints already use. This means /predict now accepts whatever
    is more convenient for the frontend to send: "Aspirin" and
    "CC(=O)OC1=CC=CC=C1C(=O)O" both work.
    """
    try:
        smiles_to_graph(value)  # just validating -- discard the graph here
        return value  # it was already valid SMILES
    except ValueError:
        pass  # not valid SMILES -- try treating it as a drug name instead

    resolved = name_to_smiles(value)
    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=f"'{value}' is not a valid SMILES string and could not "
                   f"be resolved as a drug name either.",
        )
    return resolved


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict drug-drug interaction risk for a pair of drugs.

    Accepts EITHER a SMILES string or a plain drug name (e.g.
    "Aspirin") in each field -- whichever is easier for the frontend
    to send. Plain names are resolved to SMILES automatically via
    PubChem.
    """
    smiles_a = resolve_to_smiles(request.drug_a_smiles)
    smiles_b = resolve_to_smiles(request.drug_b_smiles)

    try:
        graph_a = smiles_to_graph(smiles_a)
        graph_b = smiles_to_graph(smiles_b)
    except ValueError as e:
        # Shouldn't normally happen since resolve_to_smiles() already
        # validated -- kept as a safety net, not the primary error path.
        raise HTTPException(status_code=400, detail=str(e))

    data_a = molgraph_to_pyg_data(graph_a)
    data_b = molgraph_to_pyg_data(graph_b)

    batch_a = Batch.from_data_list([data_a])
    batch_b = Batch.from_data_list([data_b])

    with torch.no_grad():
        logits = _model(batch_a, batch_b)
        risk_score = torch.sigmoid(logits).item()

    return PredictionResponse(
        risk_score=round(risk_score, 4),
        is_risky=risk_score > 0.5,
    )


def image_to_smiles(image_bytes: bytes) -> str:
    """
    Full pipeline for one drug box photo: OCR -> extract likely drug
    name -> look up SMILES via PubChem. Raises HTTPException with a
    clear message at whichever step fails, so the frontend can show
    the user something actionable ("couldn't read the box" vs
    "couldn't find that drug") instead of a generic error.
    """
    text_lines = extract_text_lines(image_bytes)
    if not text_lines:
        raise HTTPException(
            status_code=400,
            detail="Could not read any text from the image. Try a clearer, "
                   "well-lit photo of the drug box.",
        )

    smiles = extract_drug_name_from_ocr(text_lines)
    if not smiles:
        raise HTTPException(
            status_code=404,
            detail=f"Read text from the image ({text_lines}) but could not "
                   f"match it to a known drug. Try a clearer photo, or use "
                   f"the /predict endpoint with the drug name/SMILES directly.",
        )
    return smiles


@app.post("/predict-from-image", response_model=PredictionResponse)
async def predict_from_image(
    drug_a_image: UploadFile = File(...),
    drug_b_image: UploadFile = File(...),
):
    """
    Predict drug-drug interaction risk from two photos of drug boxes.

    Pipeline per image: OCR reads the box -> extract the likely drug
    name -> look up its SMILES via PubChem -> feed both into the same
    GNN model /predict already uses.
    """
    image_a_bytes = await drug_a_image.read()
    image_b_bytes = await drug_b_image.read()

    smiles_a = image_to_smiles(image_a_bytes)
    smiles_b = image_to_smiles(image_b_bytes)

    return predict(PredictionRequest(drug_a_smiles=smiles_a, drug_b_smiles=smiles_b))


def audio_to_smiles(audio_bytes: bytes, source_format: str = "webm") -> str:
    """
    Full pipeline for one spoken drug name: convert to WAV (browsers'
    live mic recordings typically arrive as webm) -> transcribe ->
    look up SMILES via PubChem. Mirrors image_to_smiles() above.
    """
    try:
        wav_bytes = convert_to_wav(audio_bytes, source_format=source_format)
    except Exception as e:
        print(f"[api] Audio conversion failed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Could not process the audio file. Make sure it's a "
                   "valid recording.",
        )

    transcribed_text = transcribe_audio(wav_bytes)
    if not transcribed_text:
        raise HTTPException(
            status_code=400,
            detail="Could not understand the audio. Try a clearer recording "
                   "with less background noise.",
        )

    smiles = name_to_smiles(transcribed_text)
    if not smiles:
        raise HTTPException(
            status_code=404,
            detail=f"Heard '{transcribed_text}' but could not match it to a "
                   f"known drug. Try again, or use the /predict endpoint "
                   f"with the drug name/SMILES directly.",
        )
    return smiles


@app.post("/predict-from-speech", response_model=PredictionResponse)
async def predict_from_speech(
    drug_a_audio: UploadFile = File(...),
    drug_b_audio: UploadFile = File(...),
):
    """
    Predict drug-drug interaction risk from two spoken drug names.

    Pipeline per audio clip: convert to WAV (from webm, the typical
    browser live-microphone format) -> speech-to-text transcription ->
    look up SMILES via PubChem -> feed both into the same GNN model
    /predict already uses.

    Accepts webm audio (standard browser MediaRecorder output) by
    default. If your frontend sends a different format, adjust the
    source_format argument passed to audio_to_smiles() below.
    """
    audio_a_bytes = await drug_a_audio.read()
    audio_b_bytes = await drug_b_audio.read()

    smiles_a = audio_to_smiles(audio_a_bytes)
    smiles_b = audio_to_smiles(audio_b_bytes)

    return predict(PredictionRequest(drug_a_smiles=smiles_a, drug_b_smiles=smiles_b))