SETUP.md
=========

Full setup instructions for the RxReveal backend, including the parts
that a plain `pip install -r requirements.txt` can't handle on its own
(PyTDC needs a special install flag, and it also needs three small
stub files to work around a Windows-incompatible dependency it drags
in for an unrelated feature).

Run every command from inside `rxreveal-backend/src`, with your venv
activated.

---

## 1. Create and activate a virtual environment

    py -3.12 -m venv venv
    venv\Scripts\activate

(Must be Python 3.12 specifically -- newer versions don't yet have
pre-built installers for some of these packages, which forces slow,
error-prone compilation from source on Windows.)

---

## 2. Install the standard packages

    pip install -r requirements.txt

---

## 3. Install PyTDC separately, without its dependencies

PyTDC's own listed dependencies pull in an old, incompatible version
of scikit-learn (needs compiling from source, needs Microsoft's C++
Build Tools) and a huge, unnecessary set of ML libraries unrelated to
what we actually use it for. Installing it this way avoids all of that:

    pip install PyTDC --no-deps

---

## 4. Add the three stub files

PyTDC imports three packages at startup for an unrelated single-cell
genomics feature we never use -- one of which (`tiledbsoma`) has no
Windows installer at all. These three empty files satisfy the import
without needing the real (and largely unusable-on-Windows) packages.

Create these three files inside `rxreveal-backend/src`, each containing
just a one-line comment (content doesn't matter, they just need to exist):

    tiledbsoma.py
    cellxgene_census.py
    gget.py

Quickest way, from inside `src` with your venv active:

    "# stub" | Out-File -FilePath tiledbsoma.py -Encoding utf8
    "# stub" | Out-File -FilePath cellxgene_census.py -Encoding utf8
    "# stub" | Out-File -FilePath gget.py -Encoding utf8

---

## 5. Verify everything works

    python -c "from tdc.multi_pred import DDI; print('TDC OK')"
    python -c "import torch, torch_geometric, rdkit; print('Core ML stack OK')"
    python -c "import easyocr; print('OCR OK')"
    python -c "import speech_recognition; print('Speech recognition OK')"

If all four print their "OK" message with no errors, setup is complete.

---

## 6. Run the API

    uvicorn api:app --reload --host 0.0.0.0

Visit http://127.0.0.1:8000/docs to confirm it's running.