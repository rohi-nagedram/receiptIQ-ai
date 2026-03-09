# ReceiptIQ 🧾
### On-Device AI Receipt Data Extraction
**The GPT Challenge 2026 | GPT-4o + EasyOCR + RunAnywhere SDK**

> Privacy-first · Zero cloud image exposure · Works offline · Under 3 seconds

---

## Problem
63 million Indian SMEs process paper receipts manually — 3-5 minutes per receipt, high error rates, and existing cloud OCR tools send your financial data to third-party servers.

## Solution
ReceiptIQ extracts structured data from any receipt image entirely on-device. The image never leaves your device. Only OCR text is sent to GPT-4o.

```json
{
  "merchant":           "Ishwar Motors",
  "gstin":              "23AAJFI7828R1ZM",
  "date":               "28/07/2025",
  "amount_INR":         "600,000",
  "category":           "Vehicle",
  "payment_method":     "Bank/Finance",
  "invoice_number":     "66",
  "extraction_method":  "GPT-4o",
  "processing_time_s":  2.4,
  "data_sent_to_cloud": false
}
```

---

## Pipeline
```
Receipt Image → OpenCV Preprocessing → EasyOCR (On-Device) → GPT-4o (Text Only) → Structured JSON
```

---

## Repo Structure
```
receiptiq-ai/
├── app.py                      ← Streamlit web app (deployed link)
├── receiptiq.py                ← Core pipeline (CLI)
├── ReceiptIQ_v4.ipynb          ← Google Colab notebook
├── requirements.txt            ← All dependencies
├── sample_invoice.png          ← Test invoice image
├── ReceiptIQ_Presentation.pptx ← Hackathon PPT (10 slides)
├── DEMO_VIDEO_SCRIPT.md        ← Exact 2-minute demo script
├── SUBMISSION.md               ← Submission checklist
├── PROJECT_REPORT.md           ← Full technical report
└── README.md                   ← This file
```

---

## Quick Start

### Option 1 — Streamlit App (Deployed)
👉 **https://your-app.streamlit.app** *(deploy using steps below)*

### Option 2 — Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → select `ReceiptIQ_v4.ipynb`
3. **Runtime → Run all**
4. Paste your OpenAI API key when asked
5. Upload your receipt image

### Option 3 — Local CLI
```bash
pip install -r requirements.txt
python receiptiq.py --image sample_invoice.png --api-key YOUR_OPENAI_KEY
```

---

## Deploy to Streamlit Cloud (Free — for Deployed Link)
1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Add `OPENAI_API_KEY` in Secrets (optional)
6. Click Deploy — your link is live in 2 minutes

---

## RunAnywhere SDK
```python
from runanywhere import RunAnywhere
app = RunAnywhere(app_name="ReceiptIQ", on_device=True, privacy_mode=True)
```

---

## Tech Stack
| Component | Technology | Runs |
|-----------|------------|------|
| OCR Engine | EasyOCR (EN+HI) | On-Device |
| Image Processing | OpenCV | On-Device |
| AI Extraction | GPT-4o JSON mode | Cloud (text only) |
| Deployment SDK | RunAnywhere | On-Device |
| Web App | Streamlit | Deployed |

---

## Hackathon
**The GPT Challenge 2026 · Unstop · Deadline: 12 March 2026**
See `SUBMISSION.md` for full checklist and `DEMO_VIDEO_SCRIPT.md` for the 2-minute video.

---
MIT License
