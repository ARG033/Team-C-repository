# Backend Integration Guide - Fake Review Detection

Quick start guide for integrating the fake review detection system.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start (3 Lines)

```python
from integration.api_interface import ReviewAnalyzer

analyzer = ReviewAnalyzer()  # Initialize once when server starts
result = analyzer.predict("Great product!")  # Analyze review
```

---

## API Response Format

```python
{
    'success': True,
    'prediction': 'FAKE',           # or 'REAL'
    'confidence': 0.8734,            # 0.0 to 1.0
    'explanation': 'Sentiment: Genuine reviews show balanced emotions; Length: Genuine reviews provide more detail',
    'details': {
        'model_verdict': 'FAKE',
        'xai_verdict': 'LIKELY FAKE',
        'flags': 3,
        'agreement': True            # Do model and XAI agree?
    }
}
```

---

## Flask Integration

```python
from flask import Flask, request, jsonify
from integration.api_interface import ReviewAnalyzer

app = Flask(__name__)
analyzer = ReviewAnalyzer()  # Initialize once at startup

@app.post('/predict')
def predict():
    review = request.json.get('review', '')
    result = analyzer.predict(review)
    return jsonify(result)

@app.get('/health')
def health():
    return jsonify(analyzer.health_check())

@app.get('/info')
def info():
    return jsonify(analyzer.get_model_info())
```

---

## FastAPI Integration

```python
from fastapi import FastAPI
from pydantic import BaseModel
from integration.api_interface import ReviewAnalyzer

app = FastAPI()
analyzer = ReviewAnalyzer()  # Initialize once at startup

class ReviewRequest(BaseModel):
    review: str

@app.post("/predict")
def predict(request: ReviewRequest):
    result = analyzer.predict(request.review)
    return result

@app.get("/health")
def health():
    return analyzer.health_check()

@app.get("/info")
def info():
    return analyzer.get_model_info()
```

---

## Error Handling

The API handles errors gracefully:

```python
# Empty input
result = analyzer.predict("")
# Returns: {'success': False, 'error': 'Review text cannot be empty'}

# Invalid input
result = analyzer.predict(None)
# Returns: {'success': False, 'error': 'Review text must be a string'}

# Analysis error
result = analyzer.predict("review text")
# Returns: {'success': False, 'error': 'Analysis failed: ...'}
```

Always check `result['success']` before using predictions.

---

## Available Methods

### `predict(review_text)`
Analyze a single review. **Main method.**

**Returns:** Complete analysis with prediction and explanation

---

### `get_model_info()`
Get model metadata (for `/info` endpoint).

**Returns:** Model name, features, labels, output format

---

### `health_check()`
Quick health check (for `/health` endpoint).

**Returns:** `{'status': 'healthy', 'model_loaded': True}`

---

## Performance Notes

- **First request**: ~2-3 seconds (model loading from HF Hub)
- **Subsequent requests**: ~100-300ms per review
- **Memory**: ~2GB RAM (with model loaded)
- **GPU**: Optional (auto-detected if available)

---

## Example cURL Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "AMAZING!!! Best product ever!!!"}'
```

**Response:**
```json
{
  "success": true,
  "prediction": "FAKE",
  "confidence": 0.9234,
  "explanation": "Sentiment: Genuine reviews show balanced emotions; Capitalization: Indicates emotional exaggeration",
  "details": {
    "model_verdict": "FAKE",
    "xai_verdict": "LIKELY FAKE",
    "flags": 3,
    "agreement": true
  }
}
```

---

## Troubleshooting

### Model Loading Error
```
Error: Could not load model from HuggingFace Hub
```
**Solution:** Check internet connection, verify model checkpoint: `ARG33/DistilBERT-finetuned`

### Memory Error
```
Error: CUDA out of memory
```
**Solution:** Model will automatically fall back to CPU if GPU memory is insufficient

### Import Error
```
ModuleNotFoundError: No module named 'integration'
```
**Solution:** Ensure you're running from project root directory

---

## Support

For issues or questions, contact the ML team or check:
- Model checkpoint: `ARG33/DistilBERT-finetuned`
- Project documentation in `docs/` folder
- Test files in `tests/` folder for usage examples

---

**That's it! Integration should take less than 10 minutes.** 🚀