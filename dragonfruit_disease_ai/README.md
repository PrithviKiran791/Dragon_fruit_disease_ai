# 🌵 Dragon Fruit Disease Advisory Chatbot with XAI

A vision-based disease advisory system that combines:
- **CNN Classification** (EfficientNet / ResNet) for disease detection
- **Grad-CAM XAI** to highlight infected visual regions
- **Knowledge Database** linking symptoms to pathogens
- **Conversational Chatbot** for treatment recommendations

## Pipeline
```
User uploads image → Preprocessing → CNN Model → Disease Prediction
→ Grad-CAM Heatmap → Knowledge DB → Chatbot Recommendation
```

## Project Structure
```
dragonfruit_disease_ai/
├── dataset/
│   ├── train/          # Training images (per disease class folder)
│   ├── validation/     # Validation images
│   └── test/           # Test images
├── models/             # Saved model weights (.h5 / .pt)
├── xai/                # Grad-CAM XAI scripts
├── chatbot/            # Knowledge DB + recommendation logic
├── app/                # Flask/Streamlit web application
└── results/            # Prediction outputs, heatmaps, reports
```

## Disease Classes (Expected)
- `Healthy`
- `Anthracnose` (Colletotrichum spp.)
- `Stem_Canker` (Neoscytalidium dimidiatum)
- `Fruit_Rot`
- `Brown_Spot`

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python app/app.py
```
