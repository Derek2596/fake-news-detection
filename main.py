from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import torch
import torch.nn as nn
import pickle

app = FastAPI(title="Fake News Classifier API")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join("static", "index.html"))

# ---- Load vectorizer ----
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---- Model definitions must match the one used in training ----
class NewsClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)
    
class ComplexNewsClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.out = nn.Linear(prev_dim, 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)
    
class BasicNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)
    
model_files = {
    "Default": ("model.pt", NewsClassifier),
    "Basic": ("model_basic.pt", BasicNewsClassifier),
    "Complex": ("model_complex.pt", ComplexNewsClassifier)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {}
for name, (file, ModelClass) in model_files.items():
    m = ModelClass(input_dim=len(vectorizer.get_feature_names_out()))
    m.load_state_dict(torch.load(file, map_location=device))
    m.eval()
    models[name] = m

# ---- Input schema ----
class Item(BaseModel):
    text: str

# ---- Predict endpoint ----
@app.post("/predict")
def predict(item: Item):
    # TF-IDF Transform
    X = vectorizer.transform([item.text]).toarray()
    x_tensor = torch.tensor(X, dtype=torch.float32)

    results = []
    with torch.no_grad():
        for name, model in models.items():
            logits = model(x_tensor)
            pred = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
            results.append({
                "model": name,
                "prediction": "fake" if pred == 0 else "real",
                "confidence": round(confidence, 3)
            })

    return {"results": results}
