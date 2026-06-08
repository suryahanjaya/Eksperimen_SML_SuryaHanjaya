# Eksperimen_SML_SuryaHanjaya
## Dicoding SMSML — Kriteria 1: Eksperimen & Preprocessing Automation

---

## 📁 Struktur

```
Eksperimen_SML_SuryaHanjaya/
├── .github/
│   └── workflows/
│       └── preprocessing.yml       ← GitHub Actions (trigger push + manual)
├── dataset_raw/
│   └── adult.data                  ← Raw UCI Adult Income Dataset
└── preprocessing/
    ├── Eksperimen_SuryaHanjaya.ipynb  ← Notebook eksperimen (EDA + preprocessing)
    ├── automate_SuryaHanjaya.py       ← Script otomasi preprocessing
    ├── adult_income_preprocessed.csv  ← Hasil preprocessing
    └── scaler.pkl                     ← Fitted StandardScaler
```

---

## ⚙️ Cara Menjalankan

```bash
pip install pandas numpy scikit-learn joblib
python preprocessing/automate_SuryaHanjaya.py
```

## 🔗 Link

- **GitHub Actions**: https://github.com/suryahanjaya/Eksperimen_SML_SuryaHanjaya/actions
