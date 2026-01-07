## ML-QSAR for PD-L1 Small Molecule Inhibitors

## 📌 Overview
Machine learning pipeline for predicting PD-L1 inhibitory activity using XGBoost.

## 🚀 How to Use
```bash
git clone https://github.com/YOUR-USERNAME/ML-QSAR-PD-L1-Inhibitors.git
cd ML-QSAR-PD-L1-Inhibitors
pip install pandas numpy rdkit scikit-learn xgboost
python qsar_pipeline.py
```

## 📂 Data Files

Training Data (mol1.txt):
info: small molecule inhibitors
Format: SMILES<TAB>pIC50
Example: 
``` bash
Cc1c(COc2ccc3nc(CN)c(NCc4ccccc4)n3c2)cccc1-c1ccccc1 4.99
Cc1c(COc2ccc3nc(CN)c(NC(C)c4ccccc4)n3c2)cccc1-c1ccccc1 5.02
...
```

Prediction Data (ligands.txt):
info: FDA approved drugs(until 2025)
Format: One SMILES per line
Example:
``` bash
O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O
CC1CCc2c(N3CCC(O)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23
...
```

## 📊 Expected Output
  Console: Model performance (R², MSE) and predictions
  File: Saved model as best_xgboost_model.pkl
  Plot: Feature importance visualization


## 📧 Contact
a.garmabdary@gmail.com




