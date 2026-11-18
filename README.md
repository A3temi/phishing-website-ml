# Machine Learning-Based Phishing Website Classification

Group project for the Machine Learning assignment. The project uses the UCI PhiUSIIL Phishing URL (Website) Dataset to compare supervised models, apply K-Means clustering, and evaluate cross-validation, hyperparameter tuning, and feature selection.

## Repository contents

- `Phishing_Website_Classification_and_Clustering.ipynb` — main analysis notebook
- `generate_assignment_outputs.py` — executes the notebook and creates the curated report artifacts
- `PhiUSIIL_Phishing_URL_Dataset.csv` — project dataset
- `requirements.txt` — Python dependencies
- `outputs_phishing/` — generated figures, tables, and trained models

The DOCX assignment and proposal files are intentionally excluded from version control.

## Run the notebook

```bash
python -m venv .venv
```

Activate the environment, install dependencies, and launch Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook
```

Open the notebook and run all cells. The final notebook section creates the curated report artifacts in `outputs_phishing/`. Alternatively, run `python generate_assignment_outputs.py`. Keep the CSV in the repository root. The notebook uses an offline labelled dataset and does not visit websites.

## Collaboration workflow

```bash
git pull
git checkout -b your-name/change-description
# make and test changes
git add .
git commit -m "Describe the change"
git push -u origin your-name/change-description
```

Use separate branches for changes and do not commit virtual environments or DOCX files.

## Dataset note

The dataset includes precomputed URL and webpage-derived features. Because `URLSimilarityIndex` is highly discriminative in this benchmark, the report should distinguish full-feature benchmark results from leakage-aware or reduced-feature results.

Dataset source: [UCI PhiUSIIL Phishing URL (Website) Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bwebsite%2Bdataset)
