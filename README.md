# Life Expectancy Prediction

A Python data-mining / machine-learning project reconstructed from the CSE 4120 lab report.

## Project contents

- `main.py` — dataset exploration, EDA, preprocessing, Linear Regression and Decision Tree Regression.
- `requirements.txt` — Python dependencies.
- `Life Expectancy Data.csv` — **not included**; add the dataset file yourself.
- `outputs/` — generated plots and model comparison results (created when you run the script).

## 1. Requirements

- Python 3.9 or newer
- The WHO Life Expectancy dataset CSV, named exactly `Life Expectancy Data.csv`

Download the Life Expectancy dataset used in the lab report and put the CSV in the same folder as `main.py`.

## 2. Open the project

Open this folder in PyCharm or VS Code.

## 3. Create and activate a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
```

## 4. Install libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Run

Make sure `Life Expectancy Data.csv` is beside `main.py`, then run:

```bash
python main.py
```

The script prints dataset information and model metrics. It creates an `outputs` folder containing:
- histograms
- boxplots before outlier treatment
- correlation heatmap
- actual-vs-predicted plots for both models
- `model_comparison.csv`

## Notes about corrections made during reconstruction

The PDF contains some incomplete or malformed code fragments (for example, missing parentheses and a broken KNN-imputer assignment). The runnable version repairs those issues.

It also:
- fits preprocessing on training data rather than the full dataset, reducing data leakage;
- keeps the target (`Life expectancy`) in years instead of scaling it, so reported errors are interpretable in years;
- uses the Decision Tree's predicted values in its own plot;
- applies `max_depth=5`, as described in the report's procedure.

Because of these corrections, the metrics will not necessarily match the values printed in the PDF. The report's Linear Regression and Decision Tree scores were calculated on a normalized target, while this reconstructed script evaluates the target in years.

## GitHub upload

Create a new empty repository on GitHub. Then, from this project folder, run:

```bash
git init
git add main.py requirements.txt README.md .gitignore
git commit -m "Add life expectancy prediction project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/life-expectancy-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use the repository URL shown by GitHub.

If you want to upload the dataset too, check its license/redistribution terms first. If you do not upload it, keep the CSV locally; the code will run on your computer but GitHub visitors will need to obtain the dataset separately.
