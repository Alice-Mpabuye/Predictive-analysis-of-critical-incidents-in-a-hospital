# Sepsis survival prediction

This project focuses on the `predictive analysis of hospital critical incidents` using data science and machine learning techniques.
The main objective is to concentrate on cases of `sepsis` observed in a hospital in order to develop a predictive model capable of estimating the probability of a patient being affected by this critical condition.

For this project, we are using the dataset available at the following address: `https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records`

Technologies:
- Python
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Streamlit

---

### medical context

Sepsis is a life-threatening condition that occurs when the immune system reacts excessively to an infection, leading to organ dysfunction.
Sepsis is one of the most common causes of death worldwide.


---
## Project Organization
The analysis is organized into several Jupyter notebooks, each corresponding to a stage in the data pipeline.
The raw data is explored separately, cleaned in a dedicated notebook, and then saved as clean CSV files used for further analysis.

---
## Data exploration (`01_data_exploration.ipynb`)

### Missing values
No missing values were detected in the three cohorts (primary, study, validation).

### Duplicates
Repeated rows were observed within each cohort. There were 108,693 strict duplicates in the primary cohort, but this is not necessarily an error.

### Cleaning decisions
It was decided to apply intra-cohort cleaning, identical for each dataset, in order to avoid any interference between cohorts and to ensure the validity of subsequent analyses.

## Data cleanning (`02_data_cleanning.ipynb`)

Intra-cohort cleaning

Cleaning was performed independently on each cohort. The rules applied include:

- removal of exact duplicates
- filtering of observations that do not comply with consistency limits (age, binary values)

After cleaning, the primary cohort went from 110,204 to 1,511 observations.
No deletions were made between cohorts in order to preserve the independence of the sets.
Due to the absence of a unique identifier per patient, it is not possible to trace each deleted line individually.

## Exploratory Data Analysis and Visualization (`03_EDA+visualisation.ipynb`)

This notebook explores the cleaned datasets through descriptive statistics and visualizations.

Main analyses:

- Age distribution analysis (histograms, boxplots)
- Comparison between survivors and non-survivors
- Target variable distribution and class imbalance analysis
- Correlation analysis between numerical variables
- Preliminary outlier inspection
- Targeted scatter plots to assess relationships between key variables

This step provides insights into the data structure, distributions, and potential predictive signals, and informs modeling decisions.

## Anomaly detection (`04_Anomaly_detection.ipynb`)

This notebook investigates potential anomalous observations using both machine learning–based and statistical approaches.

Methods applied:
- Isolation Forest for multivariate anomaly detection
- Hyperparameter sensitivity and stability analysis (Jaccard similarity)
- Visualization of detected anomalies and anomaly scores
- Statistical outlier detection using Z-score and IQR methods
- Discussion of anomaly treatment strategies (winsorization vs. deletion)

This phase aims to identify extreme or rare cases and justify appropriate handling strategies prior to modeling.

## Predictive modeling (`05_Predictive_model.ipynb`)

This notebook builds and evaluates a predictive model for hospital outcome.

Key steps:
- Train/test split with class stratification
- Random Forest classifier training
- Model evaluation using precision, recall, F1-score, and confusion matrix
- Handling of class imbalance
- Integration of preprocessing and modeling into a unified pipeline
- Model persistence using joblib

The final output is a trained and serialized prediction pipeline, ready for inference.

## Dashboard UX (`src/`)

The final stage focuses on model deployment and user interaction.
Components:
- `preprocessing.py`: contain the function `IQRWinsorizer` whos's doing the winsorization
- `inference.py`: loads the trained pipeline and exposes a predict() function
- `dashboard.py`: Streamlit dashboard for:
  - displaying dataset statistics and EDA graphs
  - performing real-time predictions based on user input
  - presenting results in an accessible, interactive format
This step bridges data science and practical application, demonstrating how the model can be used in a real-world setting.


