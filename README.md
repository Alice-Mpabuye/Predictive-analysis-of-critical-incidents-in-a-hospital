# Predictive Analysis of Critical Incidents in a Hospital

Work in progress.

This project focuses on predictive analysis of hospital critical incidents
using data analysis and machine learning techniques.

Technologies:
- Python
- Pandas
- Scikit-learn
- numpy
- matplotlib


---

The analysis is organized into several Jupyter notebooks, each corresponding to a stage in the data pipeline.
The raw data is explored separately, cleaned in a dedicated notebook, and then saved as clean CSV files used for further analysis.

## Data exploration

### Missing values
No missing values were detected in the three cohorts (primary, study, validation).

### Duplicates
Repeated rows were observed within each cohort. There were 108,693 strict duplicates in the primary cohort, but this is not necessarily an error.

### Cleaning decisions
It was decided to apply intra-cohort cleaning, identical for each dataset, in order to avoid any interference between cohorts and to ensure the validity of subsequent analyses.

## Data cleanning

Intra-cohort cleaning

Cleaning was performed independently on each cohort. The rules applied include:

- removal of exact duplicates
- filtering of observations that do not comply with consistency limits (age, binary values)

After cleaning, the primary cohort went from 110,204 to 1,511 observations.
No deletions were made between cohorts in order to preserve the independence of the sets.
Due to the absence of a unique identifier per patient, it is not possible to trace each deleted line individually.


