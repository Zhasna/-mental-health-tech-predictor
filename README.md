# Mental Health in Tech — Treatment Predictor

A machine learning project that predicts whether someone in the tech 
industry is likely to seek mental health treatment, based on the 
OSMI Mental Health in Tech Survey dataset (2014–2019).

## Live Demo
https://osmimentalhealthtreatmentpredictor.streamlit.app/

## Project Overview
Mental health in the workplace is an important but often overlooked issue 
in the tech industry. This project builds a classification model to predict 
treatment-seeking behavior and identifies which factors matter most.

**Key finding:** Personal mental health history (prior diagnosis, past 
disorders) is a far stronger predictor of seeking treatment than workplace 
factors like employer benefits or company size.

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 84% |
| Random Forest | 85% |

Random Forest was selected as the final model.

## Top Predictors
1. Ever been diagnosed with a mental health disorder
2. Had a mental health disorder in the past
3. How often disorder interferes with work when treated
4. How often disorder interferes with work when untreated
5. Currently have a mental health disorder

## Tech Stack
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- SQLite

## Dataset
OSMI Mental Health in Tech Survey — available on Kaggle.
Covers 4,218 respondents across 5 survey years (2014–2019).

## Author
Hasna Fahima Zahir Hussain — B.Tech Information Technology, Easwari Engineering College
https://github.com/Zhasna
www.linkedin.com/in/hasna-fahima15
