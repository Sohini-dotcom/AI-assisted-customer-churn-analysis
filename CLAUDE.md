# Churn Analytics Project

## About
Portfolio project. I am a beginner, so explain things in plain English.
Goal: analyze customer churn with pandas, build a Power BI dashboard,
then add an AI layer (LLM API, ML model, RAG).
Dataset: data/raw/Telco-Customer-Churn.csv (7043 rows, 21 columns,
IBM sample telecom data from Kaggle). Target column: Churn (Yes/No).

## Rules
- NEVER modify anything in data/raw/
- Save cleaned data to data/processed/
- Write simple, well-commented Python using pandas
- Propose a plan and wait for my approval before big changes
- Ask before installing any new library
- Scripts go in src/, charts and reports go in outputs/
- Use the Python inside the venv folder to run scripts
  except as templates for script structure.

## Cleaning decisions already made
- Final cleaned dataset: a stratified random sample of 2000 rows
  (stratify on Churn, random_state=42), with exactly these 12 columns:
  customerID, SeniorCitizen, Dependents, tenure, InternetService,
  OnlineSecurity, TechSupport, Contract, PaymentMethod, MonthlyCharges,
  TotalCharges, Churn
- TotalCharges is text in the raw file, with blank values for
  customers with tenure 0. Investigate these and propose a fix.