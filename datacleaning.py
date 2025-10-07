
# BATCH 1 – Cleaning First 6 Columns
# Columns: job_id, title, location, department, salary_range, company_profile

import pandas as pd

# Load dataset
df = pd.read_csv("fake_job_postings.csv")


# 1. job_id
# What: Drop this column
# Why: It's just a unique identifier and has no predictive value
# How: Use df.drop()
df.drop(columns=["job_id"], inplace=True)


# 2. title
# What: Clean the job title text
# Why: Case consistency and text normalization
# How: Lowercase + strip spaces + remove punctuation
df['title'] = df['title'].astype(str).str.lower().str.strip()


# 3. location
# What: Extract useful info from messy location strings
# Why: Some locations are missing city or have inconsistent formatting
# How:
#   - Extract country (first part before the first comma)
#   - Optionally extract state
df['country'] = df['location'].astype(str).str.split(',').str[0].str.strip().str.upper()
df['state'] = df['location'].astype(str).str.split(',').str[1].str.strip().str.upper()

# You can drop original location column after extracting
df.drop(columns=['location'], inplace=True)


# 4. department
# What: Drop this column
# Why: Often missing or vague (e.g., "tech", "general"), low information value
# How: drop it
df.drop(columns=['department'], inplace=True)


# 5. salary_range
# What: Create a new feature 'has_salary'
# Why: The salary column is too sparse and inconsistent to fill or convert,
#         but its presence or absence can be a useful indicator
# How:
df['has_salary'] = df['salary_range'].notnull().astype(int)
# Optional: drop original column
df.drop(columns=['salary_range'], inplace=True)


# 6. company_profile
# What: Drop this column
# Why: Usually empty and not directly useful for modeling
df.drop(columns=['company_profile'], inplace=True)


# BATCH 1 CLEANED! Preview and save
print("Batch 1 cleaned. Preview:")
print(df.head())
df.to_csv("cleaned_batch1.csv", index=False)

# BATCH 2 – Cleaning Text + Categorical Fields

import re


# 1. description
# What: Clean and standardize job descriptions
# Why: Core signal for model — remove noise, standardize input
# How:
df['description'] = (
    df['description'].astype(str)
    .str.lower()
    .str.replace(r'<.*?>', '', regex=True)   # Remove HTML tags
    .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)  # Remove punctuation
    .str.replace(r'\s+', ' ', regex=True)   # Remove extra spaces
    .str.strip()
)


# 2. requirements
# What: Clean like description (same reasons)
df['requirements'] = (
    df['requirements'].astype(str)
    .str.lower()
    .str.replace(r'<.*?>', '', regex=True)
    .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)


# 3. benefits
# What: Fill missing with 'not provided', clean formatting
# Why: Sparse, but presence/absence might hold signal
df['benefits'] = (
    df['benefits'].astype(str)
    .fillna("not provided")
    .str.lower()
    .str.replace(r'<.*?>', '', regex=True)
    .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)


# 4. employment_type
# What: Clean & normalize
# Why: Categorical feature useful for prediction
df['employment_type'] = (
    df['employment_type']
    .fillna("unknown")
    .str.lower()
    .str.strip()
)


# 5. required_experience
# What: Fill missing, normalize entries
df['required_experience'] = (
    df['required_experience']
    .fillna("not specified")
    .str.lower()
    .str.strip()
)


# 6. required_education
# What: Fill nulls, reduce to general levels
df['required_education'] = (
    df['required_education']
    .fillna("not specified")
    .str.lower()
    .replace({
        "high school or equivalent": "high school",
        "some college coursework completed": "some college",
        "bachelor's degree": "bachelor",
        "master's degree": "master",
        "doctorate": "phd"
    })
)


# BATCH 2 CLEANED! Preview and save
print("Batch 2 cleaned. Preview:")
print(df[['description', 'requirements', 'benefits', 'employment_type', 'required_experience', 'required_education']].head())

df.to_csv("cleaned_batch2.csv", index=False)

#BATCH 3 – Cleaning Meta Columns and Target


# 1. industry
# What: Fill nulls and standardize
# Why: May be useful for visualization (optional for model)
df['industry'] = (
    df['industry']
    .fillna("not specified")
    .str.lower()
    .str.strip()
)


# 2. function
# What: Clean like industry
df['function'] = (
    df['function']
    .fillna("not specified")
    .str.lower()
    .str.strip()
)


# 3. telecommuting
# What: Ensure binary 0 or 1, no nulls
df['telecommuting'] = df['telecommuting'].fillna(0).astype(int)


# 4. has_company_logo
# What: Ensure binary 0 or 1, no nulls
df['has_company_logo'] = df['has_company_logo'].fillna(0).astype(int)


# 5. has_questions
# What: Ensure binary 0 or 1, no nulls
df['has_questions'] = df['has_questions'].fillna(0).astype(int)


# 6. fraudulent (target)
# What: Validate only — should be 0 or 1
print("Target column check:\n", df['fraudulent'].value_counts())


# BATCH 3 CLEANED! Preview and save
print("Batch 3 cleaned. Preview:")
print(df[['industry', 'function', 'telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']].head())

# Save final cleaned dataset
df.to_csv("cleaned_final_dataset.csv", index=False)
