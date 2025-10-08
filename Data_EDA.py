#  EDA – Phase 1: Univariate Analysis

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


#  Target variable distribution
plt.figure()
sns.countplot(data=df, x='fraudulent', palette='Set2')
plt.title("Target Variable: Real vs Fake Job Postings")
plt.xticks([0, 1], ['Real (0)', 'Fake (1)'])
plt.ylabel("Count")
plt.xlabel("Fraudulent")
plt.show()


#  Top 10 Job Titles
plt.figure()
top_titles = df['title'].value_counts().head(10)
sns.barplot(x=top_titles.values, y=top_titles.index, palette='Blues_d')
plt.title("Top 10 Most Common Job Titles")
plt.xlabel("Frequency")
plt.ylabel("Job Title")
plt.show()


#  Employment Type
plt.figure()
sns.countplot(data=df, y='employment_type', order=df['employment_type'].value_counts().index, palette='viridis')
plt.title("Employment Type Distribution")
plt.xlabel("Count")
plt.ylabel("Employment Type")
plt.show()


#  Education
plt.figure()
sns.countplot(data=df, y='required_education', order=df['required_education'].value_counts().index, palette='mako')
plt.title("Required Education Distribution")
plt.xlabel("Count")
plt.ylabel("Education Level")
plt.show()

#  Telecommuting vs Onsite
plt.figure()
sns.countplot(data=df, x='telecommuting', palette='pastel')
plt.title("Telecommuting Distribution")
plt.xticks([0, 1], ['Onsite (0)', 'Remote (1)'])
plt.xlabel("Telecommuting")
plt.ylabel("Count")
plt.show()

#  Has Company Logo
plt.figure()
sns.countplot(data=df, x='has_company_logo', palette='coolwarm')
plt.title("Job Postings with vs. without Company Logo")
plt.xticks([0, 1], ['No Logo (0)', 'Has Logo (1)'])
plt.xlabel("Company Logo Presence")
plt.ylabel("Count")
plt.show()


# EDA Phase 2 – Bivariate Analysis: Feature vs Fraudulent

# Load seaborn & matplotlib again if needed
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# 1. Employment Type vs Fraudulent
plt.figure()
sns.countplot(data=df, x='employment_type', hue='fraudulent', palette='Set2')
plt.title("Employment Type vs Fraudulent")
plt.xlabel("Employment Type")
plt.ylabel("Count")
plt.legend(title="Fraudulent", labels=["Real", "Fake"])
plt.show()

# 2. Required Education vs Fraudulent
plt.figure()
sns.countplot(data=df, x='required_education', hue='fraudulent', palette='Set1')
plt.title("Required Education vs Fraudulent")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Education")
plt.ylabel("Count")
plt.legend(title="Fraudulent", labels=["Real", "Fake"])
plt.show()

# 3. Has Company Logo vs Fraudulent
plt.figure()
sns.countplot(data=df, x='has_company_logo', hue='fraudulent', palette='coolwarm')
plt.title("Company Logo Presence vs Fraudulent")
plt.xticks([0, 1], ['No Logo (0)', 'Has Logo (1)'])
plt.xlabel("Company Logo")
plt.ylabel("Count")
plt.legend(title="Fraudulent", labels=["Real", "Fake"])
plt.show()

# 4. Telecommuting vs Fraudulent
plt.figure()
sns.countplot(data=df, x='telecommuting', hue='fraudulent', palette='pastel')
plt.title("Remote Jobs vs Fraudulent")
plt.xticks([0, 1], ['Onsite (0)', 'Remote (1)'])
plt.xlabel("Telecommuting")
plt.ylabel("Count")
plt.legend(title="Fraudulent", labels=["Real", "Fake"])
plt.show()

# 5. Has Questions vs Fraudulent
plt.figure()
sns.countplot(data=df, x='has_questions', hue='fraudulent', palette='mako')
plt.title("Screening Questions vs Fraudulent")
plt.xticks([0, 1], ['No (0)', 'Yes (1)'])
plt.xlabel("Has Screening Questions")
plt.ylabel("Count")
plt.legend(title="Fraudulent", labels=["Real", "Fake"])
plt.show()
