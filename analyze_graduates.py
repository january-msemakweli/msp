import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Set styling for plots
plt.style.use('ggplot')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load the datasets
graduates_df = pd.read_csv('GRADUATES EMPLOYMENT STATUS_with_stem.csv')
prospective_df = pd.read_csv('PROSPECTIVE GRADUATES_with_stem.csv')

# Clean and prepare the data
graduates_df.columns = graduates_df.columns.str.strip().str.lower()
prospective_df.columns = prospective_df.columns.str.strip().str.lower()

# Basic info about the datasets
print("Graduates dataset shape:", graduates_df.shape)
print("Prospective graduates dataset shape:", prospective_df.shape)

# ==== Employment Status Analysis ====

# Count employment statuses
employment_status_counts = graduates_df['employment status'].value_counts()
print("\nEmployment Status Counts:")
print(employment_status_counts)

# Calculate percentage of STEM vs non-STEM
stem_counts = graduates_df['is_stem'].value_counts(normalize=True) * 100
print("\nPercentage of STEM vs non-STEM graduates:")
print(stem_counts)

# Cross-tabulate employment status and STEM field
employment_by_stem = pd.crosstab(
    graduates_df['employment status'], 
    graduates_df['is_stem'], 
    normalize='index'
) * 100
print("\nEmployment Status by STEM field (%):")
print(employment_by_stem)

# ==== Missing Data Analysis ====

# Check for missing values
missing_values = graduates_df.isnull().sum()
print("\nMissing Values in Graduates Dataset:")
print(missing_values[missing_values > 0])

# Count incomplete entries (entries with NaN in important fields)
important_fields = ['name', 'university attended', 'field of study', 'employment status', 'contacts', 'email adress']
incomplete_entries = graduates_df[graduates_df[important_fields].isnull().any(axis=1)]
print(f"\nNumber of incomplete entries: {len(incomplete_entries)}")

# For entries marked as Employed but missing organization or job title
employed_missing_details = graduates_df[
    (graduates_df['employment status'] == 'Employed') & 
    (graduates_df['organization/company/sector'].isnull() | graduates_df['job title'].isnull())
]
print(f"\nEmployed graduates missing organization or job title: {len(employed_missing_details)}")

# ==== Create Visualizations ====

# 1. Employment Status Distribution
plt.figure(figsize=(10, 6))
ax = employment_status_counts.plot(kind='bar', color=sns.color_palette("Set2"))
plt.title('Distribution of Employment Status Among Graduates', fontsize=16)
plt.xlabel('Employment Status', fontsize=14)
plt.ylabel('Number of Graduates', fontsize=14)
plt.xticks(rotation=0)

# Add count labels on top of bars
for i, count in enumerate(employment_status_counts):
    plt.text(i, count + 1, str(count), ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('employment_status_distribution.png', dpi=300, bbox_inches='tight')

# 2. Employment Status by STEM Field
plt.figure(figsize=(12, 7))
employment_by_stem_counts = pd.crosstab(graduates_df['employment status'], graduates_df['is_stem'])
employment_by_stem_counts.columns = ['Non-STEM', 'STEM']
ax = employment_by_stem_counts.plot(kind='bar', stacked=True)
plt.title('Employment Status by STEM vs Non-STEM Field', fontsize=16)
plt.xlabel('Employment Status', fontsize=14)
plt.ylabel('Number of Graduates', fontsize=14)
plt.xticks(rotation=0)
plt.legend(title='Field Type')

# Add count labels
for i, (_, row) in enumerate(employment_by_stem_counts.iterrows()):
    non_stem_count = row['Non-STEM']
    stem_count = row['STEM']
    total = non_stem_count + stem_count
    plt.text(i, non_stem_count/2, str(int(non_stem_count)), ha='center', va='center', color='white', fontsize=11)
    plt.text(i, non_stem_count + stem_count/2, str(int(stem_count)), ha='center', va='center', color='white', fontsize=11)
    plt.text(i, total + 1, f'Total: {total}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('employment_status_by_stem.png', dpi=300, bbox_inches='tight')

# 3. University Distribution of Graduates
university_counts = graduates_df['university attended'].value_counts().head(10)
plt.figure(figsize=(12, 7))
ax = university_counts.plot(kind='bar', color=sns.color_palette("Set2"))
plt.title('Top 10 Universities Attended by Graduates', fontsize=16)
plt.xlabel('University', fontsize=14)
plt.ylabel('Number of Graduates', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Add count labels
for i, count in enumerate(university_counts):
    plt.text(i, count + 0.5, str(count), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('university_distribution.png', dpi=300, bbox_inches='tight')

# 4. Graduation Year Distribution
plt.figure(figsize=(10, 6))
year_counts = graduates_df['graduation year'].value_counts().sort_index()
ax = year_counts.plot(kind='bar', color=sns.color_palette("Set2"))
plt.title('Distribution of Graduation Years', fontsize=16)
plt.xlabel('Graduation Year', fontsize=14)
plt.ylabel('Number of Graduates', fontsize=14)
plt.xticks(rotation=0)

# Add count labels
for i, count in enumerate(year_counts):
    plt.text(i, count + 0.5, str(count), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('graduation_year_distribution.png', dpi=300, bbox_inches='tight')

# 5. Gender Distribution by Employment Status
gender_employment = pd.crosstab(graduates_df['gender'], graduates_df['employment status'])
plt.figure(figsize=(12, 7))
ax = gender_employment.plot(kind='bar', stacked=False)
plt.title('Gender Distribution by Employment Status', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Number of Graduates', fontsize=14)
plt.xticks(rotation=0)
plt.legend(title='Employment Status')

# Add count labels for each category
for i, row in enumerate(gender_employment.itertuples()):
    gender = row[0]
    for j, count in enumerate(row[1:]):
        plt.text(i, count + 0.5, str(count), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('gender_employment_status.png', dpi=300, bbox_inches='tight')

# 6. Prospective Graduates - Career Interests
plt.figure(figsize=(10, 6))
career_counts = prospective_df['what career path are you most interested in after graduation?'].value_counts()
ax = career_counts.plot(kind='bar', color=sns.color_palette("Set2"))
plt.title('Career Interests of Prospective Graduates', fontsize=16)
plt.xlabel('Career Path', fontsize=14)
plt.ylabel('Number of Students', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Add count labels
for i, count in enumerate(career_counts):
    plt.text(i, count + 0.1, str(count), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('prospective_career_interests.png', dpi=300, bbox_inches='tight')

# 7. Prospective Graduates - Support Needed
# This is a multi-select column, so we need to process it differently
support_options = [
    'cv writing support', 'interview preparation', 'job search strategies', 
    'linkedin/profile branding', 'networking opportunities', 
    'entrepreneurship guidance', 'understanding job market expectations'
]

support_counts = {}
for option in support_options:
    support_counts[option] = prospective_df['what kind of support would you find most helpful right now? (check all that apply)'].str.contains(option, case=False).sum()

support_df = pd.DataFrame(list(support_counts.items()), columns=['Support Type', 'Count'])
support_df = support_df.sort_values('Count', ascending=False)

plt.figure(figsize=(12, 7))
ax = sns.barplot(x='Support Type', y='Count', data=support_df, palette="Set2")
plt.title('Support Needed by Prospective Graduates', fontsize=16)
plt.xlabel('Type of Support', fontsize=14)
plt.ylabel('Number of Students', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Add count labels
for i, count in enumerate(support_df['Count']):
    plt.text(i, count + 0.1, str(count), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('prospective_support_needed.png', dpi=300, bbox_inches='tight')

print("\nAnalysis complete. Visualizations saved as PNG files.") 