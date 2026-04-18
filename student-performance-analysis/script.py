import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(r"E:\Git\datasets\Student_performance_data _.csv")

# Create a 2x2 grid of subplots, 14 inches wide and 10 inches tall for the visualization of all graphs 
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# Title
fig.suptitle("Student Performance Analysis", fontsize=16, fontweight='bold')

# Plot 1 (top-left) - Scatter plot showing relationship between study time of student and their GPA
sns.scatterplot(x='StudyTimeWeekly', y='GPA', data=df, ax=axes[0, 0])
axes[0, 0].set_title("Study Time vs GPA")  # Set subplot title
axes[0, 0].set_xlabel("Weekly Study Hours") # Set x-axis label

# Plot 2 (top-right) - Scatter plot showing relationship between absences and GPA
sns.scatterplot(x='Absences', y='GPA', data=df, ax=axes[0, 1])
axes[0, 1].set_title("Attendance vs GPA")

# Plot 3 (bottom-left) - Box plot comparing GPA distribution between genders (0:Female, 1:Male)
sns.boxplot(x='Gender', y='GPA', data=df, ax=axes[1, 0])
axes[1, 0].set_title("Gender vs GPA")

# Plot 4 (bottom-right) - Histogram showing how study hours are distributed
sns.histplot(df['StudyTimeWeekly'], bins=10, ax=axes[1, 1])
axes[1, 1].set_title("Study Hours Distribution")

# Automatically adjust spacing between subplots to avoid overlap
plt.tight_layout()

plt.show()
