# Importing and Installing Libraries 

pip install wordcloud

## Importing All Libraries 

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools
from pathlib import Path
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
%matplotlib inline

# Loading Dataset

df = pd.read_csv('/kaggle/input/marketing-campaign/marketing_campaign.csv',sep='\t')
df.head()

# About Dataset

# ---------- Function: Display Basic Info ----------
def basic_info(df):
    print("BASIC INFO")
    print(f"→ Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\n→ Column List:", df.columns.tolist())

    print("\nDataFrame Info:")
    buffer = StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())

# ---------- Function: Summary Stats ----------
def summary_statistics(df):
    print("\nSUMMARY STATISTICS:")
    print(df.describe(include='all').T)

# ---------- Function: Missing Values ----------
def missing_value_analysis(df):
    print("\nMISSING VALUE ANALYSIS:")
    missing = df.isnull().sum().to_frame('Missing Count')
    missing['Missing %'] = (missing['Missing Count'] / len(df)) * 100
    print(missing[missing['Missing Count'] > 0])

# ---------- Function: Unique Value Count ----------
def unique_value_analysis(df):
    print("\nUNIQUE VALUES PER COLUMN:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

# ---------- Function: Correlation Heatmap ----------
def correlation_heatmap(df):
    print("\nCORRELATION HEATMAP:")
    plt.figure(figsize=(14, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Between Numerical Features")
    plt.tight_layout()
    plt.show()

# ---------- Function: Histograms ----------
def plot_histograms(df):
    print("\nHISTOGRAMS:")
    df.select_dtypes(include='number').hist(figsize=(15, 10), bins=30, edgecolor='black')
    plt.suptitle("Distribution of Numerical Features", fontsize=16)
    plt.tight_layout()
    plt.show()

# ---------- Function: Boxplots ----------
def plot_boxplots(df):
    print("\nBOXPLOTS FOR OUTLIER CHECK:")
    for col in df.select_dtypes(include='number').columns:
        plt.figure(figsize=(6, 1.5))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

# ---------- MAIN RUN ----------
if __name__ == "__main__":

    # Call each analysis step
    basic_info(df)
    summary_statistics(df)
    missing_value_analysis(df)
    unique_value_analysis(df)
    correlation_heatmap(df)
    plot_histograms(df)
    plot_boxplots(df)

# Analysis and Insights of Dataset

**Dataset Size:**
# The dataset contains 2,240 customer records and 29 features.

# **Data Types Overview:**
# a Most columns are numeric (25 out of 29).
# b 3 are categorical or string-based: Education, Marital_Status, and Dt_Customer.

# **Missing Data:**
# a Only Income has missing values (24 entries, ~1.07%).
# b Other fields are 100% complete.

# **Date Field:**
# a Dt_Customer likely represents the date when the customer enrolled.
# b Needs conversion to datetime format for time-based analysis (e.g., tenure).

# **Customer Demographics:**
# a Year_Birth spans a wide range — useful for identifying age groups.
# b Education and Marital_Status offer good segmentation potential.

# **Household Info:**
# a Kidhome and Teenhome indicate presence of children and teenagers at home.
# b Can be used to build a family size feature.

# **Monetary Columns:**
# a Features like MntWines, MntMeatProducts, etc., represent product category spending.
# b Some customers spend significantly more — worth exploring high-value customers.

# **Marketing Response:**
# a AcceptedCmp1 to AcceptedCmp5 show responses to past campaigns.
# b Response is a final campaign response flag (target variable in many cases).

# **Engagement Metrics:**
# a NumWebPurchases, NumCatalogPurchases, NumStorePurchases, and NumWebVisitsMonth indicate channel preference.
# b Important for channel effectiveness analysis.

# **Customer Recency:**
# a Recency shows days since last purchase — useful for RFM segmentation.

# **Dummy Variables:**
# a Z_CostContact and Z_Revenue are constants (zero variance) — likely placeholder fields and can be dropped.

# **Potential Feature Engineering:**
# a Age from Year_Birth.
# b Customer Tenure from Dt_Customer.
# c Total Children = Kidhome + Teenhome.
# d Total Spending = sum of all Mnt* columns.

# **Outlier Checks Needed:**
# a Boxplots show presence of high-spending outliers, especially in wine and meat purchases.

# **Categorical Columns:**
# a Education and Marital_Status have several categories — might benefit from grouping or encoding.

# **Overall Data Quality:**
# a Clean structure, minimal missing values, and rich feature set — well-suited for segmentation, prediction, and customer profiling.

# USE CASE ANALYSIS

# Step 1: Drop rows where 'Income' is missing
df_cleaned = df.dropna(subset=['Income'])

# Step 2: Convert 'Dt_Customer' column to datetime
df_cleaned['Dt_Customer'] = pd.to_datetime(df_cleaned['Dt_Customer'], errors='coerce')

# Step 3: Check if any conversion failed and got NaT 
invalid_dates = df_cleaned['Dt_Customer'].isna().sum()

# Display results
cleaning_summary = {
    "Original Row Count": len(df),
    "Row Count After Dropping Missing Income": len(df_cleaned),
    "Rows Dropped": len(df) - len(df_cleaned),
    "Invalid Dates After Conversion": invalid_dates
}

# Cleaning dataframe summary 
cleaning_summary_df = pd.DataFrame.from_dict(cleaning_summary, orient='index', columns=['Value'])

print(cleaning_summary_df)

invalid_dates

df_cleaned.head()

# Newest customer's enrolment date
newest_date = df_cleaned['Dt_Customer'].max()
print("Newest customer's enrolment date:", newest_date)

# Oldest customer's enrolment date
oldest_date = df_cleaned['Dt_Customer'].min()
print("Oldest customer's enrolment date:", oldest_date)

df_cleaned.describe()

# Normal Distribution of Age distribution 

# Get current year 
current_year = datetime.datetime.now().year

# Create Age column
df_cleaned['Age'] = current_year - df_cleaned['Year_Birth']

plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Age'], bins=30, kde=True)
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True)
plt.show()

**Summary of Age Distribution Analysis:**
# 1. The majority of customers fall between 40 and 70 years old, with a peak around 50–55 years.
# 2. There is a long tail on the right, indicating a small number of customers older than 80 — possibly data anomalies (e.g., age 100+).
# 3. The distribution is slightly left-skewed, suggesting more older customers than younger ones.

# Normal Distribution of Income distribution 

plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Income'].dropna(), bins=30, kde=True, color='orange')
plt.title('Distribution of Customer Income')
plt.xlabel('Income')
plt.ylabel('Count')
plt.grid(True)
plt.show()

**Summary of Income Distribution Analysis:**

# 1. The distribution is right-skewed (positively skewed), meaning:Most customers earn between 30,000 to 90,000.
# 2. A small number of customers have very high incomes (up to 600K), which are clear outliers.
# 3. The curve has one dominant peak, showing a mode in the mid-income range.

# Customer Education Level Visualisation

plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x='Education', order=df_cleaned['Education'].value_counts().index, palette='Set2')
plt.title('Distribution of Customer Education Level')
plt.xlabel('Education Level')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


**Summary of Education Level:**

# 1. Graduation is the most common education level among customers, with over 1,000 individuals — indicating that the majority of your customer base is well-educated.
# 2. PhD holders and Master’s degree holders form the next significant segments, showing a highly educated customer base overall.
# 3. 2n Cycle and Basic education customers represent a small minority, sugges

# Step 1: Define product spending columns
spending_columns = [
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]

# Step 2: Create Total_Spending column in the main DataFrame
df_cleaned['Total_Spending'] = df_cleaned[spending_columns].sum(axis=1)

# Aggregate mean income and spending by education
edu_grouped = df_cleaned.groupby('Education').agg({
    'Income': 'mean',
    'Total_Spending': 'mean'
}).reset_index()

# Sort education levels by income (optional)
edu_grouped = edu_grouped.sort_values(by='Income', ascending=False)

# Calculate difference for line plot (optional)
edu_grouped['Diff'] = edu_grouped['Income'] - edu_grouped['Total_Spending']

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Stacked bar
ax1.bar(edu_grouped['Education'], edu_grouped['Income'], label='Average Income', color='skyblue')
ax1.bar(edu_grouped['Education'], edu_grouped['Total_Spending'], 
        bottom=edu_grouped['Income'], label='Average Spending', color='orange')

ax1.set_xlabel('Education Level')
ax1.set_ylabel('Amount')
ax1.set_title('Average Income vs Total Spending by Education Level')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(loc='upper left')

# Line plot showing the difference
ax2 = ax1.twinx()
ax2.plot(edu_grouped['Education'], edu_grouped['Diff'], color='red', marker='o', label='Income - Spending')
ax2.set_ylabel('Difference (Income - Spending)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

Summary of Income Vs Total Spending by Education Level Distribution Analysis:

# PhD, Master, and Graduation holders have similarly high average incomes, all above 48,000+, while their spending remains relatively modest, creating a large income-to-spending gap.
# Customers with 2n Cycle education earn less, and their spending also drops accordingly — though the gap remains significant.
# Basic education customers have the lowest income (~20,000) and least spending, with a much smaller margin between earnings and expenses.
# The red line (Income – Spending) shows a clear downward trend from PhD to Basic, highlighting decreasing savings capacity or affordability with lower education levels.

# Distribution of Number of Children

# Step 1: Create total number of children column
df_cleaned['Total_Children'] = df_cleaned['Kidhome'] + df_cleaned['Teenhome']

# Step 2: Create countplot
plt.figure(figsize=(8, 5))
sns.countplot(x='Total_Children', data=df_cleaned, palette='viridis', order=sorted(df_cleaned['Total_Children'].unique()))
plt.title('Distribution of Number of Children in Household', fontsize=14)
plt.xlabel('Total Number of Children (Kids + Teenagers)', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

Summary of Income Vs Total Spending by Education Level Distribution Analysis:

# 632 customers have no child.
# Most of the customers have one child.
# 416 customers have two child.
# 50 customers have three child.

# Customer Product Preference

# Define product category columns
product_cols = [
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]

# Sum spending per category
category_sales = {col.replace('Mnt', ''): df_cleaned[col].sum() for col in product_cols}

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set2')
wordcloud.generate_from_frequencies(category_sales)

# Plotting 
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Product Category Popularity Based on Total Spending', fontsize=14)
plt.tight_layout()
plt.show()

# Complaints Distribution on Product Category

# Step 1: Prepare the data
product_cols = [
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]

categories = []
sales = []
complaints = []

for col in product_cols:
    category = col.replace('Mnt', '')
    buyers = df_cleaned[df_cleaned[col] > 0]
    
    categories.append(category)
    sales.append(len(buyers))
    complaints.append(len(buyers[buyers['Complain'] == 1]))

df_plot = pd.DataFrame({
    'Product_Category': categories,
    'Sales': sales,
    'Complaints': complaints
})

# Sort by total sales for better visual flow
df_plot = df_plot.sort_values(by='Sales', ascending=False)

# Melt for seaborn
df_melted = df_plot.melt(id_vars='Product_Category', 
                         value_vars=['Sales', 'Complaints'], 
                         var_name='Metric', 
                         value_name='Count')

# Step 2: Plot
plt.figure(figsize=(12, 6))
sns.set_style('whitegrid')
ax = sns.barplot(data=df_melted, x='Product_Category', y='Count', hue='Metric', palette='Paired')

# Step 3: Add value labels
for p in ax.patches:
    value = int(p.get_height())
    ax.annotate(f'{value}', (p.get_x() + p.get_width() / 2., value + 5),
                ha='center', va='center', fontsize=9, color='black', rotation=0)

# Step 4: Final styling
plt.title('Number of Sales vs Number of Complaints by Product Category', fontsize=14)
plt.xlabel('Product Category')
plt.ylabel('Number of Customers')
plt.ylim(0, max(df_plot['Sales']) + 50)  # scale tightly
plt.legend(title='')
plt.tight_layout()
plt.show()

# Step 1: Prepare the data
product_cols = [
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]

categories = []
complaint_percents = []

for col in product_cols:
    category = col.replace('Mnt', '')
    buyers = df_cleaned[df_cleaned[col] > 0]
    complaints = buyers[buyers['Complain'] == 1]
    
    if len(buyers) > 0:
        percent = (len(complaints) / len(buyers)) * 100
    else:
        percent = 0.0
    
    categories.append(category)
    complaint_percents.append(round(percent, 2))

# Create DataFrame
df_percent = pd.DataFrame({
    'Product_Category': categories,
    'Complaint_Percentage': complaint_percents
})

# Sort by complaint %
df_percent = df_percent.sort_values(by='Complaint_Percentage', ascending=False)

# Step 2: Plot
plt.figure(figsize=(10, 6))
sns.set_style('whitegrid')
ax = sns.barplot(data=df_percent, x='Product_Category', y='Complaint_Percentage', palette='coolwarm')

# Add value labels
for p in ax.patches:
    value = p.get_height()
    ax.annotate(f'{value:.1f}%', 
                (p.get_x() + p.get_width() / 2., value + 0.5),
                ha='center', va='bottom', fontsize=9)

# Final styling
plt.title('Complaint % by Product Category (Based on Buyers)', fontsize=14)
plt.xlabel('Product Category')
plt.ylabel('Complaint %')
plt.ylim(0, max(df_percent['Complaint_Percentage']) + 5)
plt.tight_layout()
plt.show()

**Summary of complaints% by product category**

# 1. GoldProds has the highest complaint rate at just 1.0%, followed closely by Wines, MeatProducts, SweetProducts, and Fruits — all hovering around 0.9%.
# 2. FishProducts has the lowest complaint rate, slightly below 0.8%.
# 3. The overall range of complaint percentages is very narrow, indicating uniformly high customer satisfaction across product categories.

# Step 1: Select only numeric columns
numeric_df = df_cleaned.select_dtypes(include='number')

# Step 2: Compute correlation matrix
corr_matrix = numeric_df.corr()

# Step 3: Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
            linewidths=0.5, cbar_kws={"shrink": 0.75})
plt.title('Correlation Matrix of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()

# Customer Segmentation

# Step 1: Select product spending columns
product_cols = [
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]

X = df_cleaned[product_cols]

# Step 2: Standardize the data
# Product spending columns (e.g., MntWines, MntGoldProds, etc.) have different scales — some range from 0–1000, others from 0–100.
# Without standardization, clustering algorithms like K-Means will give more weight to features with higher magnitude, even if they're not more important.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled

# Step 3: Use Elbow Method to choose k
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Step 4: Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Final clustering with chosen k 
# Choosing cluster with K = 4 
kmeans_final = KMeans(n_clusters=4, random_state=42)
df_cleaned['Product_Cluster'] = kmeans_final.fit_predict(X_scaled)

cluster_profiles = df_cleaned.groupby('Product_Cluster')[product_cols].mean()
cluster_profiles.index.name = 'Cluster'
print(cluster_profiles)

# Optional: Restore original scale for cluster centers
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans_final.cluster_centers_),
    columns=product_cols
)
cluster_centers.index.name = 'Cluster'

# Plot
cluster_centers.plot(kind='bar', figsize=(12, 6), colormap='tab10')
plt.title('Average Spending per Product Category by Cluster')
plt.ylabel('Average Spending')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Summary of Clusters

# **A. Cluster 0: "Wine & Meat Lovers"**
# 1. Very high spending on Wines (700) and MeatProducts (360)    
# 2. Moderate on Fish, Sweet, Gold 
# 3. Likely premium or loyal customers who enjoy quality food & drink 
# 4. Business Tip:
#     1. Target with exclusive wine/meat bundles 
#     2. Offer loyalty programs or premium club memberships

# **B. Cluster 1: "Low Spenders"**
# 1. Very low spending across all categories
# 2. Small spike only in Wines (~90)
# 3. Business Tip:
#     1. Engage with entry-level offers or discount triggers
#     2. Upsell bundles to encourage more multi-category purchases

# **C. Cluster 2: "All-Round Enthusiasts"**
# 1. High spenders across ALL categories
# 2. Especially strong in Wines, Meat, Fish, Sweets, and GoldProds
# 3. Business Tip:
#     1. Treat as top-tier VIP customers
#     2. Ideal segment for personalized marketing, early access, and                 cross-selling across all products

# **D. Cluster 3: "Gift Givers & Mixed Shoppers"**
# 1. High on Wines, GoldProds, and Meat
# 2. Moderate in all other categories
# 3. Business Tip:
#     1. Likely value giftable items
#     2. Focus on holiday campaigns, gift cards, and customizable bundles


# Income Spending of Clusters

cluster_income_spending = df_cleaned.groupby('Product_Cluster')[['Income', 'Total_Spending']].mean().reset_index()


# Set dark theme
plt.style.use('dark_background')

# Define custom color palette for better contrast on black
custom_palette = {
    0: '#FF6F61',  # Soft Red
    1: '#6B5B95',  # Purple
    2: '#88B04B',  # Green
    3: '#FFA500'   # Orange
}

plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=df_cleaned,
    x='Total_Spending',
    y='Income',
    hue='Product_Cluster',
    palette=custom_palette,
    alpha=0.85,
    s=70,
    edgecolor='white'
)

plt.title('Income-Spending Basis Clustering Profile', fontsize=15, weight='bold', color='white')
plt.xlabel('Total Spending', fontsize=12, color='white')
plt.ylabel('Income', fontsize=12, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(title='Cluster', title_fontsize='13', loc='best', facecolor='black', edgecolor='white')
plt.tight_layout()
plt.show()

# Summary of Income Spending of Clusters

# 1. Cluster 0 (Red):
#     a. Wide range of spending, with moderate to high spending
#     b. Moderate to high income levels
#     c. Includes some outliers with very high spending
#     d. Likely your core revenue-generating segment

# 2. Cluster 1 (Purple):
#     a. Densely packed at the low income and low spending region
#     b. Includes several extremely low spenders
#     c. Possibly new customers, low engagement, or budget buyers

# 3. Cluster 2 (Green):
#     a. Spread horizontally at higher spending ranges
#     b. Moderate to high income
#     c. Indicates value-focused but active buyers
#     d. Likely loyal shoppers who respond well to bundles or cross-sells

# 4. Cluster 3 (Yellow):
#     a. Relatively low income but moderate spending
#     b. May include customers who spend beyond their means or value your products highly
#     c. Might respond well to discounts or installment options

# Share of Orders Leveraging Discounts

# Getting TotalPurchases and Avoid division by zero
df_cleaned['TotalPurchases'] = (
    df_cleaned['NumWebPurchases'] +
    df_cleaned['NumCatalogPurchases'] +
    df_cleaned['NumStorePurchases']
)

df_cleaned['Pct_Discount_Buy'] = np.where(
    df_cleaned['TotalPurchases'] > 0,
    (df_cleaned['NumDealsPurchases'] / df_cleaned['TotalPurchases']) * 100,
    0
)

# Step 2: Get average % per cluster
discount_pct_cluster = df_cleaned.groupby('Product_Cluster')['Pct_Discount_Buy'].mean().round(2)
labels = [f"Cluster {i}" for i in discount_pct_cluster.index]
values = discount_pct_cluster.values

# Step 3: Create donut chart
colors = sns.color_palette("Set3", len(labels))
plt.figure(figsize=(10, 10))
wedges, texts, autotexts = plt.pie(
    values,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops=dict(width=0.4),
    textprops=dict(color='black', fontsize=12)
)

# Add center circle and title
plt.gca().set_aspect('equal')
plt.title('Average % of Purchases from Discounts by Cluster', fontsize=14, weight='bold', pad=20)
plt.tight_layout()
plt.show()

# Summary of Discount purchases

# 1. Cluster 1	46.3%
#     Highly deal-driven – these customers buy nearly half of their items on      discount. Very price sensitive.
# 2. Cluster 3	26.6%
#    Moderate discount usage – value-conscious but not entirely deal-            dependent.
# 3. Cluster 0	15.3%
#    Occasionally uses discounts – selectively responsive to promotions.         Likely prefers full value.
# 4. Cluster 2	11.9%
#    Least influenced by discounts. Possibly high-income, quality-focused        buyers.

# Promotional Campaign Engagement

# Define promotion columns
promotion_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']

# Calculate total promotions accepted per cluster
total_accepted_by_cluster = df_cleaned.groupby('Product_Cluster')[promotion_cols].sum()
total_accepted_by_cluster['Total_Accepted'] = total_accepted_by_cluster.sum(axis=1)
total_accepted_by_cluster = total_accepted_by_cluster.reset_index()

# Set colorblind-friendly color palette (ColorBrewer: Set2 / Paired / Accent / Tableau 10)
colors = sns.color_palette("tab10", len(promotion_cols))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
total_accepted_by_cluster.set_index('Product_Cluster')[promotion_cols].plot(
    kind='bar',
    stacked=True,
    ax=ax,
    color=colors,
    edgecolor='black'
)

# Styling
ax.set_title('Total Promotions Accepted by Campaign & Cluster', fontsize=14, weight='bold')
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Number of Accepted Promotions', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.legend(title='Campaign', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, title_fontsize=11)

# Improve tick labels
ax.set_xticklabels([f'Cluster {i}' for i in total_accepted_by_cluster['Product_Cluster']])
plt.tight_layout()
plt.show()


# Summary of Promotion used

# 1. Cluster 0 – Most Promotion-Responsive
#     a. Highest total number of accepted campaigns (~270)
#     b. Especially strong engagement in Cmp1, Cmp4, and Cmp5
#     c. Likely highly engaged or loyal segment
#     d. Action: Prioritize this segment for multi-step campaigns, product           launches, and high-frequency email marketing

# 3. Cluster 1 – Cmp3 Dominated
#     a. ~150 total campaign acceptances
#     b. Dominated almost entirely by Cmp3, minimal response to others
#     c. May be selectively responsive or driven by a single strong offer
#     d. Action: Analyze what made Cmp3 successful and replicate its                 strategy; personalized and targeted campaigns work best here

# 4. Cluster 2 – Balanced Promotion Responders
#     a. ~175 total promotions accepted
#     b. Fairly even across Cmp1, Cmp4, and Cmp5
#     c. Indicates multi-campaign effectiveness
#     d. Action: Use cross-channel campaigns and progressive incentives; this        group responds to variety

# 5. Cluster 3 – Least Promotion-Responsive
#     a. Only ~70 total promotions accepted
#     b. Small spikes in Cmp3 and Cmp5
#     c. Likely less engaged, more resistant or unaware
#     d. Action: Use awareness campaigns, retargeting, or non-intrusive              offers like app notifications or loyalty nudges


# Salary distribution of cluster on basis of number of children

cluster_kids_spend = df_cleaned.groupby('Product_Cluster')[['Total_Spending', 'Total_Children']].mean().reset_index()
cluster_kids_spend.columns = ['Cluster', 'Avg_Spending', 'Avg_Children'] 

plt.figure(figsize=(12, 6))
sns.set(style='whitegrid')

# Strip plot to show actual data points with jitter
sns.stripplot(
    data=df_cleaned,
    x='Total_Children',
    y='Total_Spending',
    hue='Product_Cluster',
    jitter=0.25,
    dodge=True,
    palette='Set2',
    alpha=0.7,
    edgecolor='gray'
)

plt.title('Spending Distribution by Number of Children and Cluster', fontsize=14, weight='bold')
plt.xlabel('Number of Children (Kids + Teens)', fontsize=12)
plt.ylabel('Total Spending', fontsize=12)
plt.legend(title='Cluster', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# Summary of Total Spending per Number of children

# 1. Cluster 0 (Green) – High Spenders with Few Kids
#     a. Spending remains consistently high, even with 0 or 1 child
#     b. Appears to be affluent individuals or dual-income families
#     c. Represented across all child counts, including some with 3

# 2. Cluster 1 (Orange) – Large Families with Lower Spending
#     a. Most concentrated at 1–3 children
#     b. Spending is consistently low (below 1000)
#     c. Possibly budget-conscious families, large households

# 3. Cluster 2 (Blue) – Balanced Spenders with Few Kids
#     a. Mostly customers with 0–1 child
#     b.Moderate to high spending range (~1000–2000)

# Indicates selective, quality-focused spending

# Characteristics of clusters based on spending and age 

cluster_age_spend = df_cleaned.groupby('Product_Cluster')[['Age', 'Total_Spending']].mean().reset_index()
cluster_age_spend.columns = ['Cluster', 'Avg_Age', 'Avg_Spending']

# Set style
sns.set(style='whitegrid')

# Create FacetGrid per cluster
g = sns.lmplot(
    data=df_cleaned,
    x='Age',
    y='Total_Spending',
    hue='Product_Cluster',
    col='Product_Cluster',
    col_wrap=2,
    palette='Set2',
    height=4,
    scatter_kws={'alpha': 0.6, 's': 30},
    line_kws={'linewidth': 2}
)

# Titles and layout
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Spending vs Age by Cluster (Trend Comparison)', fontsize=16, weight='bold')
g.set_axis_labels("Age", "Total Spending")
g.set_titles("Cluster {col_name}")

plt.tight_layout()
plt.show()

# Summary of clusters based on spending and age 

# 1. Cluster 0: High Spenders, Younger Audience
#     a. Customers are mostly aged 30–60
#     b. Spending is consistently high (~1000–2000+)
#     c. Flat trend line: Age doesn't significantly affect spending here
#     d. Likely represents young professionals or dual-income households

# 2. Cluster 1: Large Volume, Low Spending
#     a. Customers skew younger, mostly under 50
#     b. Spending is low overall, but trend slightly increases with age
#     c. Indicates that older customers in this cluster spend slightly more, but the change is         small

# 3. Cluster 2: Spending Drops with Age
#     a. Wide age range: 30 to 90+
#     b. Strong negative correlation: As customers age, spending drops
#     c. Younger customers here spend significantly more

# 4. Cluster 3: Mid Spend, Slight Age-Driven Growth
#     a. Most customers between 40–75
#     b. Positive trend: Spending grows slightly with age
#     c. Indicates increasing trust or income over time

# Churn Rate , Key AHA Moment and Loyality Score 

# 2. Churn flag
df_cleaned['Is_Churned'] = df_cleaned['Recency'] > 60

# 3. Aha moment: accepted any campaign and >3 web purchases
df_cleaned['Aha_Moment'] = (
    (df_cleaned[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1) > 0) &
    (df_cleaned['NumWebPurchases'] > 3)
)

# 4. Loyalty score: campaigns + purchases + (inverted recency)
df_cleaned['Loyalty_Score'] = (
    df_cleaned[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1) +
    df_cleaned['NumWebPurchases'] +
    df_cleaned['NumStorePurchases'] +
    (60 - df_cleaned['Recency'])
)

# 5. Group and summarize
summary = df_cleaned.groupby('Product_Cluster').agg({
    'Is_Churned': 'mean',
    'Aha_Moment': 'mean',
    'Loyalty_Score': 'mean'
}).reset_index()

summary.columns = ['Cluster', 'Churn_Rate', 'Aha_Moment_Rate', 'Loyalty_Score']

# Optional: print summary DataFrame
print(summary)

# Setup
sns.set(style='whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
colors = {
    'Churn_Rate': '#EF476F',
    'Aha_Moment_Rate': '#06D6A0',
    'Loyalty_Score': '#118AB2'
}

# 1. Churn Rate %
sns.barplot(data=summary, x='Cluster', y='Churn_Rate', ax=axes[0], color=colors['Churn_Rate'])
axes[0].set_title('🔁 Churn Rate by Cluster', fontsize=14, weight='bold')
axes[0].set_ylabel('Churn Rate (%)')
axes[0].set_xlabel('Cluster')
for p in axes[0].patches:
    height = p.get_height()
    axes[0].annotate(f"{height:.1%}", (p.get_x() + p.get_width()/2, height + 0.005), 
                     ha='center', va='bottom', fontsize=10, weight='bold')

# 2. Aha Moment Rate %
sns.barplot(data=summary, x='Cluster', y='Aha_Moment_Rate', ax=axes[1], color=colors['Aha_Moment_Rate'])
axes[1].set_title('💡 Aha Moment Rate by Cluster', fontsize=14, weight='bold')
axes[1].set_ylabel('Aha Moment Rate (%)')
axes[1].set_xlabel('Cluster')
for p in axes[1].patches:
    height = p.get_height()
    axes[1].annotate(f"{height:.1%}", (p.get_x() + p.get_width()/2, height + 0.005), 
                     ha='center', va='bottom', fontsize=10, weight='bold')

# 3. Loyalty Score
sns.barplot(data=summary, x='Cluster', y='Loyalty_Score', ax=axes[2], color=colors['Loyalty_Score'])
axes[2].set_title('❤️ Loyalty Score by Cluster', fontsize=14, weight='bold')
axes[2].set_ylabel('Loyalty Score')
axes[2].set_xlabel('Cluster')
for p in axes[2].patches:
    height = p.get_height()
    axes[2].annotate(f"{height:.1f}", (p.get_x() + p.get_width()/2, height + 1), 
                     ha='center', va='bottom', fontsize=10, weight='bold')

# Final polish
plt.suptitle("📊 Cluster Behavior Overview", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Summary of clusters based on Churn Rate, AHA Moment and Loyalty

# **Churn Rate by Cluster**
# 1. Cluster 0: Highest churn rate (40.8%) — needs proactive retention
# 2. Cluster 1: Moderate churn (37.5%) — slightly better but still at risk
# 3. Cluster 2: Lowest churn (37.2%) — most stable segment
# 4. Cluster 3: High churn (40.6%) — needs re-engagement strategy

# **Aha Moment Rate by Cluster**
# 1. Cluster 0: Highest aha moment rate (34.4%) — strong engagement trigger present
# 2. luster 1: Very low aha moment rate (4.8%) — customers not experiencing value
# 3. Cluster 2: Moderate engagement (27.3%) — healthy trigger behavior
# 4. Cluster 3: Average engagement (22.8%) — improvement possible

# **Cluster Loyalty Scoring**
# 1. Cluster 2: Most loyal (26.4) — repeat purchases & campaign interaction
# 2. Cluster 0: High loyalty (25.5) — strong engagement
# 3. Cluster 3: Moderate loyalty (22.9) — opportunity to increase retention
# 4. Cluster 1: Lowest loyalty (18.2) — disengaged segment

# Executive Summary: Customer Segment Profiles (Cluster-Based)

**Segment A – Value-Conscious Families (Cluster 0)**
# 1. Profile:
#     1. Age: 25–50
#     2. Income: Low-to-Mid (₹5,000–₹40,000)
#     3. Spending: Low (< ₹500)
#     4. Mostly parents with one child
#     5. Long-standing customers (300+ days)
#     6. Rarely respond to promotions or use discounts

# 2. Recommendation:
#     1. Focus on product bundling and basic loyalty perks
#     2. Low price sensitivity; promotion strategy not critical
#     3. Retention via experience-based offers (5.g., referral incentives)

# **Segment B – Career-Focused Professionals (Cluster 1)**
# 1. Profile:
#     1. Age: 30–60
#     2.Income: Upper Mid (₹65,000–₹85,000)
#     3. Spending: High (₹550–₹2,000)
#     4. Educated, mostly married, no children
#     5. Moderate campaign engagement (~50%)
#     6. ow use of discounts

# 2. Recommendation:
#     1. Excellent segment for premium services and annual memberships
#     2. Promote exclusive access and milestone rewards
#     3. Focus on experience, convenience, and service quality

# **Segment C – Price-Sensitive Parents (Cluster 2)**
# 1. Profile:
#     1. Age: 35–60
#     2. Income: Mid-to-Upper Mid (₹50,000–₹80,000)
#     3. Spending: Moderate (₹250–₹1,800)
#     4. Married with children (1+), loyal base (400+ days)
#     5. Weak response to promotions, but actively use discounts

# 2. Recommendation:
#     1. Perfect fit for family packs, bulk offers, and discount-based loyalty
#     2. Test price-based segmentation, flash deals, and cashback campaigns
#     3. Retain with value-driven programs, not flashy offers

# **Segment D – Mature Value Shoppers (Cluster 3)**
# 1. Profile:
#     1. Age: 40–65
#     2. Income: Lower Mid (₹40,000–₹60,000)
#     3. Spending: Low (< ₹500)
#     4. All are parents (typically 2 children), shorter lifecycle (~150 days)
#     5. Rarely respond to campaigns, but frequently use discounts

# 2. Recommendation:
#     1. Consider discount loyalty tiers (5.g., points-based system)
#     2. Segment for “essentials” category targeting
#     3. Monitor closely for churn signals, reinforce retention with budget-driven benefits
