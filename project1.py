import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score , recall_score , confusion_matrix 
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Load the dataset
try:
    df = pd.read_csv("project1\\heavy_customer_segmentation_dataset.csv")
except FileNotFoundError as f:
    print(f"File not found. Please ensure the dataset is in the correct path. {f}")
    exit()

# Initial data exploration

print(f"info: {df.info()}")
print(f"isnull: {df.isnull().sum()}")
print(f"describe: {df.describe()}")

# Data Visualization

# Histogram of Age distribution
plt.hist(df["Age"], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig("project1/age_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Scatter plot of Annual Income vs Spending Score
plt.scatter(df["Annual_Income"], df["Spending_Score"], alpha=0.6, color='orange')
plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("project1/Annual_income_vs_spending.png", dpi=300, bbox_inches="tight")
plt.show()

# Count of each category in 'Purchase'
purchase_counts = df['Purchase'].value_counts()
plt.bar(purchase_counts.index, purchase_counts.values, color=['red', 'green'])
plt.xticks([0,1], ['No Purchase', 'Purchase'])
plt.ylabel('Number of Customers')
plt.title('Purchase Count of Customers')
plt.savefig("project1/purchase_count.png", dpi=300, bbox_inches="tight")
plt.show()

# Encoding categorical variables

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

df = pd.get_dummies(df, columns=["City"])
# Convert boolean columns to integers
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

#feature scaling
ss = StandardScaler()
# Feature scaling (exclude the target variable 'Purchase')
features = ["Age", "Annual_Income", "Spending_Score", "Visit_Frequency", "Online_Shopping"]
df[features] = ss.fit_transform(df[features])

# Split the dataset
x = df[features]
y = df["Purchase"]  # Ensure 'Purchase' remains as discrete classes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# KMeans Clustering

model = KMeans(n_clusters=3, random_state=42 , n_init=10)
df["Groups"] = model.fit_predict(df[features])

group_names = {0: "Low Spending", 1: "Average Spending", 2: "Premium Spending"}
df["Groups_Name"] = df["Groups"].map(group_names)


for group_name in df["Groups_Name"].unique():
    cluster_data = df[df["Groups_Name"] == group_name]
    plt.scatter(cluster_data["Annual_Income"], cluster_data["Spending_Score"], label=f'{group_name}', alpha=0.6)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("KMeans Clustering of Customers")
plt.legend()
plt.savefig("project1/kmeans_clustering.png", dpi=300, bbox_inches="tight")
plt.show()

# Principal Component Analysis (PCA)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])
pca = PCA(n_components=2 , random_state=42)
pca_data = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(pca_data , columns=["PCA1","PCA2"])

plt.figure(figsize=(8,6))
plt.scatter(pca_df["PCA1"], pca_df["PCA2"], alpha=0.6, color='purple')
plt.title("PCA of Customer Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("project1/pca_customer_data.png", dpi=300, bbox_inches="tight")
plt.show()

# Model Evaluation

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")

cm = confusion_matrix(y_test , y_pred)

plt.figure(figsize=(8,8))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xticks([0,1], ["No Purchase", "Purchase"])
plt.yticks([0,1], ["No Purchase", "Purchase"])
plt.ylabel('True label', fontsize=16 )
plt.xlabel('Predicted label',fontsize=16)
plt.tight_layout()
plt.savefig("project1/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Bar plot of average Purchase rate by cluster

plt.figure(figsize=(8,6))
ax = sns.barplot(x='Groups_Name', y='Purchase', data=df, palette='Set2')

plt.title('Average Purchase Rate by Cluster')
plt.ylabel('Purchase Rate')

# Har bar ke upar value show karna
for p in ax.patches:
    height = p.get_height()  # bar ki height = y-value
    ax.text(p.get_x() + p.get_width()/2, height + 0.03,  # thoda upar text ke liye
            f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.grid(axis='y')
plt.tight_layout()
plt.savefig("project1/average_purchase_rate_by_cluster.png", dpi=300, bbox_inches="tight")
plt.show()
