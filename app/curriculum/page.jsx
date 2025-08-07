"use client"
import Image from "next/image"
import { useState, useEffect } from "react"
import { Play, BookOpen, Clock, Users, Award, ExternalLink, Eye, EyeOff, Copy, Check } from "lucide-react"

export default function CurriculumPage() {
  const [isLoaded, setIsLoaded] = useState(false)
  const [showAnswers, setShowAnswers] = useState({})
  const [copiedCode, setCopiedCode] = useState({})

  useEffect(() => {
    setTimeout(() => {
      window.scrollTo(0, 0)
    }, 100)
    setIsLoaded(true)
  }, [])

  const curriculumData = {
    videos: [
      {
        id: 1,
        title: "Introduction",
        description:
          "Get to know our team and why we created this curriculum. In this video, we’ll introduce ShiftSC, explain the purpose behind our AI Ethics initiative, and share what you can expect from the series. ",
        duration: "2 min",
        youtubeId: "ri9gSNfMUZI", 
        thumbnail: "/curric/open.png",
      },
      {
        id: 2,
        title: "Foundations of Data Science",
        description:
          "Explore the foundations of data science through hands-on work with datasets, visualizations, and distributions, while reflecting on how ethics shape data collection and analysis.",
        duration: "28 min",
        youtubeId: "UWABqbRE2a4",
        thumbnail: "/curric/0.png",
      },
      {
        id: 3,
        title: "Machine Learning",
        description:
          "Learn the core principles of machine learning, including types of learning, the model development process, and the ethical considerations involved in building intelligent systems.",
        duration: "40 min",
        youtubeId: "jSE4ER82V2Q",
        thumbnail: "/curric/1.png",
      },
      {
        id: 4,
        title: "Bias and Fairness",
        description:
          "Understand how bias enters machine learning systems and explore key definitions, fairness metrics, and approaches to achieving both group and individual fairness.",
        duration: "24 min",
        youtubeId: "SAv-btA7Rjw",
        thumbnail: "/curric/2.png",
      },
      {
        id: 5,
        title: "Safety and Robustness",
        description: 
          "Examine how models can fail under adversarial conditions, and learn how to test and strengthen them using techniques like FGSM, noise-based attacks, and a look into policy and regulatory safeguards.",
        duration: "18 min",
        youtubeId: "VJdLzyTowAU",
        thumbnail: "/curric/3.png",
      },
      { 
        id: 6,
        title: "Capstone",
        description:
          "Apply everything you’ve learned in a real-world challenge—tackling fairness in hospital or scholarship allocation or revisiting criminal justice tools like COMPAS—to design more equitable, responsible AI systems.",
        duration: "14 min",
        youtubeId: "_5FODTnFya8",
        thumbnail: "/curric/4.png",
      },
      {
        id: 7,
        title: "Closing Remarks",
        description:
          "A brief thank-you to students for engaging with the course, along with a reminder to stay tuned for future modules and opportunities to keep learning.",
        duration: "1 min",
        youtubeId: "1QQSsZwtkKU",
        thumbnail: "/curric/close.png",
      },
    ],
    notebooks: [
      {
        id: 1,
        title: "The Power of Data",
        description:
          "Hands-on practice with data manipulation, cleaning, and visualization using pandas and matplotlib on the classic Iris dataset.",
        colabUrl: "https://colab.research.google.com/drive/1LAj5-j9n_T2R4tSUi62Tap3XlifBCHUC?usp=sharing",
        relatedVideos: [0],
        practiceActivities: [
          "Complete pandas exercises to sort, filter, and analyze iris flower data",
          "Create scatter plots, histograms, and multi-panel visualizations using matplotlib",
          "Explore a self-selected dataset and build custom data visualizations",
        ],
        codeAnswers: [
          {
            question: "Working with DataFrames: Sorting, Filtering, and Summarizing",
            code: `# 1. Sort the data set by sepal width in descending order
df_sorted = df.sort_values(by='sepal width', ascending=False)
print("=== Sorted data: ===")
print(df_sorted.head())

# 2. Filter out all versicolor species
# Assuming the species column is named 'species' and coded as integers
df_versicolor = df[df['species'] == 1]
print("=== Versicolors: ===")
print(df_versicolor.head())

# 3. Remove data of petal widths less than 1 cm
df_bigpetals = df[df['petal width'] >= 1]
print("=== Petal width >= 1 cm: ===")
print(df_bigpetals.head())

# 4. Find the average, max, and minimum petal width value in the dataframe.
max = df['petal width'].max()
min = df['petal width'].min()
avg = df['petal width'].mean()
print(f"=== Petal size Summary stats ===\nMax: {max}, Min: {min}, Avg: {avg}")`,
          },
          {
            question: "Visualizing Data with Matplotlib: How Can We Explore Patterns Through Plots?",
            code: `import matplotlib.pyplot as plt
import pandas as pd

# Assuming df is already defined and loaded

# 1. Scatter plot: Petal Width vs Sepal Width for Setosa (species 0)
setosa = df[df['species'] == 0]

plt.figure(figsize=(6, 4))
plt.scatter(setosa['sepal width'], setosa['petal width'])
plt.title('Setosa: Petal Width vs Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Width (cm)')
plt.grid(True)
plt.show()

# 2. Line plot: y = x^2
function = [x**2 for x in range(100)]

plt.figure(figsize=(6, 4))
plt.plot(function)
plt.title('Plot of x² from 0 to 99')
plt.xlabel('x')
plt.ylabel('x²')
plt.grid(True)
plt.show()

# 3. Subplots: Histogram of average petal/sepal dimensions by species
averages = df.groupby('species').mean(numeric_only=True)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].bar(averages.index, averages['petal width'])
axs[0, 0].set_title('Avg Petal Width by Species')

axs[0, 1].bar(averages.index, averages['petal length'])
axs[0, 1].set_title('Avg Petal Length by Species')

axs[1, 0].bar(averages.index, averages['sepal width'])
axs[1, 0].set_title('Avg Sepal Width by Species')

axs[1, 1].bar(averages.index, averages['sepal length'])
axs[1, 1].set_title('Avg Sepal Length by Species')

for ax in axs.flat:
    ax.set_xlabel('Species')
    ax.set_ylabel('Length (cm)')

plt.tight_layout()
plt.show()

# 4. Multiple function plots: x³ and |x - 50|
f1 = [x**3 for x in range(100)]
f2 = [abs(x - 50) for x in range(100)]

plt.figure(figsize=(6, 4))
plt.plot(f1, label='x³')
plt.plot(f2, label='|x - 50|')
plt.title('Multiple Functions on One Graph')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()`,
          },
          {
            question: "Can You Explore a New Dataset on Your Own?",
            code: `from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
yourdata = load_breast_cancer()
df = pd.DataFrame(yourdata.data, columns=yourdata.feature_names)

# Add target labels (0 = malignant, 1 = benign)
df['target'] = yourdata.target

# Sort by 'mean area' descending
df_sorted = df.sort_values(by='mean area', ascending=False)

# Filter: only include tumors with 'mean radius' above 20
df_large_radius = df[df['mean radius'] > 20]

# Visualization 1: Scatter plot of mean radius vs mean texture
plt.figure(figsize=(6, 4))
plt.scatter(df['mean radius'], df['mean texture'], alpha=0.5, c=df['target'], cmap='coolwarm')
plt.title('Mean Radius vs Mean Texture (Colored by Diagnosis)')
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.grid(True)
plt.colorbar(label='Target (0 = Malignant, 1 = Benign)')
plt.show()

# Visualization 2: Boxplot of mean area grouped by diagnosis
plt.figure(figsize=(6, 4))
df.boxplot(column='mean area', by='target', grid=False)
plt.title('Mean Area by Diagnosis')
plt.suptitle('')
plt.xlabel('Diagnosis (0 = Malignant, 1 = Benign)')
plt.ylabel('Mean Area')
plt.show()`,
          },
        ],
      },
      {
        id: 2,
        title: "Machine Learning Models: Decision Trees and Neural Networks",
        description:
          "Build and train decision tree classifiers and PyTorch neural networks while experimenting with hyperparameters to optimize performance.",
        colabUrl: "https://colab.research.google.com/drive/1rcAdszKtS2NWKztXj-hj7XTwQkmByvMD?usp=sharing",
        relatedVideos: [1],
        practiceActivities: [
          "Train and visualize a decision tree classifier using scikit-learn's DecisionTreeClassifier and plot_tree functions",
          "Experiment with neural network hyperparameters (hidden_dim, learning_rate, num_epochs) to optimize model accuracy",
          "Build and train PyTorch neural networks with different architectures to understand performance impacts",
        ],
        codeAnswers: [
          {
            question: "Loading the Iris Dataset",
            code: `# 📥 Load the Iris dataset
import pandas as pd
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target column (species)
df['target'] = iris.target

# Preview the data
df.head()`,
          },
          {
            question: "Can Sepal Length Predict Petal Length? (Linear Regression)",
            code: `# 📈 Linear Regression: Sepal Length → Petal Length
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define inputs (X) and outputs (y)
X = df[['sepal length (cm)']]
y = df['petal length (cm)']

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Predict petal length
df['predicted_petal_length'] = model.predict(X)

# Plot the regression line
plt.figure(figsize=(6, 4))
plt.scatter(X, y, label='Actual', alpha=0.6)
plt.plot(X, df['predicted_petal_length'], color='red', label='Regression Line')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Linear Regression: Sepal Length → Petal Length')
plt.legend()
plt.grid(True)
plt.show()`,
          },
          {
            question: "Can We Classify Iris Species Using a Decision Tree?",
            code: `# 🌳 Decision Tree Classifier and Visualization
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Define features (X) and target (y)
X = df[iris.feature_names]
y = df['target']

# Train decision tree
tree_model = DecisionTreeClassifier(random_state=0, max_depth=3)
tree_model.fit(X, y)

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(
    tree_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("Decision Tree Trained on Iris Dataset")
plt.show()`,
          },
        ],
      },
      {
        id: 3,
        title: "Measuring and Mitigating Bias",
        description:
          "Hands-on practice identifying and measuring bias in datasets using statistical methods and visualization techniques.",
        colabUrl: "https://colab.research.google.com/drive/1ze4V4SCNc9K1nweVanBHj0MGbx986vsP?usp=sharing",
        relatedVideos: [2],
        practiceActivities: [
          "Calculate group-level fairness metrics like false positive rates across demographic groups",
          "Explore group-level fairness by computing accuracy scores for different demographic groups",
          "Implement bias mitigation techniques using sample weights to balance group representation",
        ],
        codeAnswers: [
          {
            question: "How Accurate Is the Model for Each Gender? (Group-wise Evaluation)",
            code: `from sklearn.metrics import accuracy_score

# Filter groups by gender
group_male = X_test[X_test['gender_encoded'] == 0]
group_female = X_test[X_test['gender_encoded'] == 1]

# Calculate accuracy for each group
acc_male = accuracy_score(group_male['actual'], group_male['predicted'])
acc_female = accuracy_score(group_female['actual'], group_female['predicted'])

# Display results
print(f"Male Accuracy: {acc_male:.2f}")
print(f"Female Accuracy: {acc_female:.2f}")`,
          },
          {
            question: "How Can We Make Accuracy More Fair? (Reweighting with Sample Weights)",
            code: `from sklearn.utils import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Calculate sample weights to give equal importance to male and female samples
weights = compute_sample_weight(class_weight='balanced', y=X_train['gender_encoded'])

# Train logistic regression with sample weights
model_weighted = LogisticRegression()
model_weighted.fit(X_train[['experience', 'gender_encoded']], X_train['actual'], sample_weight=weights)

# Predict using the weighted model
X_test['predicted_weighted'] = model_weighted.predict(X_test[['experience', 'gender_encoded']])

# Re-calculate accuracy per group
group_m = X_test[X_test['gender_encoded'] == 0]
group_f = X_test[X_test['gender_encoded'] == 1]

print("Male Accuracy (weighted):", accuracy_score(group_m['actual'], group_m['predicted_weighted']))
print("Female Accuracy (weighted):", accuracy_score(group_f['actual'], group_f['predicted_weighted']))`,
          },
        ],
      },
      {
        id: 4,
        title: "Model Robustness and Adversarial Testing",
        description:
          "Explore neural network robustness by testing with Gaussian noise and visualizing performance drops.",
        colabUrl: "https://colab.research.google.com/drive/10GRzvHx2FctMjBwwqeduT5rohmmz2Hzm?usp=sharing",
        relatedVideos: [3],
        practiceActivities: [
          "Add Gaussian noise to test images and compare model predictions on clean vs noisy data",
          "Evaluate model accuracy across increasing noise levels to measure robustness",
          "Visualize the relationship between noise standard deviation and model performance degradation",
        ],
        codeAnswers: [
          {
            question: "Solutions Available in Module Video",
            code: `# In this exercise, the solutions are covered in the module video where Richa goes through them and explains them in detail.
# 
# Please refer to the video lecture for comprehensive explanations of the adversarial testing techniques and robustness measures.
# 
# The video covers:
# - Gaussian noise testing
# - Performance degradation analysis  
# - Visualization techniques
# - Best practices for model robustness evaluation`,
          },
        ],
      },
      {
        id: 5,
        title: "Capstone Project",
        description:
          "Apply your learning in a comprehensive capstone project that addresses ethical AI challenges in healthcare allocation, scholarship distribution, or criminal justice reform.",
        colabUrl: "https://colab.research.google.com/drive/your-notebook-5",
        colabUrls: [
          {
            title: "Hospital Bed Allocation",
            url: "https://colab.research.google.com/drive/1capstone-hospital-allocation",
            description: "Design and implement a fair hospital bed allocation system"
          },
          {
            title: "Scholarship Distribution", 
            url: "https://colab.research.google.com/drive/2capstone-scholarship-distribution",
            description: "Create an equitable scholarship distribution algorithm"
          },
          {
            title: "COMPAS Analysis",
            url: "https://colab.research.google.com/drive/3capstone-compas-analysis", 
            description: "Analyze and improve the COMPAS recidivism prediction tool"
          }
        ],
        relatedVideos: [4],
        practiceActivities: [
          "Design and implement a fair hospital bed allocation system",
          "Create an equitable scholarship distribution algorithm",
          "Analyze and improve the COMPAS recidivism prediction tool",
        ],
        codeAnswers: [
          {
            question: "Hospital Track",
            code: `# Complete Healthcare Data Science Pipeline with Fairness and Safety Measures

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load your patient dataset
# df = pd.read_csv('patient_data.csv')

# ========================================
# 1. EXPLORING THE DATA
# ========================================

# Inspect missing values and duplicates
print("=== DATA OVERVIEW ===")
print(df.info())
print("\\n=== MISSING VALUES ===")
print(df.isnull().sum())
print(f"\\nTotal duplicates: {df.duplicated().sum()}")

# Explore the number of categories within categorical variables
print("\\n=== CATEGORICAL VARIABLES ANALYSIS ===")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique categories")
    if unique_count <= 20:  # Show categories if reasonable number
        print(f"  Categories: {df[col].value_counts().to_dict()}")
    print()

# Create visualizations to understand the data
print("=== DATA VISUALIZATION ===")
plt.figure(figsize=(15, 10))

# Distribution of numerical variables
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
n_num = len(numerical_cols)
for i, col in enumerate(numerical_cols[:6]):  # Show first 6 numerical columns
    plt.subplot(2, 3, i+1)
    plt.hist(df[col].dropna(), bins=30, alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Categorical variables visualization
plt.figure(figsize=(15, 8))
n_cat = min(len(categorical_cols), 4)  # Show up to 4 categorical variables
for i, col in enumerate(categorical_cols[:n_cat]):
    plt.subplot(2, 2, i+1)
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Correlation heatmap for numerical variables
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.show()

# ========================================
# 2. CLEANING THE DATA
# ========================================

print("\\n=== DATA CLEANING ===")

# Drop or fill missing values
print(f"Before cleaning - Shape: {df.shape}")
# Option 1: Drop rows with missing values
# df = df.dropna()

# Option 2: Fill with mean/median for numerical, mode for categorical
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

print(f"After cleaning - Shape: {df.shape}")

# Normalize or scale numerical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_cols) > 0:
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Numerical columns standardized")

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
categorical_cols = [col for col in categorical_cols if col != 'Admission_Type']  # Exclude target variable

for col in categorical_cols:
    unique_count = df[col].nunique()
    
    # Use Label Encoding for ordinal or high cardinality variables
    if unique_count > 10:
        print(f"Label encoding {col} ({unique_count} categories)")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Use One-Hot Encoding for nominal variables with few categories
    else:
        print(f"One-hot encoding {col} ({unique_count} categories)")
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

print(f"Final dataset shape: {df.shape}")

# ========================================
# 3. PATIENT PRIVACY PROTECTION
# ========================================

print("\\n=== PATIENT PRIVACY PROTECTION ===")

# Remove direct identifiers like name
if 'Name' in df.columns:
    df = df.drop(columns=['Name'], errors='ignore')
    print("Removed Name column")

# Generalize features (e.g., age → age group)
if 'age' in df.columns:
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 18, 35, 60, 100], 
                            labels=['0-18', '19-35', '36-60', '60+'])
    print("Age generalized to age groups")

# Add noise to sensitive columns
def add_noise(col, epsilon=0.1):
    return col + np.random.normal(0, epsilon, size=len(col))

# Example: Add noise to income if it exists
if 'income' in df.columns:
    df['income'] = add_noise(df['income'])
    print("Added noise to income column")

# Prepare features and target
if 'Admission_Type' in df.columns:
    X = df.drop(columns=['Admission_Type'])
    y = df['Admission_Type']
else:
    # If target column has different name, adjust accordingly
    print("Available columns:", df.columns.tolist())
    target_col = input("Enter the target column name (e.g., 'Admission_Type'): ")
    X = df.drop(columns=[target_col])
    y = df[target_col]

# Encode target variable if it's categorical
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

print(f"Features shape: {X.shape}")
print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

# ========================================
# 4. MODEL TRAINING
# ========================================

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model (Random Forest)
print("\\n=== TRAINING RANDOM FOREST MODEL ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_pred = rf_model.predict(X_test)
print("Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, rf_pred))

# Train Neural Network
print("\\n=== TRAINING NEURAL NETWORK MODEL ===")
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)

# Evaluate Neural Network  
nn_pred = nn_model.predict(X_test)
print("Neural Network Performance:")
print(f"Accuracy: {accuracy_score(y_test, nn_pred):.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, nn_pred))

# Choose the better performing model for further analysis
if accuracy_score(y_test, rf_pred) >= accuracy_score(y_test, nn_pred):
    best_model = rf_model
    best_pred = rf_pred
    model_name = "Random Forest"
    print(f"\\n{model_name} selected as the best model")
else:
    best_model = nn_model
    best_pred = nn_pred
    model_name = "Neural Network"
    print(f"\\n{model_name} selected as the best model")

# ========================================
# 5. FAIRNESS METRICS
# ========================================

print("\\n=== FAIRNESS EVALUATION ===")

# Look for financial/insurance-related columns
financial_cols = [col for col in X.columns if any(term in col.lower() 
                 for term in ['insurance', 'billing', 'income', 'payment', 'financial', 'cost'])]

print(f"Found potential financial columns: {financial_cols}")

# Example pseudocode implementation:
# Split by financial groups and evaluate fairness
if financial_cols:
    financial_col = financial_cols[0]  # Use first financial column
    
    print(f"\\nEvaluating fairness across {financial_col} groups:")
    
    for group in X_test[financial_col].unique():
        idx = X_test[financial_col] == group
        if idx.sum() > 0:  # Only process if group has samples
            group_accuracy = accuracy_score(y_test[idx], best_pred[idx])
            group_precision = precision_score(y_test[idx], best_pred[idx], average='weighted', zero_division=0)
            group_recall = recall_score(y_test[idx], best_pred[idx], average='weighted', zero_division=0)
            
            print(f"Group {group}:")
            print(f"  Accuracy: {group_accuracy:.3f}")
            print(f"  Precision: {group_precision:.3f}")
            print(f"  Recall: {group_recall:.3f}")
            print(f"  Sample size: {idx.sum()}")
            print()

# Alternative: Check for gender-based fairness if gender column exists
gender_cols = [col for col in X.columns if 'gender' in col.lower()]
if gender_cols:
    gender_col = gender_cols[0]
    print(f"\\nEvaluating fairness across gender groups ({gender_col}):")
    
    for group in X_test[gender_col].unique():
        idx = X_test[gender_col] == group
        if idx.sum() > 0:
            group_accuracy = accuracy_score(y_test[idx], best_pred[idx])
            group_precision = precision_score(y_test[idx], best_pred[idx], average='weighted', zero_division=0)
            group_recall = recall_score(y_test[idx], best_pred[idx], average='weighted', zero_division=0)
            
            print(f"Gender {group}:")
            print(f"  Accuracy: {group_accuracy:.3f}")
            print(f"  Precision: {group_precision:.3f}")
            print(f"  Recall: {group_recall:.3f}")
            print(f"  Sample size: {idx.sum()}")
            print()

# ========================================
# 6. SAFETY MEASUREMENTS
# ========================================

print("\\n=== SAFETY AND ROBUSTNESS TESTING ===")

# Add noise to test data and compare performance
def add_random_noise(X, epsilon=0.05):
    """Add random noise to numerical features"""
    X_noisy = X.copy()
    numerical_features = X.select_dtypes(include=[np.number]).columns
    
    for col in numerical_features:
        noise = np.random.normal(0, epsilon, size=X[col].shape)
        X_noisy[col] = X[col] + noise
    
    return X_noisy

# Test model robustness with noisy data
X_test_noisy = add_random_noise(X_test, epsilon=0.05)
y_pred_noisy = best_model.predict(X_test_noisy)

print("Performance on clean test data:")
print(f"Accuracy: {accuracy_score(y_test, best_pred):.3f}")

print("\\nPerformance on noisy test data:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_noisy):.3f}")

print(f"\\nRobustness score (difference): {abs(accuracy_score(y_test, best_pred) - accuracy_score(y_test, y_pred_noisy)):.3f}")

# Test with different noise levels
print("\\n=== ROBUSTNESS ACROSS NOISE LEVELS ===")
noise_levels = [0.01, 0.05, 0.1, 0.2]
robustness_scores = []

for epsilon in noise_levels:
    X_test_noisy = add_random_noise(X_test, epsilon=epsilon)
    y_pred_noisy = best_model.predict(X_test_noisy)
    noisy_accuracy = accuracy_score(y_test, y_pred_noisy)
    robustness_scores.append(noisy_accuracy)
    print(f"Noise level {epsilon}: Accuracy = {noisy_accuracy:.3f}")

# Visualize robustness
plt.figure(figsize=(10, 6))
original_accuracy = accuracy_score(y_test, best_pred)
plt.plot([0] + noise_levels, [original_accuracy] + robustness_scores, 'b-o')
plt.xlabel('Noise Level (Epsilon)')
plt.ylabel('Accuracy')
plt.title('Model Robustness Across Noise Levels')
plt.grid(True)
plt.show()

# Feature importance analysis
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

print("✓ Model trained and evaluated")
print("✓ Fairness metrics computed")
print("✓ Safety and robustness testing completed")`,
          },
          {
            question: "Education Track",
            code: `# Complete Data Science Pipeline with Fairness and Safety Measures

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load your education dataset
# df = pd.read_csv('education_data.csv')

# ========================================
# 1. EXPLORING THE DATA
# ========================================

# Inspect missing values and duplicates
print("=== DATA OVERVIEW ===")
print(df.info())
print("\\n=== MISSING VALUES ===")
print(df.isnull().sum())
print(f"\\nTotal duplicates: {df.duplicated().sum()}")

# Explore categorical variables
print("\\n=== CATEGORICAL VARIABLES ANALYSIS ===")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique categories")
    if unique_count <= 20:
        print(f"  Categories: {df[col].value_counts().to_dict()}")
    print()

# Create visualizations
print("=== DATA VISUALIZATION ===")
plt.figure(figsize=(15, 10))

# Distribution of numerical variables
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
n_num = len(numerical_cols)
for i, col in enumerate(numerical_cols[:6]):
    plt.subplot(2, 3, i+1)
    plt.hist(df[col].dropna(), bins=30, alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Categorical variables visualization
plt.figure(figsize=(15, 8))
n_cat = min(len(categorical_cols), 4)
for i, col in enumerate(categorical_cols[:n_cat]):
    plt.subplot(2, 2, i+1)
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Correlation heatmap
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.show()

# ========================================
# 2. CLEANING THE DATA
# ========================================

print("\\n=== DATA CLEANING ===")

# Fill missing values
print(f"Before cleaning - Shape: {df.shape}")
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

print(f"After cleaning - Shape: {df.shape}")

# Normalize numerical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_cols) > 0:
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Numerical columns standardized")

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
categorical_cols = [col for col in categorical_cols if col != 'Scholarship_Awarded']  # Exclude target

for col in categorical_cols:
    unique_count = df[col].nunique()
    
    if unique_count > 10:
        print(f"Label encoding {col} ({unique_count} categories)")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    else:
        print(f"One-hot encoding {col} ({unique_count} categories)")
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

print(f"Final dataset shape: {df.shape}")

# ========================================
# 3. PRIVACY PROTECTION
# ========================================

print("\\n=== PRIVACY PROTECTION ===")

# Remove direct identifiers
if 'Student_Name' in df.columns:
    df = df.drop(columns=['Student_Name'], errors='ignore')
    print("Removed Student_Name column")

# Generalize features
if 'age' in df.columns:
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 18, 25, 35, 100], 
                            labels=['0-18', '19-25', '26-35', '35+'])
    print("Age generalized to age groups")

# Add noise to sensitive columns
def add_noise(col, epsilon=0.1):
    return col + np.random.normal(0, epsilon, size=len(col))

if 'income' in df.columns:
    df['income'] = add_noise(df['income'])
    print("Added noise to income column")

# Prepare features and target
if 'Scholarship_Awarded' in df.columns:
    X = df.drop(columns=['Scholarship_Awarded'])
    y = df['Scholarship_Awarded']
else:
    print("Available columns:", df.columns.tolist())
    target_col = input("Enter the target column name (e.g., 'Scholarship_Awarded'): ")
    X = df.drop(columns=[target_col])
    y = df[target_col]

# Encode target variable
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

print(f"Features shape: {X.shape}")
print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

# ========================================
# 4. MODEL TRAINING
# ========================================

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
print("\\n=== TRAINING RANDOM FOREST MODEL ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_pred = rf_model.predict(X_test)
print("Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, rf_pred))

# Train Neural Network
print("\\n=== TRAINING NEURAL NETWORK MODEL ===")
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)

# Evaluate Neural Network
nn_pred = nn_model.predict(X_test)
print("Neural Network Performance:")
print(f"Accuracy: {accuracy_score(y_test, nn_pred):.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, nn_pred))

# Choose best model
if accuracy_score(y_test, rf_pred) >= accuracy_score(y_test, nn_pred):
    best_model = rf_model
    best_pred = rf_pred
    model_name = "Random Forest"
    print(f"\\n{model_name} selected as the best model")
else:
    best_model = nn_model
    best_pred = nn_pred
    model_name = "Neural Network"
    print(f"\\n{model_name} selected as the best model")

# ========================================
# 5. FAIRNESS METRICS
# ========================================

print("\\n=== FAIRNESS EVALUATION ===")

# Look for demographic columns
demographic_cols = [col for col in X.columns if any(term in col.lower() 
                   for term in ['gender', 'race', 'ethnicity', 'income', 'region'])]

print(f"Found potential demographic columns: {demographic_cols}")

# Evaluate fairness across demographic groups
if demographic_cols:
    demographic_col = demographic_cols[0]
    
    print(f"\\nEvaluating fairness across {demographic_col} groups:")
    
    for group in X_test[demographic_col].unique():
        idx = X_test[demographic_col] == group
        if idx.sum() > 0:
            group_accuracy = accuracy_score(y_test[idx], best_pred[idx])
            group_precision = precision_score(y_test[idx], best_pred[idx], average='weighted', zero_division=0)
            group_recall = recall_score(y_test[idx], best_pred[idx], average='weighted', zero_division=0)
            
            print(f"Group {group}:")
            print(f"  Accuracy: {group_accuracy:.3f}")
            print(f"  Precision: {group_precision:.3f}")
            print(f"  Recall: {group_recall:.3f}")
            print(f"  Sample size: {idx.sum()}")
            print()

# ========================================
# 6. SAFETY MEASUREMENTS
# ========================================

print("\\n=== SAFETY AND ROBUSTNESS TESTING ===")

# Add noise to test data
def add_random_noise(X, epsilon=0.05):
    """Add random noise to numerical features"""
    X_noisy = X.copy()
    numerical_features = X.select_dtypes(include=[np.number]).columns
    
    for col in numerical_features:
        noise = np.random.normal(0, epsilon, size=X[col].shape)
        X_noisy[col] = X[col] + noise
    
    return X_noisy

# Test model robustness
X_test_noisy = add_random_noise(X_test, epsilon=0.05)
y_pred_noisy = best_model.predict(X_test_noisy)

print("Performance on clean test data:")
print(f"Accuracy: {accuracy_score(y_test, best_pred):.3f}")

print("\\nPerformance on noisy test data:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_noisy):.3f}")

print(f"\\nRobustness score (difference): {abs(accuracy_score(y_test, best_pred) - accuracy_score(y_test, y_pred_noisy)):.3f}")

# Test with different noise levels
print("\\n=== ROBUSTNESS ACROSS NOISE LEVELS ===")
noise_levels = [0.01, 0.05, 0.1, 0.2]
robustness_scores = []

for epsilon in noise_levels:
    X_test_noisy = add_random_noise(X_test, epsilon=epsilon)
    y_pred_noisy = best_model.predict(X_test_noisy)
    noisy_accuracy = accuracy_score(y_test, y_pred_noisy)
    robustness_scores.append(noisy_accuracy)
    print(f"Noise level {epsilon}: Accuracy = {noisy_accuracy:.3f}")

# Visualize robustness
plt.figure(figsize=(10, 6))
original_accuracy = accuracy_score(y_test, best_pred)
plt.plot([0] + noise_levels, [original_accuracy] + robustness_scores, 'b-o')
plt.xlabel('Noise Level (Epsilon)')
plt.ylabel('Accuracy')
plt.title('Model Robustness Across Noise Levels')
plt.grid(True)
plt.show()

# Feature importance analysis
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

print("✓ Model trained and evaluated")
print("✓ Fairness metrics computed")
print("✓ Safety and robustness testing completed")`,
          },
          {
            question: "Law Track",
            code: `# Complete Patient Data ML Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Assuming we have a dataframe 'df' loaded with patient data
# df = pd.read_csv('your_patient_data.csv')

# ===== EXPLORING THE DATA =====
print("🔍 Exploring the Data")
print("=" * 50)

# Inspect missing values and duplicates
print("Dataset Info:")
print(df.info())
print(f"\nMissing values per column:")
print(df.isnull().sum())
print(f"\nTotal duplicated rows: {df.duplicated().sum()}")

# Explore the number of categories within categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
print(f"\nCategorical Variables Analysis:")
for col in categorical_columns:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique categories")
    if unique_count <= 10:  # Show categories for columns with <= 10 unique values
        print(f"  Categories: {list(df[col].unique())}")
    print()

# Create visualizations to understand the data better
plt.figure(figsize=(15, 10))

# Plot distribution of numerical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
for i, col in enumerate(numeric_columns[:6], 1):  # Limit to first 6 numeric columns
    plt.subplot(2, 3, i)
    plt.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Correlation heatmap for numeric variables
if len(numeric_columns) > 1:
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Numeric Variables')
    plt.show()

# ===== CLEANING THE DATA =====
print("\n🧹 Cleaning the Data")
print("=" * 50)

# Drop or fill missing values
df_cleaned = df.copy()

# Fill numeric columns with median, categorical with mode
for col in df_cleaned.columns:
    if df_cleaned[col].dtype in ['int64', 'float64']:
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    else:
        df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown', inplace=True)

# Remove duplicates
df_cleaned = df_cleaned.drop_duplicates()
print(f"Shape after cleaning: {df_cleaned.shape}")

# Normalize or scale numerical columns
scaler = StandardScaler()
numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])

# Encode categorical variables
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

# For high cardinality columns, combine rare categories
for col in categorical_cols:
    value_counts = df_cleaned[col].value_counts()
    # Combine categories that appear less than 1% of the time into 'Other'
    rare_categories = value_counts[value_counts < len(df_cleaned) * 0.01].index
    if len(rare_categories) > 0:
        df_cleaned[col] = df_cleaned[col].replace(rare_categories, 'Other')
        print(f"Combined {len(rare_categories)} rare categories in {col} into 'Other'")

# Use One-Hot Encoding for categorical variables with reasonable number of categories
for col in categorical_cols:
    if df_cleaned[col].nunique() <= 10:  # One-hot encode if <= 10 categories
        dummies = pd.get_dummies(df_cleaned[col], prefix=col, drop_first=True)
        df_cleaned = pd.concat([df_cleaned, dummies], axis=1)
        df_cleaned.drop(col, axis=1, inplace=True)
    else:  # Use label encoding for high cardinality
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])

print(f"Final shape after encoding: {df_cleaned.shape}")

# ===== PATIENT PRIVACY =====
print("\n🔒 Patient Privacy Protection")
print("=" * 50)

# Remove direct identifiers like name (if they exist)
identifier_columns = ['FirstName', 'LastName', 'MiddleName', 'PatientID', 'SSN']
existing_identifiers = [col for col in identifier_columns if col in df_cleaned.columns]
if existing_identifiers:
    df_cleaned = df_cleaned.drop(columns=existing_identifiers)
    print(f"Removed identifier columns: {existing_identifiers}")

# Generalize features (e.g., age → age group)
if 'age' in df_cleaned.columns:
    df_cleaned['age_group'] = pd.cut(df_cleaned['age'], 
                                   bins=[0, 18, 35, 60, 100], 
                                   labels=['0-18', '19-35', '36-60', '60+'])
    df_cleaned = df_cleaned.drop('age', axis=1)
    # One-hot encode age groups
    age_dummies = pd.get_dummies(df_cleaned['age_group'], prefix='age_group', drop_first=True)
    df_cleaned = pd.concat([df_cleaned, age_dummies], axis=1)
    df_cleaned = df_cleaned.drop('age_group', axis=1)
    print("Generalized age into age groups")

# Add noise to sensitive columns (example with income if it exists)
def add_noise(col, epsilon=0.1):
    """Add Laplacian noise for differential privacy"""
    noise = np.random.laplace(0, epsilon, size=len(col))
    return col + noise

if 'income' in df_cleaned.columns:
    df_cleaned['income'] = add_noise(df_cleaned['income'], epsilon=0.1)
    print("Added noise to income column for privacy protection")

# Prepare features and target
# Assuming 'RawScore' is the target variable (modify as needed)
target_column = 'RawScore'  # Change this to your actual target column
if target_column in df_cleaned.columns:
    X = df_cleaned.drop(columns=[target_column])
    y = df_cleaned[target_column]
else:
    print(f"Warning: Target column '{target_column}' not found. Please specify the correct target column.")
    # Use the last column as target for demonstration
    X = df_cleaned.iloc[:, :-1]
    y = df_cleaned.iloc[:, -1]

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# ===== TRAINING THE MODEL =====
print("\n🏋️ Training the Model")
print("=" * 50)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network Model
mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2
)

print("Training Neural Network...")
mlp_model.fit(X_train, y_train)

# Make predictions
y_pred = mlp_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Neural Network RMSE: {rmse:.4f}")

# For classification tasks, you might want to use MLPClassifier instead
# from sklearn.neural_network import MLPClassifier

# ===== FAIRNESS METRICS =====
print("\n⚖️ Fairness Metrics")
print("=" * 50)

# Assuming we have a sensitive attribute like 'gender' or 'ethnicity'
# Split predictions by group and compare accuracy or other metrics

# Example: Evaluate fairness across gender groups (if gender column exists)
sensitive_attributes = ['gender', 'ethnicity', 'race']  # Add your sensitive attributes
existing_sensitive = [attr for attr in sensitive_attributes if any(col.startswith(attr) for col in X.columns)]

if existing_sensitive:
    # For demonstration, let's create a binary classification version
    y_test_binary = (y_test > y_test.median()).astype(int)
    y_pred_binary = (y_pred > np.median(y_pred)).astype(int)
    
    # Find columns that start with sensitive attribute names
    for attr in existing_sensitive:
        attr_columns = [col for col in X_test.columns if col.startswith(attr)]
        if attr_columns:
            # Create groups based on the sensitive attribute
            for col in attr_columns:
                group_mask = X_test[col] == 1
                if group_mask.sum() > 10:  # Only analyze if group has sufficient samples
                    group_accuracy = accuracy_score(y_test_binary[group_mask], y_pred_binary[group_mask])
                    other_accuracy = accuracy_score(y_test_binary[~group_mask], y_pred_binary[~group_mask])
                    
                    print(f"\nFairness Analysis for {col}:")
                    print(f"  Group with {col}=1 accuracy: {group_accuracy:.4f}")
                    print(f"  Group with {col}=0 accuracy: {other_accuracy:.4f}")
                    print(f"  Accuracy difference: {abs(group_accuracy - other_accuracy):.4f}")
                    
                    # Classification report for each group
                    print(f"\n  Classification Report for {col}=1:")
                    print(classification_report(y_test_binary[group_mask], y_pred_binary[group_mask]))

# ===== SAFETY MEASUREMENTS =====
print("\n🛡️ Safety Measurements")
print("=" * 50)

# Test robustness by adding noise to test data
def add_random_noise(X, epsilon=0.05):
    """Add random noise to test robustness"""
    noise = np.random.normal(0, epsilon, size=X.shape)
    return X + noise

# Add noise to test data and compare performance
X_test_noisy = add_random_noise(X_test.values, epsilon=0.05)
y_pred_noisy = mlp_model.predict(X_test_noisy)

# Compare performance
rmse_original = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_noisy = np.sqrt(mean_squared_error(y_test, y_pred_noisy))

print(f"Original RMSE: {rmse_original:.4f}")
print(f"Noisy RMSE: {rmse_noisy:.4f}")
print(f"Performance degradation: {((rmse_noisy - rmse_original) / rmse_original * 100):.2f}%")

# Additional robustness test: Feature importance perturbation
try:
    # Train a Random Forest to get feature importance
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    y_train_binary = (y_train > np.median(y_train)).astype(int)
    rf_model.fit(X_train, y_train_binary)
    
    feature_importance = rf_model.feature_importances_
    important_features = X_train.columns[np.argsort(feature_importance)[-5:]]  # Top 5 important features
    
    print(f"\nTop 5 most important features: {list(important_features)}")
    
    # Test robustness by perturbing important features
    X_test_perturbed = X_test.copy()
    for feature in important_features:
        if feature in X_test_perturbed.columns:
            # Add targeted noise to important features
            noise_std = X_test_perturbed[feature].std() * 0.1  # 10% of std as noise
            X_test_perturbed[feature] += np.random.normal(0, noise_std, size=len(X_test_perturbed))
    
    y_pred_perturbed = mlp_model.predict(X_test_perturbed)
    rmse_perturbed = np.sqrt(mean_squared_error(y_test, y_pred_perturbed))
    
    print(f"RMSE with perturbed important features: {rmse_perturbed:.4f}")
    print(f"Performance degradation from feature perturbation: {((rmse_perturbed - rmse_original) / rmse_original * 100):.2f}%")

except Exception as e:
    print(f"Could not perform feature importance analysis: {e}")

# Adversarial robustness test using gradient-based perturbations (simplified)
def simple_adversarial_test(model, X_test, y_test, epsilon=0.01):
    """Simple adversarial robustness test"""
    X_adv = X_test.copy()
    
    # Add small perturbations in the direction that maximizes error
    for i in range(len(X_test)):
        original_pred = model.predict([X_test.iloc[i]])[0]
        best_perturbation = None
        max_error = 0
        
        # Try small perturbations in different directions (simplified approach)
        for _ in range(10):  # Limited iterations for demonstration
            perturbation = np.random.normal(0, epsilon, size=X_test.shape[1])
            X_perturbed = X_test.iloc[i].values + perturbation
            
            try:
                new_pred = model.predict([X_perturbed])[0]
                error = abs(new_pred - original_pred)
                
                if error > max_error:
                    max_error = error
                    best_perturbation = perturbation
            except:
                continue
        
        if best_perturbation is not None:
            X_adv.iloc[i] = X_test.iloc[i] + best_perturbation

    return X_adv

print("\nTesting adversarial robustness...")
try:
    X_test_adv = simple_adversarial_test(mlp_model, X_test, y_test, epsilon=0.01)
    y_pred_adv = mlp_model.predict(X_test_adv)
    rmse_adv = np.sqrt(mean_squared_error(y_test, y_pred_adv))
    
    print(f"RMSE under adversarial attack: {rmse_adv:.4f}")
    print(f"Adversarial robustness gap: {((rmse_adv - rmse_original) / rmse_original * 100):.2f}%")
except Exception as e:
    print(f"Adversarial robustness test failed: {e}")

print("\n✅ Pipeline Complete!")
print("=" * 50)
print(f"Final model performance summary:")
print(f"- Original RMSE: {rmse_original:.4f}")
print(f"- Model is trained on {X_train.shape[0]} samples with {X_train.shape[1]} features")
print(f"- Privacy protection measures applied")
print(f"- Fairness metrics evaluated")
print(f"- Robustness tests completed")
`
          },
        ],
      },
    ],
  }

  const toggleAnswers = (notebookId) => {
    setShowAnswers((prev) => ({
      ...prev,
      [notebookId]: !prev[notebookId],
    }))
  }

  const copyCode = async (code, notebookId, answerIndex) => {
    try {
      await navigator.clipboard.writeText(code)
      setCopiedCode({ [`${notebookId}-${answerIndex}`]: true })
      setTimeout(() => {
        setCopiedCode((prev) => ({ ...prev, [`${notebookId}-${answerIndex}`]: false }))
      }, 2000)
    } catch (err) {
      console.error("Failed to copy code:", err)
    }
  }

  return (
    <div
    className={`min-h-screen transition-all duration-1000 ${
      isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
    }`}
    >
      {/* Hero Section */}
      <section className="bg-purple-700 text-white py-16 mt-8 mb-6 overflow-hidden">
        <div className="container mx-auto px-4 text-center">
          <div className="flex items-center justify-center gap-3 mb-6">
            <h1 className="text-5xl font-medium text-white">AI Ethics Curriculum</h1>
          </div>

          <div className="max-w-4xl mx-auto space-y-6">
            <p className="text-xl font-normal leading-relaxed">
              An accessible introduction to the fundamentals of ethical AI for students and beginners in computer science.
            </p>
            <p className="text-lg font-light text-white/90">
              Learn how to approach AI with responsibility and care—developing systems that are fair, transparent, and aligned with societal values.
            </p>
          </div>
        </div>
      </section>

      {/* Curriculum Description */}
      <section className="py-16 bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl font-medium text-gray-800 mb-8 text-center">Course Overview</h2>

            <div className="grid md:grid-cols-2 gap-8 mb-12">
              <div className="text-center p-6 bg-purple-100 rounded-2xl">
                <Play className="h-12 w-12 text-purple-600 mx-auto mb-4" />
                <h3 className="text-xl font-medium text-gray-800 mb-2">Video Lectures</h3>
                <p className="text-black font-light">Comprehensive video content covering core concepts</p>
              </div>
              <div className="text-center p-6 bg-purple-100 rounded-2xl">
                <BookOpen className="h-12 w-12 text-purple-600 mx-auto mb-4" />
                <h3 className="text-xl font-medium text-gray-800 mb-2">Hands-on Labs</h3>
                <p className="text-black font-light">Interactive Colab notebooks with practical exercises</p>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 shadow-lg">
            <h3 className="text-2xl font-medium text-gray-800 mb-4 text-center">What You'll Learn</h3>
            <div className="grid md:grid-cols-2 gap-6 text-black font-light">
              <ul className="space-y-3">
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span>Core concepts in data science, including cleaning data and visualizing distributions</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span>Introduction to machine learning, its types, workflows, and ethical considerations</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span>Understanding bias and fairness through metrics, group/individual fairness models</span>
                </li>
              </ul>
              <ul className="space-y-3">
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span>Hands-on robustness testing with adversarial attacks and noise-based methods</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span>Exploration of the regulatory landscape surrounding AI safety and accountability</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span>Apply your learning in a capstone project on real-world issues like hospital or scholarship allocation</span>
                </li>
              </ul>
            </div>
          </div>
          
          <div className="bg-purple-100 border rounded-2xl p-6 mt-8">
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className=" text-sm font-bold">💡</span>
              </div>
              <div>
                <h4 className="text-lg font-medium  mb-2">A Note About Coding</h4>
                <p className="font-light leading-relaxed">
                  There are many different ways to write code to solve the same problem. The code examples provided are just one approach—what matters most is that you understand what you're writing and that it works correctly. Don't worry if your solution looks different from the examples; as long as you can explain your approach and it produces the right results, you're on the right track!
                </p>
              </div>
            </div>
          </div>
          </div>
        </div>
      </section>

      <section className="py-8">
        <div className="container mx-auto px-4">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-4xl font-medium text-gray-800 mb-12 text-center">Video Lectures</h2>

            <div className="grid md:grid-cols-2 gap-8 w-[90%] mx-auto">
              {curriculumData.videos.map((video) => (
                <div
                  key={video.id}
                  className="bg-white rounded-2xl shadow-lg overflow-hidden hover:shadow-xl transition-all duration-300"
                >
                  <div className="relative">
                    <Image
                      src={video.thumbnail || "/placeholder.svg"}
                      alt={video.title}
                      width={350}
                      height={200}
                      className="w-full h-48 object-cover"
                    />
                    <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                      <button
                        onClick={() => window.open(`https://youtube.com/watch?v=${video.youtubeId}`, "_blank")}
                        className="bg-red-600 hover:bg-red-700 text-white p-4 rounded-full transition-colors duration-200"
                      >
                        <Play className="h-8 w-8" />
                      </button>
                    </div>
                    <div className="absolute top-4 right-4 bg-black/70 text-white px-2 py-1 rounded text-sm flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {video.duration}
                    </div>
                  </div>

                  <div className="p-6">
                    <h3 className="text-xl font-medium mb-3 text-center">{video.title}</h3>
                    <p className="font-light mb-4 text-sm leading-relaxed text-center">{video.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-4xl font-medium text mb-4 text-center">Hands-on Labs</h2>
            <p className="font-light text-sm mb-12 text-center italic">Make sure to make a copy of the notebook before starting</p>

            <div className="grid lg:grid-cols-1 gap-8">
              {curriculumData.notebooks.map((notebook) => (
                <div
                  key={notebook.id}
                  className="bg-white rounded-2xl shadow-lg p-8 hover:shadow-xl transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <h3 className="text-2xl font-normal text-gray-800 mb-2">{notebook.title}</h3>
                      <p className="font-light mb-4">{notebook.description}</p>
                      <div className="flex items-center gap-2 text-sm text-purple-600 mb-4">
                        <span>Related video:</span>
                        {notebook.relatedVideos.map((videoId, index) => (
                          <span key={videoId} className="bg-purple-100 px-2 py-1 rounded">
                            Module {videoId}
                            {index < notebook.relatedVideos.length - 1 && ","}
                          </span>
                        ))}
                      </div>
                    </div>
                    <BookOpen className="h-8 w-8 text-blue-500 flex-shrink-0" />
                  </div>

                  <div className="mb-6">
                    <h4 className="text-lg font-normal mb-3">Practice Activities</h4>
                    <ul className="space-y-2">
                      {notebook.practiceActivities.map((activity, index) => (
                        <li key={index} className="flex items-start gap-3 text-sm">
                          <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                          <span>{activity}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="mb-6 border-t pt-4">
                    <button
                      onClick={() => toggleAnswers(notebook.id)}
                      className="flex items-center gap-2 text-sm font-medium text-gray-700 hover:text-gray-900 mb-3"
                    >
                      {showAnswers[notebook.id] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      {showAnswers[notebook.id] ? "Hide" : "Show"} Code Solutions
                    </button>

                    {showAnswers[notebook.id] && (
                      <div className="space-y-6">
                        {notebook.codeAnswers.map((answer, index) => (
                          <div key={index} className="bg-gray-50 border border-gray-200 rounded-lg overflow-hidden">
                            <div className="bg-gray-100 px-4 py-2 border-b border-gray-200 flex items-center justify-between">
                              <h5 className="font-medium text-gray-800 text-sm">{answer.question}</h5>
                              <button
                                onClick={() => copyCode(answer.code, notebook.id, index)}
                                className="flex items-center gap-1 text-xs text-gray-600 hover:text-gray-800 transition-colors"
                              >
                                {copiedCode[`${notebook.id}-${index}`] ? (
                                  <>
                                    <Check className="h-3 w-3" />
                                    Copied!
                                  </>
                                ) : (
                                  <>
                                    <Copy className="h-3 w-3" />
                                    Copy
                                  </>
                                )}
                              </button>
                            </div>
                            <pre className="p-4 text-xs text-gray-800 overflow-x-auto bg-white">
                              <code>{answer.code}</code>
                            </pre>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Action Button */}
                  <div className="flex justify-end">
                    {notebook.colabUrls ? (
                      <div className="flex gap-2">
                        {notebook.colabUrls.map((colabLink, index) => (
                          <button
                            key={index}
                            onClick={() => window.open(colabLink.url, "_blank")}
                            className="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors duration-200"
                          >
                            {colabLink.title}
                            <ExternalLink className="h-3 w-3" />
                          </button>
                        ))}
                      </div>
                    ) : (
                      <button
                        onClick={() => window.open(notebook.colabUrl, "_blank")}
                        className="bg-orange-500 hover:bg-orange-600 text-white px-6 py-2 rounded-full text-sm font-medium flex items-center gap-2 transition-colors duration-200"
                      >
                        Open in Colab <ExternalLink className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
