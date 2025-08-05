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
            question: "Load and explore the Adult Census dataset",
            code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Adult Census dataset
df = pd.read_csv('adult.csv')

# Basic exploration
print("Dataset shape:", df.shape)
print("\\nGender distribution:")
print(df['sex'].value_counts())
print("\\nRace distribution:")
print(df['race'].value_counts())

# Check for missing values
print("\\nMissing values:")
print(df.isnull().sum())`,
          },
          {
            question: "Calculate statistical parity difference",
            code: `def statistical_parity_difference(df, protected_attr, target_attr):
    """Calculate statistical parity difference"""
    groups = df[protected_attr].unique()
    
    positive_rates = {}
    for group in groups:
        group_data = df[df[protected_attr] == group]
        positive_rate = (group_data[target_attr] == '>50K').mean()
        positive_rates[group] = positive_rate
        print(f"{group}: {positive_rate:.3f}")
    
    # Calculate difference between max and min rates
    spd = max(positive_rates.values()) - min(positive_rates.values())
    print(f"\\nStatistical Parity Difference: {spd:.3f}")
    return spd

# Calculate for gender
print("Statistical Parity by Gender:")
spd_gender = statistical_parity_difference(df, 'sex', 'income')`,
          },
          {
            question: "Visualize bias patterns across groups",
            code: `# Create visualization of bias patterns
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gender distribution
axes[0,0].pie(df['sex'].value_counts(), labels=df['sex'].value_counts().index, autopct='%1.1f%%')
axes[0,0].set_title('Gender Distribution')

# Income by gender
income_gender = pd.crosstab(df['sex'], df['income'], normalize='index')
income_gender.plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Income Distribution by Gender')
axes[0,1].set_ylabel('Proportion')

# Race distribution
df['race'].value_counts().plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Race Distribution')
axes[1,0].tick_params(axis='x', rotation=45)

# Income by race
income_race = pd.crosstab(df['race'], df['income'], normalize='index')
income_race.plot(kind='bar', ax=axes[1,1])
axes[1,1].set_title('Income Distribution by Race')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
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
            question: "Implement demographic parity metric",
            code: `def demographic_parity(y_true, y_pred, sensitive_attr):
    """
    Calculate demographic parity metric
    Returns the difference in positive prediction rates between groups
    """
    groups = np.unique(sensitive_attr)
    positive_rates = {}
    
    for group in groups:
        group_mask = sensitive_attr == group
        group_predictions = y_pred[group_mask]
        positive_rate = np.mean(group_predictions)
        positive_rates[group] = positive_rate
        print(f"Group {group}: Positive rate = {positive_rate:.3f}")
    
    # Calculate demographic parity difference
    dp_diff = max(positive_rates.values()) - min(positive_rates.values())
    print(f"\\nDemographic Parity Difference: {dp_diff:.3f}")
    
    return dp_diff, positive_rates

# Example usage
dp_diff, rates = demographic_parity(y_true, y_pred, sensitive_feature)`,
          },
          {
            question: "Implement equalized odds metric",
            code: `def equalized_odds(y_true, y_pred, sensitive_attr):
    """
    Calculate equalized odds metric
    Measures difference in TPR and FPR across groups
    """
    from sklearn.metrics import confusion_matrix
    
    groups = np.unique(sensitive_attr)
    tpr_diff_max = 0
    fpr_diff_max = 0
    
    tprs = {}
    fprs = {}
    
    for group in groups:
        group_mask = sensitive_attr == group
        y_true_group = y_true[group_mask]
        y_pred_group = y_pred[group_mask]
        
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tprs[group] = tpr
        fprs[group] = fpr
        
        print(f"Group {group}: TPR = {tpr:.3f}, FPR = {fpr:.3f}")
    
    tpr_diff = max(tprs.values()) - min(tprs.values())
    fpr_diff = max(fprs.values()) - min(fprs.values())
    
    print(f"\\nTPR Difference: {tpr_diff:.3f}")
    print(f"FPR Difference: {fpr_diff:.3f}")
    
    return tpr_diff, fpr_diff

# Example usage
tpr_diff, fpr_diff = equalized_odds(y_true, y_pred, sensitive_feature)`,
          },
          {
            question: "Compare fairness-accuracy trade-offs",
            code: `def fairness_accuracy_tradeoff(X, y, sensitive_attr, thresholds):
    """
    Analyze trade-off between fairness and accuracy
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    results = []
    
    for threshold in thresholds:
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Get prediction probabilities
        y_prob = model.predict_proba(X)[:, 1]
        
        # Apply threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        dp_diff, _ = demographic_parity(y, y_pred, sensitive_attr)
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'demographic_parity_diff': dp_diff
        })
        
        print(f"Threshold: {threshold:.2f}, Accuracy: {accuracy:.3f}, DP Diff: {dp_diff:.3f}")
    
    return results

# Analyze trade-offs
thresholds = np.arange(0.3, 0.8, 0.05)
tradeoff_results = fairness_accuracy_tradeoff(X_train, y_train, sensitive_train, thresholds)`,
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
            question: "Implement Laplace mechanism for differential privacy",
            code: `import numpy as np
from scipy.stats import laplace

def laplace_mechanism(true_answer, sensitivity, epsilon):
    """
    Add Laplace noise for differential privacy
    
    Args:
        true_answer: The true value to be privatized
        sensitivity: The sensitivity of the query
        epsilon: Privacy parameter (smaller = more private)
    """
    # Calculate scale parameter
    scale = sensitivity / epsilon
    
    # Generate Laplace noise
    noise = np.random.laplace(0, scale)
    
    # Add noise to true answer
    private_answer = true_answer + noise
    
    return private_answer

# Example: Private count query
true_count = 1000
sensitivity = 1  # Adding/removing one person changes count by at most 1
epsilon = 0.1    # Strong privacy

private_count = laplace_mechanism(true_count, sensitivity, epsilon)
print(f"True count: {true_count}")
print(f"Private count: {private_count:.2f}")
print(f"Error: {abs(private_count - true_count):.2f}")`,
          },
          {
            question: "Compare privacy-utility trade-offs",
            code: `def privacy_utility_analysis(true_values, sensitivity, epsilons, num_trials=100):
    """
    Analyze privacy-utility trade-off for different epsilon values
    """
    results = []
    
    for epsilon in epsilons:
        errors = []
        
        for _ in range(num_trials):
            for true_val in true_values:
                private_val = laplace_mechanism(true_val, sensitivity, epsilon)
                error = abs(private_val - true_val)
                errors.append(error)
        
        avg_error = np.mean(errors)
        std_error = np.std(errors)
        
        results.append({
            'epsilon': epsilon,
            'avg_error': avg_error,
            'std_error': std_error,
            'privacy_level': 'High' if epsilon < 0.5 else 'Medium' if epsilon < 2.0 else 'Low'
        })
        
        print(f"ε = {epsilon:.1f}: Avg Error = {avg_error:.2f} ± {std_error:.2f}")
    
    return results

# Test different epsilon values
true_values = [100, 500, 1000, 2000]
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
analysis_results = privacy_utility_analysis(true_values, 1, epsilons)

# Visualize results
import matplotlib.pyplot as plt

epsilons_plot = [r['epsilon'] for r in analysis_results]
errors_plot = [r['avg_error'] for r in analysis_results]

plt.figure(figsize=(10, 6))
plt.plot(epsilons_plot, errors_plot, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Epsilon (Privacy Parameter)')
plt.ylabel('Average Error')
plt.title('Privacy-Utility Trade-off')
plt.grid(True, alpha=0.3)
plt.show()`,
          },
          {
            question: "Simple federated learning simulation",
            code: `class FederatedLearningSimulation:
    def __init__(self, num_clients=5):
        self.num_clients = num_clients
        self.global_model = None
        self.client_models = []
        
    def initialize_models(self, input_dim):
        """Initialize global and client models"""
        from sklearn.linear_model import SGDClassifier
        
        # Global model
        self.global_model = SGDClassifier(random_state=42)
        
        # Client models (copies of global model)
        self.client_models = [
            SGDClassifier(random_state=42) 
            for _ in range(self.num_clients)
        ]
    
    def distribute_data(self, X, y):
        """Distribute data among clients"""
        n_samples = len(X)
        samples_per_client = n_samples // self.num_clients
        
        client_data = []
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            
            if i == self.num_clients - 1:  # Last client gets remaining data
                end_idx = n_samples
                
            client_data.append((X[start_idx:end_idx], y[start_idx:end_idx]))
        
        return client_data
    
    def local_training(self, client_data, epochs=5):
        """Train models locally on each client"""
        for i, (X_client, y_client) in enumerate(client_data):
            print(f"Training client {i+1} with {len(X_client)} samples")
            
            # Train local model
            for epoch in range(epochs):
                self.client_models[i].partial_fit(X_client, y_client, classes=np.unique(y_client))
    
    def federated_averaging(self):
        """Aggregate client models using federated averaging"""
        # Simple averaging of model coefficients
        if hasattr(self.client_models[0], 'coef_'):
            avg_coef = np.mean([model.coef_ for model in self.client_models], axis=0)
            avg_intercept = np.mean([model.intercept_ for model in self.client_models], axis=0)
            
            # Update global model
            self.global_model.coef_ = avg_coef
            self.global_model.intercept_ = avg_intercept
            
            print("Global model updated with federated averaging")

# Example usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize federated learning
fl_sim = FederatedLearningSimulation(num_clients=5)
fl_sim.initialize_models(X_train.shape[1])

# Distribute data and train
client_data = fl_sim.distribute_data(X_train, y_train)
fl_sim.local_training(client_data)
fl_sim.federated_averaging()

print("Federated learning simulation completed!")`,
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
            question: "LIME explanations for image classification",
            code: `# Install required packages
!pip install lime

import lime
import lime.lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from skimage.segmentation import mark_boundaries

def explain_image_prediction(model, image, class_names):
    """
    Generate LIME explanation for image classification
    """
    # Initialize LIME explainer
    explainer = lime.lime_image.LimeImageExplainer()
    
    # Generate explanation
    explanation = explainer.explain_instance(
        image, 
        model.predict_proba,
        top_labels=len(class_names),
        hide_color=0,
        num_samples=1000
    )
    
    # Get image and mask for top prediction
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=10, 
        hide_rest=False
    )
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # LIME explanation
    axes[1].imshow(mark_boundaries(temp, mask))
    axes[1].set_title('LIME Explanation')
    axes[1].axis('off')
    
    # Mask only
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Important Regions')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return explanation

# Example usage (assuming you have a trained image classifier)
# explanation = explain_image_prediction(model, test_image, ['cat', 'dog'])`,
          },
          {
            question: "SHAP explanations for tabular data",
            code: `# Install SHAP
!pip install shap

import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def shap_analysis(model, X_train, X_test, feature_names):
    """
    Comprehensive SHAP analysis for tabular data
    """
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # If binary classification, use positive class
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values
    
    # 1. Summary plot (global importance)
    print("1. Feature Importance Summary:")
    shap.summary_plot(shap_values_plot, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # 2. Waterfall plot for single prediction
    print("\\n2. Single Prediction Explanation:")
    shap.waterfall_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values_plot[0],
        X_test.iloc[0] if hasattr(X_test, 'iloc') else X_test[0],
        feature_names=feature_names,
        show=False
    )
    plt.title('SHAP Waterfall Plot - Single Prediction')
    plt.tight_layout()
    plt.show()
    
    # 3. Force plot for single prediction
    print("\\n3. Force Plot:")
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values_plot[0],
        X_test.iloc[0] if hasattr(X_test, 'iloc') else X_test[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title('SHAP Force Plot')
    plt.tight_layout()
    plt.show()
    
    # 4. Partial dependence plots
    print("\\n4. Partial Dependence:")
    for i, feature in enumerate(feature_names[:3]):  # Top 3 features
        shap.partial_dependence_plot(
            feature, model.predict_proba, X_train, ice=False,
            model_expected_value=True, feature_expected_value=True, show=False
        )
        plt.title(f'Partial Dependence: {feature}')
        plt.tight_layout()
        plt.show()
    
    return shap_values

# Example usage
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Generate SHAP explanations
shap_values = shap_analysis(model, X_train, X_test, feature_names)`,
          },
          {
            question: "Compare local vs global interpretability methods",
            code: `def interpretability_comparison(model, X_train, X_test, y_test, feature_names):
    """
    Compare different interpretability methods
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score
    
    print("=== INTERPRETABILITY METHODS COMPARISON ===\\n")
    
    # 1. Global Feature Importance (Model-specific)
    print("1. GLOBAL METHODS:")
    print("-" * 40)
    
    if hasattr(model, 'feature_importances_'):
        print("Built-in Feature Importance (Random Forest):")
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(importance_df.head(10))
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df.head(10)['feature'], importance_df.head(10)['importance'])
        plt.title('Global Feature Importance (Built-in)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    # 2. Permutation Importance (Model-agnostic)
    print("\\nPermutation Importance (Model-agnostic):")
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42
    )
    
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    print(perm_df.head(10))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(perm_df.head(10)['feature'], perm_df.head(10)['importance'])
    plt.title('Global Permutation Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    # 3. Local Methods
    print("\\n2. LOCAL METHODS:")
    print("-" * 40)
    
    # SHAP local explanation
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:5])  # First 5 samples
    
    if isinstance(shap_values, list):
        shap_values_local = shap_values[1]
    else:
        shap_values_local = shap_values
    
    print("SHAP Local Explanations (first 5 samples):")
    for i in range(5):
        print(f"\\nSample {i+1}:")
        sample_shap = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values_local[i]
        }).sort_values('shap_value', key=abs, ascending=False)
        print(sample_shap.head(5))
    
    # 4. Comparison Summary
    print("\\n3. METHOD COMPARISON:")
    print("-" * 40)
    
    comparison_data = {
        'Method': ['Built-in Importance', 'Permutation Importance', 'SHAP Global', 'SHAP Local'],
        'Scope': ['Global', 'Global', 'Global', 'Local'],
        'Model Agnostic': ['No', 'Yes', 'No*', 'No*'],
        'Computational Cost': ['Low', 'Medium', 'Medium', 'High'],
        'Interpretability': ['Medium', 'High', 'High', 'Very High']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    print("\\n* SHAP has model-agnostic versions (KernelExplainer)")
    
    return {
        'builtin_importance': importance_df if hasattr(model, 'feature_importances_') else None,
        'permutation_importance': perm_df,
        'shap_values': shap_values_local
    }

# Example usage
results = interpretability_comparison(model, X_train, X_test, y_test, feature_names)`,
          },
        ],
      },
      {
        id: 5,
        title: "Capstone Project",
        description:
          "Apply your learning in a comprehensive capstone project that addresses ethical AI challenges in healthcare allocation, scholarship distribution, or criminal justice reform.",
        colabUrl: "https://colab.research.google.com/drive/your-notebook-5",
        relatedVideos: [4],
        practiceActivities: [
          "Design and implement a fair hospital bed allocation system",
          "Create an equitable scholarship distribution algorithm",
          "Analyze and improve the COMPAS recidivism prediction tool",
        ],
        codeAnswers: [
          {
            question: "Implement a fair allocation system",
            code: `def fair_allocation_system(applicants, resources, fairness_constraints):
    """
    Design a fair allocation system for limited resources
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # Normalize applicant scores
    scaler = StandardScaler()
    normalized_scores = scaler.fit_transform(applicants[['score', 'need', 'urgency']])
    
    # Apply fairness constraints
    def calculate_fairness_score(applicant, group_representation):
        base_score = applicant['normalized_score']
        fairness_bonus = group_representation.get(applicant['group'], 0) * 0.1
        return base_score + fairness_bonus
    
    # Track group representation
    group_counts = {}
    total_allocated = 0
    
    # Sort by fairness-adjusted scores
    applicants['fairness_score'] = applicants.apply(
        lambda x: calculate_fairness_score(x, group_counts), axis=1
    )
    applicants_sorted = applicants.sort_values('fairness_score', ascending=False)
    
    # Allocate resources
    allocated = []
    for _, applicant in applicants_sorted.iterrows():
        if total_allocated < resources:
            allocated.append(applicant)
            total_allocated += 1
            
            # Update group representation
            group = applicant['group']
            group_counts[group] = group_counts.get(group, 0) + 1
    
    return allocated, group_counts

# Example usage
applicants_data = pd.DataFrame({
    'id': range(100),
    'score': np.random.normal(70, 15, 100),
    'need': np.random.uniform(0, 10, 100),
    'urgency': np.random.uniform(0, 10, 100),
    'group': np.random.choice(['A', 'B', 'C'], 100)
})

fair_allocations, group_stats = fair_allocation_system(
    applicants_data, 
    resources=50, 
    fairness_constraints={'A': 0.3, 'B': 0.3, 'C': 0.4}
)

print("Allocation complete!")
print("Group representation:", group_stats)`,
          },
          {
            question: "Evaluate fairness metrics across groups",
            code: `def evaluate_allocation_fairness(allocations, total_applicants):
    """
    Comprehensive fairness evaluation of allocation system
    """
    from sklearn.metrics import roc_auc_score
    
    # Calculate demographic parity
    def demographic_parity(allocations, group_attr):
        group_allocation_rates = allocations.groupby(group_attr)['allocated'].mean()
        dp_diff = group_allocation_rates.max() - group_allocation_rates.min()
        return dp_diff, group_allocation_rates
    
    # Calculate equal opportunity
    def equal_opportunity(allocations, group_attr, merit_attr):
        # Define high merit as top 50%
        merit_threshold = allocations[merit_attr].quantile(0.5)
        high_merit = allocations[allocations[merit_attr] >= merit_threshold]
        
        eo_rates = high_merit.groupby(group_attr)['allocated'].mean()
        eo_diff = eo_rates.max() - eo_rates.min()
        return eo_diff, eo_rates
    
    # Calculate calibration
    def calibration_fairness(allocations, group_attr, score_attr):
        calibration_errors = []
        for group in allocations[group_attr].unique():
            group_data = allocations[allocations[group_attr] == group]
            if len(group_data) > 10:  # Minimum sample size
                predicted = group_data[score_attr]
                actual = group_data['allocated']
                try:
                    auc = roc_auc_score(actual, predicted)
                    calibration_errors.append(abs(auc - 0.5))  # Distance from random
                except:
                    calibration_errors.append(0)
        
        return np.mean(calibration_errors) if calibration_errors else 0
    
    # Run evaluations
    dp_diff, dp_rates = demographic_parity(allocations, 'group')
    eo_diff, eo_rates = equal_opportunity(allocations, 'group', 'score')
    cal_error = calibration_fairness(allocations, 'group', 'fairness_score')
    
    print("=== FAIRNESS EVALUATION RESULTS ===")
    print(f"Demographic Parity Difference: {dp_diff:.3f}")
    print(f"Equal Opportunity Difference: {eo_diff:.3f}")
    print(f"Calibration Error: {cal_error:.3f}")
    
    # Fairness assessment
    fairness_score = 1 - (dp_diff + eo_diff + cal_error) / 3
    print(f"\\nOverall Fairness Score: {fairness_score:.3f}")
    
    if fairness_score > 0.8:
        print("✅ System is considered fair")
    elif fairness_score > 0.6:
        print("⚠️  System has moderate fairness issues")
    else:
        print("❌ System has significant fairness problems")
    
    return {
        'demographic_parity': dp_diff,
        'equal_opportunity': eo_diff,
        'calibration': cal_error,
        'overall_fairness': fairness_score
    }

# Evaluate the allocation system
fairness_results = evaluate_allocation_fairness(allocations_df, total_applicants)`,
          },
          {
            question: "Implement counterfactual fairness",
            code: `def counterfactual_fairness_analysis(allocations, sensitive_attr, features):
    """
    Analyze counterfactual fairness by examining how decisions change
    when sensitive attributes are modified
    """
    import copy
    
    def generate_counterfactuals(applicant, sensitive_attr, features):
        """Generate counterfactual scenarios for an applicant"""
        counterfactuals = []
        
        # Get all possible values for sensitive attribute
        possible_values = allocations[sensitive_attr].unique()
        
        for value in possible_values:
            if value != applicant[sensitive_attr]:
                # Create counterfactual applicant
                cf_applicant = applicant.copy()
                cf_applicant[sensitive_attr] = value
                counterfactuals.append(cf_applicant)
        
        return counterfactuals
    
    def predict_allocation(applicant, model):
        """Predict allocation decision for an applicant"""
        # This would use your trained allocation model
        # For demonstration, we'll use a simple rule-based approach
        score = applicant['score']
        need = applicant['need']
        urgency = applicant['urgency']
        
        # Simple allocation rule
        allocation_score = 0.4 * score + 0.3 * need + 0.3 * urgency
        return allocation_score > 70  # Threshold for allocation
    
    # Analyze counterfactual fairness
    counterfactual_results = []
    
    for idx, applicant in allocations.iterrows():
        if applicant['allocated']:  # Only analyze allocated applicants
            counterfactuals = generate_counterfactuals(applicant, sensitive_attr, features)
            
            original_decision = True  # They were allocated
            cf_decisions = []
            
            for cf in counterfactuals:
                cf_decision = predict_allocation(cf, None)  # No model for demo
                cf_decisions.append(cf_decision)
            
            # Check if any counterfactual would have different decision
            unfair_counterfactuals = [cf for cf in cf_decisions if cf != original_decision]
            
            counterfactual_results.append({
                'applicant_id': idx,
                'original_group': applicant[sensitive_attr],
                'counterfactual_groups': [cf[sensitive_attr] for cf in counterfactuals],
                'cf_decisions': cf_decisions,
                'unfair_cfs': len(unfair_counterfactuals),
                'is_counterfactually_fair': len(unfair_counterfactuals) == 0
            })
    
    # Calculate counterfactual fairness metrics
    total_analyzed = len(counterfactual_results)
    fair_decisions = sum(1 for result in counterfactual_results if result['is_counterfactually_fair'])
    cf_fairness_rate = fair_decisions / total_analyzed if total_analyzed > 0 else 0
    
    print("=== COUNTERFACTUAL FAIRNESS ANALYSIS ===")
    print(f"Total applicants analyzed: {total_analyzed}")
    print(f"Counterfactually fair decisions: {fair_decisions}")
    print(f"Counterfactual fairness rate: {cf_fairness_rate:.3f}")
    
    # Group-wise analysis
    group_cf_fairness = {}
    for group in allocations[sensitive_attr].unique():
        group_results = [r for r in counterfactual_results 
                        if r['original_group'] == group]
        if group_results:
            group_fair = sum(1 for r in group_results if r['is_counterfactually_fair'])
            group_cf_fairness[group] = group_fair / len(group_results)
    
    print("\\nGroup-wise counterfactual fairness:")
    for group, fairness in group_cf_fairness.items():
        print(f"  {group}: {fairness:.3f}")
    
    return {
        'overall_cf_fairness': cf_fairness_rate,
        'group_cf_fairness': group_cf_fairness,
        'detailed_results': counterfactual_results
    }

# Run counterfactual fairness analysis
cf_results = counterfactual_fairness_analysis(allocations, 'group', ['score', 'need', 'urgency'])`,
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
                    <button
                      onClick={() => window.open(notebook.colabUrl, "_blank")}
                      className="bg-orange-500 hover:bg-orange-600 text-white px-6 py-2 rounded-full text-sm font-medium flex items-center gap-2 transition-colors duration-200"
                    >
                      Open in Colab <ExternalLink className="h-3 w-3" />
                    </button>
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
