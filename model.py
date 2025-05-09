"""
Autism Detection ML Model

This script trains a machine learning model on the autism dataset and saves it
for use in the Streamlit application.
"""

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.ensemble import RandomForestClassifier  # ML algorithm for classification
from sklearn.preprocessing import StandardScaler  # For scaling features
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # For model evaluation
import joblib  # For saving/loading models
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns  # For enhanced visualizations
import os  # For file operations












# SECTION 1: MAIN TRAINING FUNCTION
def train_autism_model(data_path='train.csv'):
    """
    Train a machine learning model for autism detection.
    
    Args:
        data_path: Path to the CSV file containing the training data
        
    Returns:
        Trained model and related artifacts
    """
    # Read the dataset
    data = pd.read_csv(data_path)  # Load CSV data into pandas DataFrame
    
    # Basic information about the dataset
    print("Dataset shape:", data.shape)  # Show rows and columns in dataset
    print("\nClass distribution:")
    print(data['Class/ASD'].value_counts())  # Count of autism vs non-autism cases
    print(data['Class/ASD'].value_counts(normalize=True) * 100)  # Percentage distribution
    
    # Convert gender to numeric (f=0, m=1) for the model to process
    data['gender_numeric'] = data['gender'].map({'f': 0, 'm': 1})
    
    # Define the A-score features (the 10 autism screening questions)
    features = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 
               'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']
    
    # Fill missing values with median to maintain data integrity
    for feature in features:
        if data[feature].isnull().sum() > 0:
            data[feature].fillna(data[feature].median(), inplace=True)
    
    # Fill missing age values with the median age
    data['age'].fillna(data['age'].median(), inplace=True)
    
    # Fill missing gender_numeric values with mode (most common value)
    if 'gender_numeric' in data.columns and data['gender_numeric'].isnull().sum() > 0:
        data['gender_numeric'].fillna(data['gender_numeric'].mode()[0], inplace=True)
    
    # Select relevant features for the model
    X = data[features + ['age', 'gender_numeric']]  # Features for prediction
    y = data['Class/ASD']  # Target variable (autism diagnosis)
    
    print("\nSelected features:")
    print(X.head())  # Display first 5 rows of features
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features to have mean=0 and variance=1 (important for ML algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit to training data and transform it
    X_test_scaled = scaler.transform(X_test)  # Apply same scaling to test data
    
    # Train a Random Forest classifier (ensemble learning method)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Create model with 100 trees
    rf_model.fit(X_train_scaled, y_train)  # Train the model on scaled training data
    
    # Make predictions on the test set
    y_pred = rf_model.predict(X_test_scaled)  # Get predicted classes
    y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]  # Get probability of positive class
    
    # Evaluate the model's performance
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))  # Percentage of correct predictions
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))  # Precision, recall, F1-score
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))  # True/false positives/negatives
    
    # Calculate feature importance (which features most influence predictions)
    feature_importance = pd.DataFrame(
        {'Feature': X.columns, 'Importance': rf_model.feature_importances_}
    ).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Group the A1-A10 scores into four domains based on clinical relevance
    domain_mapping = {
        'pattern_recognition': ['A1_Score', 'A5_Score', 'A6_Score'],
        'sensory_processing': ['A7_Score', 'A9_Score'],
        'social_communication': ['A2_Score', 'A4_Score', 'A10_Score'],
        'repetitive_behaviors': ['A3_Score', 'A8_Score']
    }
    
    # Calculate domain scores for each sample by averaging the relevant questions
    for domain, domain_features in domain_mapping.items():
        data[domain + '_score'] = data[domain_features].sum(axis=1) / len(domain_features) * 10
        
    # Display the domain scores statistics
    domain_scores = ['pattern_recognition_score', 'sensory_processing_score', 
                    'social_communication_score', 'repetitive_behaviors_score']
    print("\nDomain scores:")
    print(data[domain_scores].describe())  # Min, max, mean, etc. of domain scores
    
    # Train a model using only the domain scores (more interpretable for users)
    X_domain = data[domain_scores]  # Features are now the domain scores
    X_domain_train, X_domain_test, y_domain_train, y_domain_test = train_test_split(
        X_domain, y, test_size=0.2, random_state=42)  # Same train/test split structure
    
    # Scale the domain features
    scaler_domain = StandardScaler()
    X_domain_train_scaled = scaler_domain.fit_transform(X_domain_train)
    X_domain_test_scaled = scaler_domain.transform(X_domain_test)
    
    # Train a Random Forest classifier on domain scores
    rf_domain_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_domain_model.fit(X_domain_train_scaled, y_domain_train)
    
    # Make predictions with the domain-based model
    y_domain_pred = rf_domain_model.predict(X_domain_test_scaled)
    y_domain_proba = rf_domain_model.predict_proba(X_domain_test_scaled)[:, 1]
    
    # Evaluate the domain-based model
    print("\nDomain-based Model Evaluation:")
    print("Accuracy:", accuracy_score(y_domain_test, y_domain_pred))
    print("\nClassification Report:")
    print(classification_report(y_domain_test, y_domain_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_domain_test, y_domain_pred))
    
    # Calculate domain importance for the domain model
    domain_importance = pd.DataFrame(
        {'Domain': X_domain.columns, 'Importance': rf_domain_model.feature_importances_}
    ).sort_values('Importance', ascending=False)
    
    print("\nDomain Importance:")
    print(domain_importance)
    
    # Calculate average domain scores for each class (autism vs non-autism)
    domain_averages = {}
    for class_val in [0, 1]:  # 0=non-autism, 1=autism
        domain_averages[class_val] = {}
        for domain in domain_scores:
            domain_averages[class_val][domain] = data[data['Class/ASD'] == class_val][domain].mean()
    
    print("\nDomain averages by class:")
    print(domain_averages)
    
    # Calculate weights for each domain based on importance and separation between classes
    domain_weights = {}
    for domain in domain_scores:
        # Weight based on importance and class separation
        importance = domain_importance[domain_importance['Domain'] == domain]['Importance'].values[0]
        separation = abs(domain_averages[1][domain] - domain_averages[0][domain])
        domain_weights[domain] = importance * separation
    
    # Normalize weights to sum to 1
    total_weight = sum(domain_weights.values())
    for domain in domain_weights:
        domain_weights[domain] /= total_weight
    
    print("\nDomain weights:")
    print(domain_weights)
    
    # Create directory for saving models if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the main model and scaler
    joblib.dump(rf_model, 'models/autism_model.pkl')
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    
    # Save the domain-based model and scaler
    joblib.dump(rf_domain_model, 'models/autism_domain_model.pkl')
    joblib.dump(scaler_domain, 'models/domain_scaler.pkl')
    
    # Save domain mappings and weights for interpretation
    joblib.dump(domain_mapping, 'models/domain_mapping.pkl')
    joblib.dump(domain_weights, 'models/domain_weights.pkl')
    joblib.dump(domain_averages, 'models/domain_averages.pkl')
    
    print("\nModels and artifacts saved to 'models' directory")
    
    # Create visualization of domain scores by class for reporting
    plt.figure(figsize=(12, 8))
    for i, domain in enumerate(domain_scores):
        plt.subplot(2, 2, i+1)  # Create 2x2 grid of plots
        sns.boxplot(x='Class/ASD', y=domain, data=data)  # Box plot showing distribution
        plt.title(domain)
    plt.tight_layout()
    plt.savefig('models/domain_score_distributions.png')
    
    # Return all artifacts for potential immediate use
    return {
        'model': rf_model,
        'scaler': scaler,
        'domain_model': rf_domain_model,
        'domain_scaler': scaler_domain,
        'domain_mapping': domain_mapping,
        'domain_weights': domain_weights,
        'domain_averages': domain_averages
    }
















# SECTION 2: TEXT ANALYSIS FUNCTION
def analyze_text_responses(text_content):
    """
    Analyze text responses for autism indicators.
    
    Args:
        text_content: String containing text to analyze
    
    Returns:
        Dictionary with text analysis results
    """
    # Define keywords associated with autism characteristics by domain
    autism_keywords = {
        'pattern_recognition': ['detail', 'pattern', 'notice', 'specific', 'organize', 'order', 'arrange', 'categorize', 'line up', 'sort'],
        'sensory_processing': ['loud', 'noise', 'bright', 'light', 'texture', 'touch', 'smell', 'taste', 'sensitive', 'overwhelm'],
        'social_communication': ['eye contact', 'literal', 'understand', 'social', 'conversation', 'friend', 'interact', 'play', 'share', 'emotion'],
        'repetitive_behaviors': ['routine', 'change', 'upset', 'repeat', 'interest', 'spin', 'flap', 'rock', 'ritual', 'same']
    }
    
    # Initialize domain scores to zero
    domain_scores = {
        'pattern_recognition': 0,
        'sensory_processing': 0,
        'social_communication': 0,
        'repetitive_behaviors': 0
    }
    
    # Lowercase the text for case-insensitive matching
    text_content = text_content.lower()
    
    # Count keywords in each domain and calculate domain scores
    for domain, keywords in autism_keywords.items():
        count = 0
        for keyword in keywords:
            if keyword in text_content:  # Check if keyword is in the text
                count += 1
        # Calculate score as percentage of keywords found (scaled to 0-10)
        domain_scores[domain] = min(10, count * 10 / len(keywords))
    
    # Generate insights based on text analysis patterns
    insights = []
    
    # Check for routine resistance indicators
    if 'routine' in text_content and ('upset' in text_content or 'difficult' in text_content or 'distress' in text_content):
        insights.append("Your description suggests your child may find changes in routine challenging, which is common in autism.")
    
    # Check for intense interests
    if 'interest' in text_content and ('intense' in text_content or 'deep' in text_content or 'focus' in text_content):
        insights.append("You've described focused interests that appear to be particularly intense or deep, which is often seen in autism.")
    
    # Check for social challenges
    if ('social' in text_content or 'interact' in text_content) and ('challenge' in text_content or 'difficult' in text_content or 'avoid' in text_content):
        insights.append("Your description indicates some social interaction challenges that align with autism characteristics.")
    
    # Calculate overall text score (average across domains)
    overall_score = sum(domain_scores.values()) / len(domain_scores)
    
    # Return structured results
    return {
        'domain_scores': domain_scores,
        'overall_score': overall_score,
        'insights': insights
    }























# SECTION 3: AGE-SPECIFIC RECOMMENDATIONS FUNCTION
def get_age_specific_recommendations(age, domain_scores):
    """
    Generate age-specific recommendations based on domain scores.
    
    Args:
        age: Child's age (float or string)
        domain_scores: Dictionary of domain scores
        
    Returns:
        List of age-appropriate recommendations
    """
    recommendations = []
    try:
        age_float = float(age)  # Convert age to float for comparison
    except (ValueError, TypeError):
        # Default to older child if age can't be determined
        age_float = 10
    
    # Pattern Recognition recommendations based on age group
    pattern_score = domain_scores.get('pattern_recognition', 5)
    if pattern_score >= 7:  # Only make recommendations for high scores
        if age_float < 5:  # Young children
            recommendations.append("For young children with strong pattern recognition: Consider visual schedules and structured play activities that leverage their detail-oriented thinking.")
        elif age_float < 12:  # School-age children
            recommendations.append("For school-age children with strong pattern recognition: Consider STEM activities, puzzles, or music lessons that build on their systematic thinking abilities.")
        else:  # Adolescents
            recommendations.append("For adolescents with strong pattern recognition: Consider coding, design, music theory, or mathematics courses that leverage their detail-oriented thinking.")
    
    # Sensory Processing recommendations based on age group
    sensory_score = domain_scores.get('sensory_processing', 5)
    if sensory_score >= 7:
        if age_float < 5:
            recommendations.append("For young children with sensory sensitivities: Create a 'sensory toolbox' with items like noise-canceling headphones, weighted lap pads, and fidget toys.")
        elif age_float < 12:
            recommendations.append("For school-age children with sensory sensitivities: Work with teachers to establish sensory breaks and accommodations in the classroom.")
        else:
            recommendations.append("For adolescents with sensory sensitivities: Teach self-advocacy skills for managing sensory needs in different environments.")
    
    # Social Communication recommendations based on age group
    social_score = domain_scores.get('social_communication', 5)
    if social_score >= 7:
        if age_float < 5:
            recommendations.append("For young children with social communication differences: Focus on play-based interaction with clear, simple language and visual supports.")
        elif age_float < 12:
            recommendations.append("For school-age children with social communication differences: Consider social skills groups and use of social stories to explain unwritten social rules.")
        else:
            recommendations.append("For adolescents with social communication differences: Consider peer mentoring programs and explicit teaching of conversational turn-taking and social context.")
    
    # Repetitive Behaviors recommendations based on age group
    repetitive_score = domain_scores.get('repetitive_behaviors', 5)
    if repetitive_score >= 7:
        if age_float < 5:
            recommendations.append("For young children with focused interests: Incorporate special interests into learning activities and use interests as motivation for new experiences.")
        elif age_float < 12:
            recommendations.append("For school-age children with focused interests: Help channel interests into clubs, projects, or structured learning opportunities.")
        else:
            recommendations.append("For adolescents with focused interests: Connect special interests to potential career paths and constructive hobbies.")
    
    return recommendations























# SECTION 4: MAIN PREDICTION FUNCTION
def predict_autism_risk(responses, text_content=None):
    """
    Predict autism risk based on question responses and optional text.
    
    Args:
        responses: Dictionary with domain scores
        text_content: Optional text to analyze
        
    Returns:
        Dictionary with autism risk assessment
    """
    # Load model artifacts
    try:
        # Load saved model files
        rf_domain_model = joblib.load('models/autism_domain_model.pkl')
        scaler_domain = joblib.load('models/domain_scaler.pkl')
        domain_weights = joblib.load('models/domain_weights.pkl')
        domain_averages = joblib.load('models/domain_averages.pkl')
    except Exception as e:
        # Fallback if models can't be loaded
        print(f"Error loading model files: {e}")
        return {
            'probability': sum(responses.values()) / (10 * len(responses)),
            'classification': 0,
            'overall_score': sum(responses.values()) / len(responses),
            'domain_scores': responses
        }
    
    # Get domain scores from user responses
    domain_scores = {
        'pattern_recognition_score': responses.get('pattern_recognition', 5),
        'sensory_processing_score': responses.get('sensory_processing', 5),
        'social_communication_score': responses.get('social_communication', 5),
        'repetitive_behaviors_score': responses.get('repetitive_behaviors', 5)
    }
    
    # Convert to DataFrame for prediction (same format as trained)
    import pandas as pd
    domain_array = pd.DataFrame([
        [domain_scores['pattern_recognition_score'],
        domain_scores['sensory_processing_score'],
        domain_scores['social_communication_score'],
        domain_scores['repetitive_behaviors_score']]
    ], columns=['pattern_recognition_score', 'sensory_processing_score', 
            'social_communication_score', 'repetitive_behaviors_score'])

    # Scale the domain scores using the saved scaler
    domain_array_scaled = scaler_domain.transform(domain_array)
    
    # Predict probability of autism
    probability = rf_domain_model.predict_proba(domain_array_scaled)[0, 1]
    
    # Calculate overall score using weighted average
    overall_score = 0
    for domain, score in domain_scores.items():
        weight = domain_weights[domain]
        overall_score += score * weight
    
    # Process text content if provided for additional insights
    text_analysis = None
    if text_content:
        text_analysis = analyze_text_responses(text_content)
        # Adjust probability with text analysis (15% weight to text)
        text_weight = 0.15
        probability = (probability * (1 - text_weight)) + (text_analysis['overall_score'] / 10 * text_weight)
    
    # Calculate domain percentiles (for visualizing where scores fall relative to typical/autistic ranges)
    domain_percentiles = {}
    for domain, score in domain_scores.items():
        non_asd_avg = domain_averages[0][domain]  # Average for non-ASD group
        asd_avg = domain_averages[1][domain]  # Average for ASD group
        
        # Simple percentile calculation based on position between averages
        if score <= non_asd_avg:
            percentile = 0.25 * (score / non_asd_avg)
        elif score >= asd_avg:
            percentile = 0.75 + 0.25 * min(1, (score - asd_avg) / asd_avg)
        else:
            range_size = asd_avg - non_asd_avg
            position = (score - non_asd_avg) / range_size
            percentile = 0.25 + position * 0.5
        
        domain_percentiles[domain.replace('_score', '')] = percentile
    
    # Build the results dictionary
    result = {
        'probability': probability,
        'classification': 1 if probability >= 0.5 else 0,  # Binary classification (0=non-ASD, 1=ASD)
        'overall_score': overall_score,
        'domain_scores': {
            'pattern_recognition': domain_scores['pattern_recognition_score'],
            'sensory_processing': domain_scores['sensory_processing_score'],
            'social_communication': domain_scores['social_communication_score'],
            'repetitive_behaviors': domain_scores['repetitive_behaviors_score']
        },
        'domain_percentiles': domain_percentiles
    }
    
    # Add text insights if text analysis was performed
    if text_analysis:
        result['text_insights'] = text_analysis['insights']
    
    # Add age-specific recommendations if age is provided
    if 'age' in responses:
        result['age_recommendations'] = get_age_specific_recommendations(responses.get('age'), result['domain_scores'])
    
    return result















# SECTION 5: SCRIPT EXECUTION SECTION
if __name__ == "__main__":
    # Train the model when script is run directly
    model_artifacts = train_autism_model('train.csv')
    
    # Test the model with example responses
    example_responses = {
        'pattern_recognition': 9.5,
        'sensory_processing': 9.2,
        'social_communication': 8.8,
        'repetitive_behaviors': 8.9
    }
    
    # Get prediction for the example
    prediction = predict_autism_risk(example_responses)
    
    # Print the results
    print("\nExample prediction:")
    print(f"Probability of autism: {prediction['probability']:.2f}")
    print(f"Classification: {'ASD' if prediction['classification'] == 1 else 'Non-ASD'}")
    print(f"Overall score: {prediction['overall_score']:.2f}")
    print("Domain scores:", prediction['domain_scores'])
    print("Domain percentiles:", prediction['domain_percentiles'])