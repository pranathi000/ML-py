# Autism Assessment Tool

This repository contains a machine learning-based autism assessment tool implemented as a Streamlit web application. The tool evaluates responses across four key domains characteristic of autism spectrum conditions and provides insights based on a trained ML model.

## Features

- **Domain-specific assessment**: Evaluates patterns of behavior and thinking across four key domains:
  - Pattern Recognition & Detailed Perception
  - Sensory Processing
  - Social Communication
  - Repetitive Behaviors & Focused Interests

- **ML-powered analysis**: Uses a Random Forest classifier trained on autism diagnostic data to assess autism likelihood.

- **Visual results**: Provides visual representations of domain scores and insights.

- **Educational insights**: Offers domain-specific feedback and recommendations.

## System Requirements

- Python 3.8 or higher
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/autism-assessment-tool.git
cd autism-assessment-tool
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Train the ML model (if not already trained):
```bash
python autism_ml_model.py
```

4. Run the Streamlit application:
```bash
streamlit run autism_app.py
```

## Usage

1. Open your web browser and navigate to the URL displayed in your terminal (typically http://localhost:8501).
2. Enter your child's age and gender on the introduction page.
3. Complete the questionnaires for each domain, answering based on your observations of your child.
4. View the assessment results, including domain scores, overall autism alignment, and domain-specific insights.

## Project Structure

- `autism_ml_model.py`: Script for training and evaluating the machine learning model.
- `autism_app.py`: Streamlit application code.
- `models/`: Directory containing trained models and related artifacts.
- `train.csv`: Training data used for model development.
- `requirements.txt`: List of required Python packages.

## Disclaimer

This tool is for educational purposes only and is not intended to replace professional diagnosis. Always consult with healthcare professionals for proper evaluation and diagnosis of autism spectrum conditions.

## Model Information

The machine learning model used in this application is a Random Forest classifier trained on autism diagnostic data. It evaluates patterns across four key domains and provides a probability score indicating the likelihood of autism spectrum condition.

The model was trained using feature mappings from standard autism assessment tools, with domain-specific weightings based on clinical relevance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset used for training was derived from autism diagnostic data.
- This tool incorporates insights from established autism assessment methodologies.

---

For questions or feedback, please open an issue in this repository.