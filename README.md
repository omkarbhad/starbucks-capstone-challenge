# ☕ Starbucks Offer Success Predictor

![Demo](demo.gif)

## 🚀 Project Overview
This project leverages advanced machine learning techniques to analyze customer interactions with promotional offers on the Starbucks rewards mobile app. The objective is to accurately predict which customers are likely to engage with specific offers (such as discounts, buy-one-get-one deals, or informational promotions). By identifying key patterns and customer segments that demonstrate higher responsiveness, Starbucks can optimize its marketing strategies, improve customer satisfaction, and increase overall marketing efficiency.

## 🌟 Key Achievements
- Achieved over **91% accuracy** in predicting customer response to Starbucks promotional offers.
- Developed a robust **CatBoost model** outperforming multiple advanced classifiers.
- Successfully implemented comprehensive **feature engineering**, enhancing predictive performance.
- Created an intuitive and interactive **Streamlit web application** for seamless model deployment and user interaction.
## 🛠️ Technologies Used
- **Python**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, LightGBM, CatBoost, RandomForest
- **Visualization**: Plotly
- **Web Framework**: Streamlit

## 📈 Model Performance Metrics

| Model           | Accuracy | F1-Score | Precision | Recall |
|-----------------|----------|----------|-----------|--------|
| **CatBoost**    | **91.32%** | **91.00%** | 89%       | 93%    |
| LightGBM        | 91.27%   | 90.98%   | 89%       | 93%    |
| Tuned LightGBM  | 91.23%   | 90.93%   | 89%       | 93%    |
| Random Forest   | 91.03%   | 90.68%   | 89%       | 92%    |
| AdaBoost        | 88.49%   | 88.00%   | 87%       | 89%    |
| Decision Tree   | 87.07%   | 86.30%   | 87%       | 86%    |

### 🎯 Best Model: **CatBoost** (Accuracy: 91.32%)

## 🔑 Top Features
- **Total Amount Spent**
- **Customer Income**
- **Customer Age**
- **Reward Amount**
- **Offer Duration**

## 📂 Dataset Overview
Provided by Udacity’s Data Scientist Nanodegree program, the dataset simulates interactions with Starbucks' mobile app:

- **Portfolio.json**: Details of promotional offers
- **Profile.json**: Customer demographics
- **Transcript.json**: User interaction logs

## 🧹 Data Preparation Steps
- Cleaned missing and inconsistent data
- Encoded categorical variables (e.g., gender, channels, offer types)
- Extracted timestamps and created temporal features
- Engineered new features for enhanced predictions (membership duration, spending behavior)

## ⚡ Quick Start

### 1. Clone and Set Up
```bash
git clone https://github.com/your-username/starbucks-capstone-challenge.git
cd starbucks-capstone-challenge
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## ⚙️ Model Training and Preprocessing Steps

### Data Preprocessing
- Loaded and merged data from multiple sources (portfolio, profile, transcript)
- Cleaned datasets by handling missing values, removing outliers, and ensuring consistency
- Encoded categorical features (e.g., gender, offer type, channel)
- Engineered features such as membership duration, monthly spend, and interaction timings

### Model Training

#### Running Preprocessing and Training Scripts

To execute data preprocessing and model training, run the following commands in your terminal:

```bash
# Run data preprocessing
python preprocessing.py

# Train machine learning models
python train_model.py
```

These scripts generate processed data in `data/processed/` and save trained models in the `models/` directory for later use.

- Split data into training and testing sets for validation
- Trained and compared multiple classifiers: CatBoost, LightGBM, RandomForest, AdaBoost, Decision Tree
- Evaluated models using accuracy, precision, recall, F1-score, and confusion matrices
- Optimized the best-performing model (LightGBM) using hyperparameter tuning (RandomizedSearchCV)
- Saved the final models for deployment

## 📋 Project Structure
- `app.py`: Interactive prediction app
- `preprocessing.py`: Data cleaning and feature engineering
- `train_model.py`: Model training and evaluation
- `models/`: Saved trained model files
- `data/processed/`: Preprocessed datasets
- `Starbucks_Capstone_notebook.ipynb`: Exploratory analysis and model training notebook

## 📝 Blog Post
Detailed project walkthrough and insights on [Medium](https://medium.com/@omkarbhad/starbucks-challenge-f131242398c9).

## 🤝 Contribution
Contributions are welcome:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Submit a Pull Request

## ⚖️ License
MIT License – see [LICENSE](LICENSE) for details.

## 🙌 Acknowledgments
- [Starbucks](https://www.starbucks.com/) & [Udacity](https://www.udacity.com/)
- [Streamlit](https://streamlit.io/), [Scikit-learn](https://scikit-learn.org/), [Plotly](https://plotly.com/), [Pandas](https://pandas.pydata.org/)
- Open-source community and contributors
