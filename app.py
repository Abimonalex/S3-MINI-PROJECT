import pandas as pd
from flask import Flask, render_template, request
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from rake_nltk import Rake
import nltk


app = Flask(__name__)

# Load the preprocessed dataset
data = pd.read_csv("new_dataset_p1.csv")

# Split the dataset into features (X) and labels (y)
X = data['extracted_keywords'].apply(lambda x: ' '.join(eval(x)))  # Convert keywords list to space-separated strings
y = data['label']

# Create a TF-IDF vectorizer to convert text data to numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Initialize the SVM model
svm_model = SVC(kernel='linear', C=2.0)
svm_model.fit(X_tfidf, y)

# Save both the trained model and vectorizer to a single PKL file
joblib.dump((svm_model, tfidf_vectorizer), 'model_and_vectorizer.pkl')

# Load both the trained model and vectorizer from the single PKL file
loaded_model, loaded_tfidf_vectorizer = joblib.load('model_and_vectorizer.pkl')

def extract_keywords(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()
    return ' '.join(keywords)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        
        # Extract keywords from the user input
        user_keywords = extract_keywords(user_input)

        # Convert the user keywords to a space-separated string
        user_input_tfidf = loaded_tfidf_vectorizer.transform([user_keywords])

        # Ensure the user_input_tfidf has the same number of features as the training data
        if user_input_tfidf.shape[1] < 5000:
            user_input_tfidf = hstack([user_input_tfidf, csr_matrix((user_input_tfidf.shape[0], 5000 - user_input_tfidf.shape[1]))])

        # Make a prediction using the loaded model
        user_pred = loaded_model.predict(user_input_tfidf)
        
        prediction1=user_pred[0]+1

        return render_template('index.html', prediction=prediction1)

if __name__ == '__main__':
    app.run(debug=True)
