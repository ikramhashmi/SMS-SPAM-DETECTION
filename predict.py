import joblib as j
import pandas as pd


model = j.load("models\\spam_classifier.pkl")
tfidf = j.load("models\\tfidf_vectorizer.pkl")

def prediction(data, model=model, tfidf=tfidf):
    if (data, str):  
        df = pd.DataFrame({"text": [data]})  
    else:
        df = pd.DataFrame(data)  
        
    transformed_data = tfidf.transform(df["text"])  
  


    predict = model.predict(transformed_data)  
    prediction_label = "Spam" if predict > 0.5 else "Ham"
    print("Prediction output:", predict)
    return prediction_label