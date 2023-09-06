# with Documentation
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    value1: float
    value2: float
    value3: float
    value4: float
    value5: float
    value6: float
    value7: float
    value8: float

@app.post("/predict")
async def predict(data: InputData):
    # data = await request.json()
    value1 = data.value1
    value2 = data.value2
    value3 = data.value3
    value4 = data.value4
    value5 = data.value5
    value6 = data.value6
    value7 = data.value7
    value8 = data.value8

    # loading the diabetes dataset to a pandas DataFrame
    diabetes_dataset = pd.read_csv("diabetes.csv")

    diabetes_dataset["Outcome"].value_counts()
    # separating the data and labels
    X = diabetes_dataset.drop(columns="Outcome", axis=1)
    Y = diabetes_dataset["Outcome"]

    # data standardization
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    Y = diabetes_dataset["Outcome"]

    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # training the model
    classifier = svm.SVC(kernel="linear")
    # training the support vector machine classifier
    classifier.fit(X_train, Y_train)

    # model evaluation
    # accuracy score on the training data
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    # accuracy score on the test data
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    # making a predictive system
    input_data = (value1, value2, value3, value4, value5, value6, value7, value8)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    prediction = classifier.predict(std_data)

    if prediction[0] == 0:
        result = "The person is not diabetic"
    else:
        result = "The person is diabetic"

    return JSONResponse({"result": result})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)