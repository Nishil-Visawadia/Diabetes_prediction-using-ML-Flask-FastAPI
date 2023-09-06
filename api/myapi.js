const { FastAPI } = require('fastapi');
const { route: documentationRoute } = require('fastapi.openapi.openapi');
const { send, json } = require('@vercel/node');
const { StandardScaler } = require('sklearn-preprocessing');
const { train_test_split, SVC, accuracy_score } = require('sklearn');

const app = new FastAPI();

// Define your API routes here
app.post('/api/predict', async (req, res) => {
    try {
        const data = await json(req);

        const { value1, value2, value3, value4, value5, value6, value7, value8 } = data;

        const input_data = [
            value1, value2, value3, value4, value5, value6, value7, value8
        ];

        const input_data_as_numpy_array = Float32Array.from(input_data);
        const input_data_reshaped = input_data_as_numpy_array.reshape([1, -1]);
        const std_data = scaler.transform(input_data_reshaped);
        const prediction = classifier.predict(std_data);

        let result;
        if (prediction[0] === 0) {
            result = 'The person is not diabetic';
        } else {
            result = 'The person is diabetic';
        }

        return send(res, 200, { result });
    } catch (error) {
        return send(res, 500, { error: 'Internal Server Error' });
    }
});

// Serve FastAPI documentation at the root URL ("/")
app.get('/', documentationRoute(app));

// Serve FastAPI documentation JSON
app.get('/openapi.json', (req, res) => {
    res.json(app.openapi());
});

// Load and preprocess your dataset
const diabetes_dataset = require('../data/diabetes.csv');
const X = diabetes_dataset.data;
const Y = diabetes_dataset.target;

const scaler = new StandardScaler();
scaler.fit(X);
const standardized_data = scaler.transform(X);
const X_train_test = train_test_split(standardized_data, Y, { test_size: 0.2, stratify: Y, random_state: 2 });
const [X_train, X_test, Y_train, Y_test] = X_train_test;

// Train the SVM model
const classifier = new SVC({ kernel: 'linear' });
classifier.fit(X_train, Y_train);

module.exports = app;
