import joblib

model = joblib.load("model.joblib")


def make_prediction(input_data):
    model_input_data = [
        input_data["sepal_length"],
        input_data["sepal_width"],
        input_data["petal_length"],
        input_data["petal_width"],
    ]
    output_data = model.predict([model_input_data])
    output_data = output_data[0]
    output_data = int(output_data)
    return output_data
