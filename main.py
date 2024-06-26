from flask import (
    Flask,
    request,
    jsonify
)

from model import make_prediction

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    output_data = make_prediction(input_data)

    return jsonify({
        "input_data": input_data,
        "output_data": output_data,
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

