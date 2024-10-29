from flask import Flask, request, jsonify
import joblib
import fetch_stock_name_ticker as stocktick
import all_getprice
import extract

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained intent model
loaded_model = joblib.load('intent_model.pkl')

# Define the intent mappings
intent_map = {
    "General_Info": lambda user_input: {"response": general_question()},
    "Get_Historical": lambda user_input: {"response": get_historical(user_input)},
    "Get_Price": lambda user_input: {"response": get_price(user_input)}
}


# Functions for intents
def general_question():
    return "That's a great question! Here's some information about the stock market..."


def get_price(user_input):
    ticker = stocktick.fetch_stock_name(user_input)
    if not ticker:
        return "Stock ticker not found."
    curr_price = all_getprice.get_current_price(ticker)
    return f"Price of {ticker} is: {curr_price}"


def get_historical(user_input):
    ticker = stocktick.fetch_stock_name(user_input)
    if not ticker:
        return "Stock ticker not found."
    date_from_query = extract.extract_dates(user_input)

    if date_from_query is None:
        return f"We know you are trying to get historical price of {ticker}. Can you be more specific on dates?"
    else:
        date_from_query = date_from_query.strftime('%Y-%m-%d')
        hist_price = all_getprice.get_historical_price(ticker, date_from_query)
        if hist_price is None:
            return "Seems like the stock market was closed that day."
        return f"The price of {ticker} on {date_from_query} was {hist_price}"


def unknown_intent():
    return "We know that it is a great question but currently this is out of our scope. But, we will be back soon. AutoProphet rocks"


# Flask route to process user input
@app.route('/chatbot', methods=['GET'])
def chatbot():
    user_input = request.args.get("query")
    if not user_input:
        return jsonify({"error": "No query provided."}), 400

    # Predict intent
    intent = loaded_model.predict([user_input])[0]

    # Find the response based on intent
    response_func = intent_map.get(intent, lambda user_input: {"response": unknown_intent()})
    response = response_func(user_input)

    return jsonify(response)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
