from google import genai

client = genai.Client(api_key="AIzaSyCTq3A-6C3f-r-R-zEeKVbeMsP_1ytK6GU")

prompt = """
Coin: Bitcoin
Latest price: 68420
Forecasted price: 69110
Predicted change: +1.01%
Trend: Mild bullish momentum

Write a short report in about 100 words.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print(response.text)