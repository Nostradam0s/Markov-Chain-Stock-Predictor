import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Download data
ticker = "AAPL"
start_date = "2023-01-01"
end_date = "2024-01-01"

df = yf.download(ticker, start=start_date, end=end_date)

#Fallback in case download fails
if df.empty or 'Close' not in df.columns:
    print(f"Failed to fetch data for {ticker}. Using fallback data.")
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Close': [150, 152, 151, 153, 155, 154, 156, 158, 157, 159]
    }).set_index('Date')

#Ensure proper formatting
df['Close'] = df['Close'].astype(float)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

#Calculate daily return state
df['Return'] = df['Close'].pct_change()
df.dropna(inplace=True)

def get_state(x):
    if x > 0.001:
        return 'Up'
    elif x < -0.001:
        return 'Down'
    else:
        return 'Flat'

df['State'] = df['Return'].apply(get_state)

#Build transition matrix
transitions = pd.crosstab(df['State'][:-1], df['State'][1:], normalize='index')
print("\n Transition Matrix:")
print(transitions)

#Predict next state using Markov
if df['State'].empty:
    print("No state data to predict from.")
else:
    today_state = df['State'].iloc[-1]
    print(f"\n Today's state: {today_state}")

    if today_state in transitions.index:
        next_state = transitions.loc[today_state].idxmax()
        print(f" Predicted next state: {next_state}")
    else:
        print("Today's state not in transition matrix. Cannot predict.")

def predict_next_state(current_state, matrix):
    probs = matrix.loc[current_state]
    return random.choices(probs.index, weights=probs.values)[0]

today_state = df['State'].iloc[-1]
predicted = predict_next_state(today_state, transitions)

print(f"Today's State: {today_state}")
print(f"Predicted Next State: {predicted}")

#Simulate N Days
def simulate_n_days(start_state, matrix, n=10):
    states_sequence = [start_state]
    current = start_state
    for _ in range(n):
        current = predict_next_state(current, matrix)
        states_sequence.append(current)
    return states_sequence

simulated_path = simulate_n_days(today_state, transitions, n=15)
print("Simulated Future States:", simulated_path)

#Visualize Matrix
sns.heatmap(transitions, annot=True, cmap='coolwarm')
plt.title('Markov Chain Transition Matrix')
plt.show()
