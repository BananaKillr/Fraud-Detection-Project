import tkinter as tk
import pickle
import pandas as pd
import random
from tkinter import messagebox
import csv
import subprocess



with open('metrics.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    metrics = next(reader)
metrics = [int(d) for d in metrics]

def add_new_row(row):
    generated_trans_num = '%032x' % random.randrange(16**32)
    read_df.loc[generated_trans_num] = row
    
def save():
    read_df.to_pickle('data.pkl')
    with open('metrics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(metrics)
    window.destroy()



def retrain():
    save()
    subprocess.run(["python", "retrain.py"], check=True)

# Function to predict the output
def predict_output():
    # Get the input values from the entry fields
    is_fraud_value = float(is_fraud_entry.get())
    category_value = int(category_entry.get())
    hour_of_day_value = int(hour_entry.get())
    day_of_week_value = int(day_entry.get())
    month_value = int(month_entry.get())
    z_score_value = float(z_score_entry.get())
    time_between_transactions_value = float(time_entry.get())
    amount_value = float(amount_entry.get())

    # Create a single data point
    single_data_point = [[category_value, hour_of_day_value, day_of_week_value, month_value, z_score_value, time_between_transactions_value, amount_value]]
    new_row_data = {
    'is_fraud': is_fraud_value, 
    'Hour of Day': hour_of_day_value,
    'Day of Week': day_of_week_value,
    'Month': month_value,
    'Category': category_value,
    'Z Score': z_score_value,
    'Normalized Log Time Difference': time_between_transactions_value,
    'Normalized Log Amount': amount_value
    }
    add_new_row(new_row_data)

    # Predict the output
    output = loaded_model.predict(single_data_point)[0]

    # Show the predicted output in a message box
    if output == is_fraud_value and output == 0: #true negative
        metrics[2] += 1
    elif output == is_fraud_value and output == 1: #true positive
        metrics[0] += 1
    elif output != is_fraud_value and output == 0: #false negative
        metrics[3] += 1
    else: #false positive
        metrics[1] += 1
        
    #[TP, FP, TN, FN]
    accuracy = (metrics[0] + metrics[2]) / (metrics[0] + metrics[1] + metrics[2] + metrics[3])
    
    if accuracy <= 0.9:
        retrain()

    messagebox.showinfo("Prediction", f"The predicted output is: {output}")
    if accuracy <= 0.9:
        retrain()
    
loaded_model = pickle.load(open('trained_model.sav','rb'))
# Create a tkinter window
window = tk.Tk()
window.protocol("WM_DELETE_WINDOW", save)
window.title("Fraud Detection GUI")
read_df = pd.read_pickle('data.pkl')

# Create labels and entry fields for input values
category_label = tk.Label(window, text="Category:")
category_label.pack()
category_entry = tk.Entry(window)
category_entry.pack()

hour_label = tk.Label(window, text="Hour of Day:")
hour_label.pack()
hour_entry = tk.Entry(window)
hour_entry.pack()

day_label = tk.Label(window, text="Day of Week:")
day_label.pack()
day_entry = tk.Entry(window)
day_entry.pack()

month_label = tk.Label(window, text="Month:")
month_label.pack()
month_entry = tk.Entry(window)
month_entry.pack()

z_score_label = tk.Label(window, text="Z Score:")
z_score_label.pack()
z_score_entry = tk.Entry(window)
z_score_entry.pack()

time_label = tk.Label(window, text="Time between Transactions:")
time_label.pack()
time_entry = tk.Entry(window)
time_entry.pack()

amount_label = tk.Label(window, text="Amount:")
amount_label.pack()
amount_entry = tk.Entry(window)
amount_entry.pack()

is_fraud_label = tk.Label(window, text="Is Fraud:")
is_fraud_label.pack()
is_fraud_entry = tk.Entry(window)
is_fraud_entry.pack()

# Create a button to predict the output
predict_button = tk.Button(window, text="Predict", command=predict_output)
predict_button.pack()
# Run the tkinter event loop
window.mainloop()