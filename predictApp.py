import tkinter as tk
from tkinter import ttk
import numpy as np
import sys
import pickle

sys.path.append('src/decision-tree-CART')  # Thêm đường dẫn vào hệ thống
sys.path.append('src/ridge-regression')
import cart_model  # Đảm bảo rằng cart_model đã được định nghĩa đúng
import ridge_model

# Load pre-trained models (adjust paths as necessary)
with open('models/ridge_model.pkl', 'rb') as f:
    ridge_weights = pickle.load(f)  # Tải trọng số của mô hình Ridge

with open('models/cart_model_best_depth.pkl', 'rb') as f:
    cart_model_best_depth = pickle.load(f)

# Hàm dự đoán dựa trên đầu vào
def predict():
    try:
        # Collect input values
        cw1 = float(entry_cw1.get())
        mid_term = float(entry_mid_term.get())
        cw2 = float(entry_cw2.get())
        model_choice = model_var.get()
        
        # Prepare data for prediction
        input_data = np.array([[cw1, mid_term, cw2]])
        
        # Dự đoán dựa trên mô hình được chọn
        if model_choice == 'Ridge':
            predicted_final = ridge_model.predict(input_data, ridge_weights)[0]
        else:
            predicted_final = cart_model_best_depth.predict(input_data)
        
        # Lấy giá trị đầu tiên trong predicted_final
        predicted_final_value = predicted_final[0] if isinstance(predicted_final, np.ndarray) else predicted_final

        # Display prediction
        label_result.config(text=f"Predicted Final Score: {predicted_final_value:.2f}")
    except ValueError:
        label_result.config(text="Please enter valid numbers for all fields.")
    except Exception as e:
        label_result.config(text=f"Error: {e}")

# Initialize main window
root = tk.Tk()
root.title("Grade Prediction Interface")

# Set background color
root.configure(bg="#B9D6F3")

# Configure grid weights for resizing
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)
root.grid_rowconfigure(5, weight=1)

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Input fields
tk.Label(root, text="CW1 Score:", bg="#B9D6F3", font=("Times New Roman", 13)).grid(row=0, column=0, sticky='ew')
entry_cw1 = tk.Entry(root, font=("Times New Roman", 13))
entry_cw1.grid(row=0, column=1, sticky='ew')

tk.Label(root, text="Mid-Term Score:", bg="#B9D6F3", font=("Times New Roman", 13)).grid(row=1, column=0, sticky='ew')
entry_mid_term = tk.Entry(root, font=("Times New Roman", 13))
entry_mid_term.grid(row=1, column=1, sticky='ew')

tk.Label(root, text="CW2 Score:", bg="#B9D6F3", font=("Times New Roman", 13)).grid(row=2, column=0, sticky='ew')
entry_cw2 = tk.Entry(root, font=("Times New Roman", 13))
entry_cw2.grid(row=2, column=1, sticky='ew')

# Model selection dropdown
tk.Label(root, text="Select Model:", bg="#B9D6F3", font=("Times New Roman", 13)).grid(row=3, column=0, sticky='ew')
model_var = tk.StringVar(value="Ridge")
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=["Ridge", "CART"], font=("Times New Roman", 13))
model_dropdown.grid(row=3, column=1, sticky='ew')

# Prediction button
predict_button = tk.Button(root, text="Predict Final Score", command=predict, font=("Times New Roman", 13))
predict_button.grid(row=4, columnspan=2, sticky='ew')

# Result display
label_result = tk.Label(root, text="Predicted Final Score: ", bg="#B9D6F3", font=("Times New Roman", 13))
label_result.grid(row=5, columnspan=2, sticky='ew')

# Run the application
root.mainloop()