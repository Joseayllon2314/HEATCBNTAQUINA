import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import sys

# === Soporte para PyInstaller ===
if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

model_path = os.path.join(base_path, "MODELOHEAT_V2.keras")
scaler_path = os.path.join(base_path, "ESCALERHEAT_V2.pkl")
logo_path = os.path.join(base_path, "taquina.png")

# === Cargar modelo y scaler ===
model = load_model(model_path)
scaler = joblib.load(scaler_path)

# === Día a número ===
dias_semana = {
    "Lunes": 1, "Martes": 2, "Miércoles": 3, "Jueves": 4,
    "Viernes": 5, "Sábado": 6, "Domingo": 7
}

# === Interfaz principal ===
root = tk.Tk()
root.title("ABINBEV - HEAT PREDICTION - TAQUINA")
root.geometry("520x600")
root.configure(bg="black")
root.resizable(False, False)

# === Estilo ttk ===
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", foreground="white", background="black", font=("Segoe UI", 10))
style.configure("TButton", foreground="black", background="yellow", font=("Segoe UI", 10, "bold"))
style.map("TButton", background=[('active', '#FFD700')])
style.configure("TCombobox", fieldbackground="white", background="white", foreground="black")

# === Mostrar logo si existe ===
if os.path.exists(logo_path):
    img = Image.open(logo_path)
    img = img.resize((150, 150))
    logo_img = ImageTk.PhotoImage(img)
    logo_label = tk.Label(root, image=logo_img, bg="black")
    logo_label.pack(pady=5)

# === Widgets de entrada ===
ttk.Label(root, text="Día de la semana:").pack(pady=5)
combo_dia = ttk.Combobox(root, values=list(dias_semana.keys()), state="readonly", width=30)
combo_dia.set("Seleccionar día")
combo_dia.pack(pady=5)

ttk.Label(root, text="Volumen de latas:").pack(pady=5)
entry_latas = ttk.Entry(root, width=30)
entry_latas.pack(pady=5)

ttk.Label(root, text="Volumen de botellas:").pack(pady=5)
entry_botellas = ttk.Entry(root, width=30)
entry_botellas.pack(pady=5)

# === Área de resultados ===
resultado_text = tk.Text(root, height=10, width=58, font=("Consolas", 10), bg="#111111", fg="#FFD700")
resultado_text.pack(pady=15)
resultado_text.config(state="disabled", relief="sunken", borderwidth=2)

# === Función principal ===
def predecir_kpi():
    try:
        dia_str = combo_dia.get()
        volumen_latas = float(entry_latas.get())
        volumen_botellas = float(entry_botellas.get())

        if dia_str not in dias_semana:
            messagebox.showerror("Error", "Selecciona un día válido.")
            return

        if ((volumen_latas < 100 and volumen_botellas == 0) or
            (volumen_botellas < 100 and volumen_latas == 0)):
            messagebox.showerror("Error", "Uno de los volúmenes es < 100 y el otro es 0.")
            return

        dia = dias_semana[dia_str]
        volumen_total = volumen_latas + volumen_botellas
        proporcion_latas = volumen_latas / volumen_total
        proporcion_botellas = volumen_botellas / volumen_total
        log_volumen_total = np.log1p(volumen_total)
        ratio_bot_lat = volumen_botellas / volumen_latas if volumen_latas > 0 else 0

        X_input = np.array([[dia, volumen_botellas, volumen_latas,
                             proporcion_latas, proporcion_botellas,
                             log_volumen_total, ratio_bot_lat]])
        X_scaled = scaler.transform(X_input)
        kpi_predicho = model.predict(X_scaled)[0][0]

        if kpi_predicho < 0:
            kpi_predicho = 0

        consumo_estimado = ((kpi_predicho * volumen_total) - 1033) / 34.26
        factor_correccion = 0.9671
        presion_distribucion = 1

        resultado_text.config(state="normal")
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, f"\n📊 RESULTADOS\n")
        resultado_text.insert(tk.END, f"Volumen total            : {volumen_total:.2f}\n")
        resultado_text.insert(tk.END, f"KPI PREDICHO             : {kpi_predicho:.4f}\n")
        resultado_text.insert(tk.END, f"CONSUMO CALCULADO        : {consumo_estimado:.2f}\n")
        resultado_text.insert(tk.END, f"FACTOR DE CORRECCIÓN     : {factor_correccion}\n")
        resultado_text.insert(tk.END, f"PRESIÓN DE DISTRIBUCIÓN  : {presion_distribucion}\n")
        resultado_text.config(state="disabled")

    except ValueError:
        messagebox.showerror("Error", "Ingresa valores numéricos válidos.")

# === Botón de predicción ===
ttk.Button(root, text="Predecir KPI", command=predecir_kpi).pack(pady=10)

# === Ejecutar aplicación ===
root.mainloop()
