import customtkinter as ctk
from tkinter import filedialog, Toplevel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue

# Uygulama penceresini oluştur
app = ctk.CTk()

# Pencere başlığını ayarla
app.title("ML Algorithm Selector Application")

# Pencere boyutunu ayarla
app.geometry("500x400")

# Global tema değişkeni
current_theme = "Dark"

# Tema değiştirme fonksiyonu
def change_theme(choice):
    global current_theme
    current_theme = choice
    if choice == "Dark":
        ctk.set_appearance_mode("dark")
    elif choice == "Light":
        ctk.set_appearance_mode("light")

# Dosya seçme fonksiyonu
def select_file():
    global data
    file_path = filedialog.askopenfilename()
    if file_path:
        file_label.configure(text=file_path)
        data = pd.read_csv(file_path, sep=";")

# Yeni pencere açma ve grafik çizme fonksiyonu
def open_new_window(algorithm):
    if 'data' not in globals():
        error_window = Toplevel(app)
        error_window.title("Error")
        error_window.geometry("500x500")
        error_label = ctk.CTkLabel(error_window, text="Please select a file first.")
        error_label.pack(pady=20)
        return

    new_window = ctk.CTkToplevel(app)
    new_window.title("Selected Algorithm")
    new_window.geometry("800x600")
    ctk.set_appearance_mode(current_theme)  # Açılan pencerenin temasını ayarla

    label = ctk.CTkLabel(new_window, text=f"Selected Algorithm: {algorithm}")
    label.pack(pady=20)

    progress_bar = ctk.CTkProgressBar(new_window, mode="indeterminate")
    progress_bar.pack(pady=10)
    progress_bar.start()

    q = queue.Queue()

    def train_model(queue):
        # Veri hazırlığı
        df = data.drop(['name', 'session', 'Unnamed: 73', 'Unnamed: 74'], axis=1)

        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, 0].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Algoritma seçimi ve modeli eğitme
        if algorithm == "Logistic Regression":
            model = LogisticRegression()
        elif algorithm == "SVM":
            model = SVC(probability=True)
        elif algorithm == "K-NN":
            model = KNeighborsClassifier()
        elif algorithm == "Random Forest":
            model = RandomForestClassifier()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_test_classes = y_test

        # Confusion Matrix
        cm = confusion_matrix(y_test_classes, y_pred)

        queue.put((cm, y_test_classes, y_prob))

    def plot_results(cm, y_test_classes, y_prob):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Confusion Matrix çizimi
        cax = ax[0].matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax, ax=ax[0])
        ax[0].set_title('Confusion Matrix')
        ax[0].set_xlabel('Predicted')
        ax[0].set_ylabel('Actual')

        for (i, j), val in np.ndenumerate(cm):
            ax[0].text(j, i, f'{val}', ha='center', va='center')

        # ROC-AUC eğrisi çizimi
        fpr, tpr, _ = roc_curve(y_test_classes, y_prob)
        roc_auc = auc(fpr, tpr)
        ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax[1].set_xlim([0.0, 1.0])
        ax[1].set_ylim([0.0, 1.05])
        ax[1].set_title('Receiver Operating Characteristic (ROC)')
        ax[1].set_xlabel('False Positive Rate')
        ax[1].set_ylabel('True Positive Rate')
        ax[1].legend(loc="lower right")

        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def on_complete(result):
        progress_bar.stop()
        progress_bar.destroy()

        cm, y_test_classes, y_prob = result
        app.after(0, plot_results, cm, y_test_classes, y_prob)

    def check_queue():
        try:
            result = q.get_nowait()
            on_complete(result)
        except queue.Empty:
            app.after(100, check_queue)

    threading.Thread(target=train_model, args=(q,)).start()
    app.after(100, check_queue)

# Tema seçenekleri
theme_options = ["Dark", "Light"]

# Algoritma seçenekleri
ml_algorithms = ["Logistic Regression", "SVM", "K-NN", "Random Forest"]

# Combobox oluştur
theme_combobox = ctk.CTkComboBox(app, values=theme_options, command=change_theme)
theme_combobox.set("Select Theme")
theme_combobox.pack(pady=10)

# Dosya seçme düğmesi oluştur
select_button = ctk.CTkButton(app, text="Select File", command=select_file)
select_button.pack(pady=10)

# Seçilen dosya yolunu gösterecek etiket oluştur
file_label = ctk.CTkLabel(app, text="No file selected")
file_label.pack(pady=10)

# Algoritma seçme combobox'ı oluştur
algorithm_combobox = ctk.CTkComboBox(app, values=ml_algorithms)
algorithm_combobox.set("Select Algorithm")
algorithm_combobox.pack(pady=10)

# Algoritmayı çalıştırma düğmesi oluştur
run_button = ctk.CTkButton(app, text="Run Algorithm", command=lambda: open_new_window(algorithm_combobox.get()))
run_button.pack(pady=20)

# Ana döngüyü başlat
app.mainloop()
