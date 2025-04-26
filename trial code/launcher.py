import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import subprocess

# === DATABASE SETUP ===
def init_db():
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    roll_no TEXT UNIQUE NOT NULL,
                    department TEXT NOT NULL
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL
            )''')

    conn.commit()
    conn.close()

def enroll_student(name, roll_no, dept):
    if not name or not roll_no or not dept:
        messagebox.showerror("Input Error", "All fields are required!")
        return
    try:
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        c.execute("INSERT INTO students (name, roll_no, department) VALUES (?, ?, ?)", (name, roll_no, dept))
        conn.commit()
        conn.close()
        messagebox.showinfo("Success", "Student enrolled successfully!")
    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Roll number must be unique!")

# === LAUNCH DETECTION SYSTEM ===
def launch_detection():
    try:
        subprocess.Popen(["python", "videotry2.py"])
    except Exception as e:
        messagebox.showerror("Launch Error", f"Could not launch detection system: {e}")

# === MAIN UI ===
def launch_ui():
    init_db()
    root = tk.Tk()
    root.title("Student Monitoring System")

    tab_control = ttk.Notebook(root)

    # --- ENROLL TAB ---
    enroll_tab = ttk.Frame(tab_control)
    tab_control.add(enroll_tab, text='Enroll Student')

    ttk.Label(enroll_tab, text="Name:").grid(column=0, row=0, padx=10, pady=5, sticky="w")
    name_entry = ttk.Entry(enroll_tab, width=30)
    name_entry.grid(column=1, row=0, padx=10)

    ttk.Label(enroll_tab, text="Roll No:").grid(column=0, row=1, padx=10, pady=5, sticky="w")
    roll_entry = ttk.Entry(enroll_tab, width=30)
    roll_entry.grid(column=1, row=1, padx=10)

    ttk.Label(enroll_tab, text="Department:").grid(column=0, row=2, padx=10, pady=5, sticky="w")
    dept_entry = ttk.Entry(enroll_tab, width=30)
    dept_entry.grid(column=1, row=2, padx=10)

    enroll_btn = ttk.Button(enroll_tab, text="Enroll", command=lambda: enroll_student(
        name_entry.get(), roll_entry.get(), dept_entry.get()))
    enroll_btn.grid(column=1, row=3, pady=10)

    # --- ALERTS TAB ---
    alerts_tab = ttk.Frame(tab_control)
    tab_control.add(alerts_tab, text='View Alerts')

    ttk.Label(alerts_tab, text="Search by Roll No:").pack(pady=(10, 0))
    roll_search_entry = ttk.Entry(alerts_tab, width=30)
    roll_search_entry.pack()

    alerts_list = tk.Text(alerts_tab, width=80, height=20)
    alerts_list.pack(padx=10, pady=10)

    def load_alerts():
        roll_no = roll_search_entry.get()
        if not roll_no:
            messagebox.showwarning("Missing Input", "Please enter a roll number.")
            return

        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        c.execute("SELECT timestamp, message, severity FROM alerts WHERE roll_no = ? ORDER BY id DESC", (roll_no,))
        logs = c.fetchall()
        conn.close()

        alerts_list.delete(1.0, tk.END)
        if not logs:
            alerts_list.insert(tk.END, "No alerts found for this student.")
        else:
            for timestamp, message, severity in logs:
                alerts_list.insert(tk.END, f"[{timestamp}] [{severity}] {message}\n")

    ttk.Button(alerts_tab, text="Load Alerts", command=load_alerts).pack(pady=5)

    # --- VIEW TAB ---
    view_tab = ttk.Frame(tab_control)
    tab_control.add(view_tab, text='View Enrolled')

    student_list = tk.Text(view_tab, width=60, height=20)
    student_list.pack(padx=10, pady=10)

    def load_students():
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        c.execute("SELECT name, roll_no, department FROM students")
        students = c.fetchall()
        student_list.delete(1.0, tk.END)
        for s in students:
            student_list.insert(tk.END, f"{s[0]} | {s[1]} | {s[2]}\n")
        conn.close()

    refresh_btn = ttk.Button(view_tab, text="Refresh List", command=load_students)
    refresh_btn.pack(pady=5)

    # --- DETECTION TAB ---
    detect_tab = ttk.Frame(tab_control)
    tab_control.add(detect_tab, text='Start Detection')

    ttk.Label(detect_tab, text="Start the cheating detection system", font=("Arial", 12)).pack(pady=20)
    ttk.Button(detect_tab, text="Launch Detection System", command=launch_detection).pack(pady=10)

    tab_control.pack(expand=1, fill='both')
    root.mainloop()



if __name__ == "__main__":
    launch_ui()
