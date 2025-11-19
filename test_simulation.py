import tkinter as tk

root = tk.Tk()
root.title("TESTE TKINTER")
root.geometry("300x200")

label = tk.Label(root, text="Se consegues ver isto, Tkinter está OK.", font=("Arial", 14))
label.pack(pady=20)

root.mainloop()
