
import pandas as pd
import numpy as np
import tkinter as tk

root = tk.Tk()
#def button_storage():

def click_storage(id):
        label = tk.Label(root, text="Armazem" )


w = tk.Label(root, text="Hello Tkinter!")
k = tk.Label(root, text="My name is Rodas Rolamentos")

but = tk.Button(root, text="Armazem", command=click_storage)

k.grid(row=0, column=0)
w.grid(row=1, column=0)
but.grid(row=2, column=0)


root.mainloop()







clients_df = pd.read_csv("clients.txt",sep=';',header=0)

warehouses_df = pd.read_csv("warehouses.txt",sep=';',header=0)
clients_df

print(warehouses_df)
