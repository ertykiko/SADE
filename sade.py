# %%
import pandas as pd
import numpy as np
import tkinter as tk

root = tk.Tk()

w = tk.Label(root, text="Hello Tkinter!")
w.pack()

root.mainloop()





# %%

clients_df = pd.read_csv("clients.txt",sep=';',header=0)

warehouses_df = pd.read_csv("warehouses.txt",sep=';',header=0)
# %%
clients_df


# %%
warehouses_df




