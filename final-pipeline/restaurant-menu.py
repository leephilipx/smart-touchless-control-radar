from tkinter import *
from tkinter import ttk
import os
from PIL import ImageTk, Image

root = Tk()
root_path = os.path.dirname(__file__)
path = os.path.join(root_path, 'ui_elements')

mainframe = ttk.Frame(root, padding="12 12 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.pack(expand=True, fill='both', anchor='center')

def push(i):
    print(path+os.listdir(path)[i])

def selection():
    pass

images = [None] * 2

for i, image_path in enumerate([item for item in os.listdir(path) if item.endswith('.png')]):
    images[i] = PhotoImage(file=os.path.join(path, image_path))

s = ttk.Style()
s.configure('TButton', indicatoron=[('pressed', '#13ed30'), ('selected', '#4a13ed')])
button_0 = ttk.Button(mainframe, command=lambda:push(0), image=images[0])
button_0.grid(row=1, column=0, sticky=(E,W))
button_1 = ttk.Button(mainframe, command=lambda:push(1), image=images[1])
button_1.grid(row=1, column=1, sticky=(E,W))
root.bind(0, lambda event: push(0))
root.bind(1, lambda event: push(1))
root.mainloop()