from tkinter import *
import tkinter as tk
import os
import keyboard
from PIL import ImageTk, Image

root_path = os.path.dirname(__file__)
path = os.path.join(root_path, 'ui_elements')

root = Tk()
# root.geometry('300x100')
mainframe = Frame(root)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.pack(expand=True, fill='both', anchor='center')

def push(i):
    print(path+os.listdir(path)[i])
    if i==0:
        border_0.config(bg='red' if border_state[0] else 'green')
        border_state[0] = not border_state[0]
    elif i==1:
        border_1.config(bg='red' if border_state[1] else 'green')
        border_state[1] = not border_state[1]

def selection(event):
    print(event)

images = [None] * 2
border_state = [True] * 2

for i, image_path in enumerate([item for item in os.listdir(path) if item.endswith('.png')]):
    images[i] = PhotoImage(file=os.path.join(path, image_path))
border_0 = LabelFrame(mainframe, bd = 6, bg = "black")
border_0.grid(column=0, row=0, sticky=(N, W, E, S))
button_0 = tk.Button(border_0, command=lambda:push(0), image=images[0])
button_0.grid(row=1, column=0, sticky=(E,W), columnspan=2)
food_label_0 = Label(border_0, text = "Philip Lee", font =("Courier", 14))
food_label_0.grid(row=2, column=0, sticky=(E,W))
desc_label_1 = Label(border_0, text = "Food for Bunny", font =("Ariel", 12))
desc_label_1.grid(row=3, column=0, sticky=(E,W), columnspan=2)
price_label_2 = Label(border_0, text = "$4.50", font =("Courier", 14))
price_label_2.grid(row=2, column=1, sticky=(E,W))

border_1 = LabelFrame(mainframe, bd = 6, bg = "black")
border_1.grid(column=1, row=0, sticky=(N, W, E, S))
button_1 = tk.Button(border_1, command=lambda:push(1), image=images[1])
button_1.grid(row=1, column=0, sticky=(E,W))
food_label_1 = Label(border_1, text = "Jane Lee", font =("Courier", 14))
food_label_1.grid(row=2, column=0, sticky=(E,W))
desc_label_1 = Label(border_1, text = "The Bunny", font =("Ariel", 12))
desc_label_1.grid(row=3, column=0, sticky=(E,W), columnspan=2)
price_label_1 = Label(border_1, text = "$4.51", font =("Courier", 14))
price_label_1.grid(row=2, column=1, sticky=(E,W))

root.bind(0, lambda event: selection(event))
root.bind(1, lambda event: selection(event))

root.mainloop()