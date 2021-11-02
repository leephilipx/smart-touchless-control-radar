from tkinter import *
import tkinter as tk
import os

class Menu:

    def __init__(self, root):
        self.mainframe = Frame(root)
        root.title("Restaurant Menu")
        root.geometry("800x600")
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.mainframe.pack(expand=True, fill='both', anchor='center')
        self.root_path = os.path.dirname(__file__)
        self.path = os.path.join(self.root_path, 'ui_elements')
        self.create_left_widgets()
        if True:
            self.create_right_widgets()
        self.binding_events(root)
        self.selection_counter = 0
        self.borders[0].config(bg='blue')

    def create_left_widgets(self):

        self.menu_max = 4
        self.images = [None] * self.menu_max
        self.borders = [None] * self.menu_max
        self.buttons = [None] * self.menu_max
        self.food_labels = [None] * self.menu_max
        self.desc_labels = [None] * self.menu_max
        self.price_labels = [None] * self.menu_max

        for i, image_path in enumerate([item for item in os.listdir(self.path) if item.endswith('.png')]):
            self.images[i] = PhotoImage(file=os.path.join(self.path, image_path))

        ## -- Menu Item --
        for i in range(self.menu_max):
            self.borders[i] = LabelFrame(self.mainframe, bd=6, bg="white")
            self.borders[i].grid(column=i%2, row=i//2, sticky=(N,W,E,S))
            self.buttons[i] = tk.Button(self.borders[i], image=self.images[i])
            self.buttons[i].grid(row=0, column=0, sticky=(E,W), columnspan=2)
            self.food_labels[i] = Label(self.borders[i], text=f"Food {i}", font=("Courier", 12))
            self.food_labels[i].grid(row=1, column=0, sticky=(E,W))
            self.price_labels[i] = Label(self.borders[i], text=f"${i}", font=("Courier", 12))
            self.price_labels[i].grid(row=1, column=1, sticky=(E,W))
            self.desc_labels[i] = Label(self.borders[i], text =f"Desc {i}", font=("Arial", 10))
            self.desc_labels[i].grid(row=2, column=0, sticky=(E,W), columnspan=2)

    def create_right_widgets(self):
        gesture = 'push'
        self.right_column = LabelFrame(self.mainframe, bd=6, bg="white")
        self.right_column.grid(column=2, row=0, rowspan=2, columnspan=2, sticky=(N,W,E,S))
        self.right_column.grid_rowconfigure(3, weight=1)
        self.gesture = Label(self.right_column, text=f"Gesture:{gesture}", font=("Courier", 12))
        self.gesture.grid(row=0, column=0, sticky=(E,W))
        self.info_label = Label(self.right_column, text=f"Confirm? Yes/No", font=("Helvatical", 12), bg='#f05959')
        self.info_label.grid(row=1, column=0, sticky=(E,W), columnspan=2, pady=20)
        self.cart = Label(self.right_column, text="Cart", font=("Arial", 15))
        self.cart.grid(row=2, column=0, sticky=(W,E), columnspan=2, pady=20)
        self.item = Label(self.right_column, text="Item 1\nhshshks\ndjhjhxjhdj\ndjhejhjkhekjd\ndeededg", font=("Courier", 12),  anchor='w')
        self.item.grid(row=3, column=0, sticky=(W,E), columnspan=2)
        self.price = Label(self.right_column, text="Total Price:", font=("Courier", 12))
        self.price.grid(row=4, column=0, sticky=(W,E), columnspan=2, pady=20)
        self.cancel = Button(self.right_column, text="Cancel", font=("Courier", 12))
        self.cancel.grid(row=5, column=0, sticky=(E,W))
        self.submit_order = Button(self.right_column, text="Submit Order", font=("Courier", 12))
        self.submit_order.grid(row=5, column=1, sticky=(E,W))

    def binding_events(self, root):
        root.bind(0, lambda event: self.keypress_handler(event))
        root.bind(1, lambda event: self.keypress_handler(event))
        root.bind(2, lambda event: self.keypress_handler(event))
        root.bind(3, lambda event: self.keypress_handler(event))
        root.bind(4, lambda event: self.keypress_handler(event))

    def keypress_handler(self, event):
        if event.char == '2': self.order()
        elif event.char == '3': self.scroll_selection()

    def scroll_selection(self):
        self.selection_counter = (self.selection_counter + 1) % self.menu_max
        for i in range(self.menu_max):
            self.borders[i].config(bg='blue' if i==self.selection_counter else 'white')

    def order(self):
        print(f"Ordering {self.selection_counter}")

root = Tk()
Menu(root)
root.mainloop()