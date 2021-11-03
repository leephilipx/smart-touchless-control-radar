from tkinter import *
import tkinter as tk
import os, csv
import numpy as np
from time import sleep
from threading import Thread

class Menu:

    def __init__(self, root):

        self.mainframe = Frame(root)
        root.title('Restaurant Menu')
        # root.geometry('800x800')
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        # self.mainframe.pack(expand=True, fill='both', anchor='center')

        self.root_path = os.path.dirname(__file__)
        self.path = os.path.join(self.root_path, 'ui_elements')
        self.gestures_list = ['background', 'holdit', 'push', 'swipedown', 'waving']

        self.menu_parser()
        self.create_left_widgets()
        self.create_right_widgets()
        self.binding_events(root)
        self.selection_counter_l = 0
        self.selection_counter_r = 0

        self.selection_mode = 1
        self.toggle_menu()


    def menu_parser(self):

        self.menu_names = []
        self.menu_prices = []
        self.menu_descs = []
        self.menu_imgs = []

        with open(os.path.join(self.path, 'menu_items.csv'), 'r') as csvfile:
            menu_list = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i, entry in enumerate(menu_list):
                if i != 0:
                    self.menu_names.append(entry[0])
                    self.menu_prices.append(float(entry[1]))
                    self.menu_descs.append(entry[2])
                    self.menu_imgs.append(tk.PhotoImage(file=os.path.join(self.path, entry[3])))

        self.menu_prices = np.array(self.menu_prices)
        self.menu_qty = np.zeros(self.menu_prices.shape, dtype=int)


    def create_left_widgets(self):

        self.menu_max = 4
        self.borders_l = [None] * self.menu_max
        self.buttons = [None] * self.menu_max
        self.name_labels = [None] * self.menu_max
        self.desc_labels = [None] * self.menu_max
        self.price_labels = [None] * self.menu_max

        ## -- Menu Item --
        for i in range(self.menu_max):
            self.borders_l[i] = tk.LabelFrame(self.mainframe, bd=6, bg='white')
            self.borders_l[i].grid(column=i%2, row=i//2, sticky=(N,W,E,S))
            self.buttons[i] = tk.Button(self.borders_l[i], image=self.menu_imgs[i])
            self.buttons[i].grid(row=0, column=0, sticky=(E,W), columnspan=2)
            self.name_labels[i] = tk.Label(self.borders_l[i], text=self.menu_names[i], font=('Courier', 10))
            self.name_labels[i].grid(row=1, column=0, sticky=(E,W))
            self.price_labels[i] = tk.Label(self.borders_l[i], text='${:.2f}'.format(self.menu_prices[i]), font=('Courier', 10))
            self.price_labels[i].grid(row=1, column=1, sticky=(E,W))
            self.desc_labels[i] = tk.Label(self.borders_l[i], text =self.menu_descs[i], font=('Arial', 8))
            self.desc_labels[i].grid(row=2, column=0, sticky=(E,W), columnspan=2)


    def create_right_widgets(self):
        self.borders_r = [None] * 3
        self.right_column = tk.LabelFrame(self.mainframe, bd=6, bg='white')
        self.right_column.grid(column=2, row=0, rowspan=2, columnspan=2, sticky=(N,W,E,S))
        self.right_column.grid_rowconfigure(3, weight=1)
        self.gesture_text = tk.StringVar()
        self.gesture_text.set(f"Gesture: {'n/a'.center(10)}")
        self.gesture_label = tk.Label(self.right_column, textvariable=self.gesture_text, bg='black', fg='white', font=('Courier', 10))
        self.gesture_label.grid(row=0, column=0, sticky=(E,W), columnspan=2)
        self.info_text = tk.StringVar()
        self.info_label = tk.Label(self.right_column, textvariable=self.info_text, font=('Helvetica', 10), fg='red', bg='white')
        self.info_label.grid(row=1, column=0, sticky=(E,W), columnspan=2, pady=10)
        self.cart = tk.Label(self.right_column, text='         Cart         ', bg='white', font=('Courier', 10))
        self.cart.grid(row=2, column=0, sticky=(E,W), columnspan=2)
        self.cart_text = tk.StringVar()
        self.item = tk.Label(self.right_column, textvariable=self.cart_text, bg='white', font=('Courier', 8))
        self.item.grid(row=3, column=0, sticky=(N,E,W), columnspan=2)
        self.total_price_text = tk.StringVar()
        self.total_price_text.set('Total Bill: $0.00')
        self.total_price_label = tk.Label(self.right_column, textvariable=self.total_price_text, bg='white', font=('Courier', 9))
        self.total_price_label.grid(row=4, column=0, sticky=(E,W), columnspan=2, pady=10)
        self.borders_r[0] = tk.Button(self.right_column, text='Add Item', font=('Courier', 8))
        self.borders_r[0].grid(row=5, column=0, sticky=(E,W))
        self.borders_r[1] = tk.Button(self.right_column, text='Cancel', font=('Courier', 8))
        self.borders_r[1].grid(row=5, column=1, sticky=(E,W))
        self.borders_r[2] = tk.Button(self.right_column, text='Submit Order', font=('Courier', 8))
        self.borders_r[2].grid(row=6, column=0, columnspan=2, sticky=(E,W))

    def binding_events(self, root):
        root.bind('0', lambda event: self.keypress_handler(event))
        root.bind('1', lambda event: self.keypress_handler(event))
        root.bind('2', lambda event: self.keypress_handler(event))
        root.bind('3', lambda event: self.keypress_handler(event))
        root.bind('4', lambda event: self.keypress_handler(event))
        root.bind('.', lambda event: self.keypress_handler(event))
    
    def flicker(self):
        sleep(0.3)
        self.gesture_label.config(bg='black')

    def keypress_handler(self, event):
        if event.char == '/':
            self.gesture_text.set(f"Gesture: {'n/a'.center(10)}")
        elif event.char == '.':
            self.gesture_label.config(bg='green')
            Thread(target=self.flicker).start()
        else:
            self.gesture_text.set(f'Gesture: {self.gestures_list[int(event.char)].center(10)}')

            # selection_mode: 1 - left menu, 2 - right menu, 3/4 - confirm menu
            if self.selection_mode == 1:
                if event.char == '1': self.toggle_menu()
                elif event.char == '2':
                    if self.selection_counter_r == 0: self.confirm(func=self.order)
                    elif self.selection_counter_r == 1: self.confirm(func=self.cancel)
                elif event.char == '3': self.scroll_selection_l()

            elif self.selection_mode == 2:
                if event.char == '1':
                    if self.selection_counter_r != 2: self.toggle_menu()
                if event.char == '2':
                    if self.selection_counter_r == 2: self.confirm(func=self.submit_order)
                    else: self.toggle_menu()
                elif event.char == '3': self.scroll_selection_r()

            elif self.selection_mode == 3 or self.selection_mode == 4:
                if event.char == '2':
                    self.info_text.set('')
                    self.selection_mode -= 2
                    self.function_aft_confirm()
                if event.char == '4':
                    self.info_text.set('')
                    self.selection_mode -= 2

    def confirm(self, func):
        self.info_text.set('Confirm? (Y:push / N:waving)')
        self.function_aft_confirm = func
        self.selection_mode += 2

    def toggle_menu(self):
        self.selection_mode = 1 if self.selection_mode == 2 else 2
        if self.selection_mode == 1:
            for i in range(self.menu_max):
                self.borders_l[i].config(bg='blue' if i==self.selection_counter_l else 'white')
            for i in range(3):
                self.borders_r[i].config(fg='green' if i==self.selection_counter_r else 'black')
        elif self.selection_mode == 2:
            for i in range(self.menu_max):
                self.borders_l[i].config(bg='white')
            for i in range(3):
                self.borders_r[i].config(fg='blue' if i==self.selection_counter_r else 'black')

    def update_cart(self):
        self.total_price_text.set(f'Total Bill: ${np.sum(self.menu_prices*self.menu_qty):.2f}')
        self.cart_text.set('\n'.join([f'{name.ljust(20)}x {str(qty).rjust(2)}' for name, qty in zip(self.menu_names, self.menu_qty) if qty > 0]))

    def order(self):
        self.menu_qty[self.selection_counter_l] = self.menu_qty[self.selection_counter_l] + 1
        self.update_cart()

    def cancel(self):
        if self.menu_qty[self.selection_counter_l] > 0:
            self.menu_qty[self.selection_counter_l] = self.menu_qty[self.selection_counter_l] - 1
            self.update_cart()

    def scroll_selection_l(self):
        self.selection_counter_l = (self.selection_counter_l + 1) % self.menu_max
        for i in range(self.menu_max):
            self.borders_l[i].config(bg='blue' if i==self.selection_counter_l else 'white')

    def scroll_selection_r(self):
        self.selection_counter_r = (self.selection_counter_r + 1) % 3
        for i in range(3):
            self.borders_r[i]['state'] = NORMAL if i==self.selection_counter_r else DISABLED
            self.borders_r[i].config(fg='blue' if i==self.selection_counter_r else 'black')

    def submit_order(self):
        self.info_text.set('Order Submitted!')
        self.menu_qty = np.zeros(self.menu_prices.shape, dtype=int)
        self.update_cart()


root = Tk()
app = Menu(root)

root.mainloop()