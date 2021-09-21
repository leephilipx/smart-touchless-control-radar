from tkinter import *
from tkinter import ttk

# from test import 

class DataAcquisiton:
    def __init__(self, root):
        root.title("Data Acquisition")
        mainframe = ttk.Frame(root, padding="12 12 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.create_widgets(mainframe)

    def create_widgets(self, mainframe):
        filename = StringVar()
        filename_entry = ttk.Entry(mainframe, width=7, textvariable=filename)
        filename_entry.grid(column=2, row=2, sticky=(W, E))
        filename_entry.focus()
        
        ttk.Button(mainframe, text="Start Measurement", command=self.loadgui).grid(column=2, row=1, sticky=W)
        ttk.Label(mainframe, text="Filename").grid(column=3, row=2, sticky=(N,W))
        ttk.Button(mainframe, text="Next", command=self.loadgui).grid(column=1, row=3, sticky=W)
        ttk.Button(mainframe, text="Stop", command=self.loadgui).grid(column=3, row=3, sticky=E)

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)
        
        root.bind("<Return>", self.loadgui)    

    def loadgui(self, *args):
        print("Bunny")

root = Tk()
DataAcquisiton(root)
root.mainloop()

