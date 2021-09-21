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
        filename_entry.grid(row=3, column=1, sticky=(W, E))

        counter = StringVar()
        counter_entry = ttk.Entry(mainframe, width=3, textvariable=counter)
        counter_entry.grid(row=3, column=5, sticky=(W, E))
        
        ttk.Button(mainframe, text="Start Measurement", command=self.loadgui).grid(row=1, column=3)
        ttk.Button(mainframe, text="Next", command=self.loadgui).grid(row=2, column=1, sticky=W)
        ttk.Button(mainframe, text="Discard", command=self.loadgui).grid(row=2, column=3)
        ttk.Button(mainframe, text="Stop", command=self.loadgui).grid(row=2, column=5)
        ttk.Label(mainframe, text="Filename").grid(row=3, column=2, sticky=(N,W))
        ttk.Label(mainframe, text="Counter").grid(row=3, column=6, sticky=(N,W))
        ttk.Separator(mainframe, orient=HORIZONTAL).grid(row=4, columnspan=7, sticky=(E,W))
        frame2 = ttk.Labelframe(mainframe, text='Magnitude plot', height=250, width=500)
        frame2.grid(row=5, column=1, columnspan=6, sticky=(N,W))
        ttk.Separator(mainframe, orient=HORIZONTAL).grid(row=9, columnspan=7, sticky=(E,W))
        frame3 = ttk.Labelframe(mainframe, text='Config info', height=250, width=300)
        frame3.grid(row=10, column=1, columnspan=6, sticky=(N,W))
        

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)
        
        filename_entry.focus()
        root.bind("<Return>", self.loadgui)    

    def loadgui(self, *args):
        print("Bunny")

root = Tk()
DataAcquisiton(root)
root.mainloop()

