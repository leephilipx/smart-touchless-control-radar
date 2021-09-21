from tkinter import *
from tkinter import ttk
from tkinter import messagebox

from daq_functions import AcconeerSensorDataCollection

radar = AcconeerSensorDataCollection(method='serial', Nframes=128, config_path='sensor_configs.json')

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
        filename_entry.grid(row=4, column=1, sticky=(W, E))
        
        ttk.Button(mainframe, text="Autoconnect", command=self.autoconnect).grid(row=1, column=1)
        self.start_b = ttk.Button(mainframe, text="Start Measurement", command=self.start)
        self.start_b.grid(row=2, column=3)
        self.next_b = ttk.Button(mainframe, text="Next", command=self.next)
        self.next_b.grid(row=3, column=1, sticky=W)
        self.discard_b = ttk.Button(mainframe, text="Discard", command=self.discard)
        self.discard_b.grid(row=3, column=3)
        self.stop_b = ttk.Button(mainframe, text="Stop", command=self.stop)
        self.stop_b.grid(row=3, column=5)
        
        self.file = ttk.Label(mainframe, text="Filename")
        self.file.grid(row=4, column=2, sticky=(N,W))
        self.counter = ttk.Label(mainframe, text="Count = 0")
        self.counter.grid(row=4, column=5)

        ttk.Separator(mainframe, orient=HORIZONTAL).grid(row=5, columnspan=7, sticky=(E,W))
        frame2 = ttk.Labelframe(mainframe, text='Magnitude plot', height=250, width=500)
        frame2.grid(row=6, column=1, columnspan=6, sticky=(N,W))

        ttk.Separator(mainframe, orient=HORIZONTAL).grid(row=9, columnspan=7, sticky=(E,W))
        frame3 = ttk.Labelframe(mainframe, text='Config info', height=250, width=300)
        frame3.grid(row=10, column=1, columnspan=6, sticky=(N,W))
        config_info = ""
        ttk.Label(frame3, text=config_info).grid(row=0, column=1, sticky=(W))
      
        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)
        
        filename_entry.focus()
        root.bind("<Return>", self.filename)    

    def autoconnect(self, *args):
        port = radar.autodetect_serial_port()
        radar.connect_sensor(port)
        print("Autoconnect")

    def start(self, *args):
        self.next_b['state'] = DISABLED
        self.discard_b['state'] = DISABLED
        self.stop_b['state'] = DISABLED
        radar.start_session()
        self.increase_count(True)
        print("Start")

    def next(self):
        self.increase_count(True)
        print("Next")

    def discard(self, *args):
        if root.counter!=0:
            self.increase_count(False)
        else:
            pass
        print("Discard")

    def stop(self, *args):
        radar.stop_session()
        print("Stop")
    
    def filename(self, *args):
        print("Save filname into file")
    
    def increase_count(self, increase):
        if root.counter <0:
            messagebox.showinfo(message='Count cannot be less than zero!')
        else:
            if increase == True:
                root.counter +=1
            elif increase == False:
                root.counter -=1
            self.counter.config(text=f"Count = {root.counter}")
    
root = Tk()
root.counter = 0
DataAcquisiton(root)
root.mainloop()

