import tkinter as tk

app = tk.Tk()
app.geometry('475x570')
app.title('DB Manager')

## MAIN FUNCTIONS ##


def truncate_logs(table_name):
    pass

def random_tracking_logs():
    pass


table_names = ['hi', 'hello']
selections1 = tk.StringVar(app)
selections1.set(table_names[-2])
selections2 = tk.StringVar(app)
selections2.set(table_names[-2])

lbl1 = tk.Label(app, text='Clear all logs (Warning: No confirmation message!)', font='monospace 8', fg='red')
but1 = tk.Button(app, text='TRUNCATE ble_rssi_logs', command=lambda:truncate_logs('ble_rssi_logs'))
but2 = tk.Button(app, text='TRUNCATE camera_aruco_logs', command=lambda:truncate_logs('camera_aruco_logs'))
but3 = tk.Button(app, text='TRUNCATE tracking_logs', command=lambda:truncate_logs('tracking_logs'))
lbl2 = tk.Label(app, text='Select table and grant/revoke privileges to \'experiment\'', font='monospace 8', fg='red')
but4 = tk.Button(app, text='GRANT experiment', command=lambda:truncate_logs('ble_rssi_logs'))
but5 = tk.Button(app, text='REVOKE experiment', command=lambda:truncate_logs('ble_rssi_logs'))
opt1 = tk.OptionMenu(app, selections1, *table_names)
lbl3 = tk.Label(app, text='Select table and grant/revoke privileges to \'frontend\'', font='monospace 8', fg='red')
but6 = tk.Button(app, text='GRANT frontend', command=lambda:truncate_logs('ble_rssi_logs'))
but7 = tk.Button(app, text='REVOKE frontend', command=lambda:truncate_logs('ble_rssi_logs'))
opt2 = tk.OptionMenu(app, selections2, *table_names)
lbl4 = tk.Label(app, text='Miscellaneous commands (Use with CAUTION!)', font='monospace 8', fg='red')
but8 = tk.Button(app, text='INSERT 100 entries INTO tracking_logs', command=random_tracking_logs)
but9 = tk.Button(app, text='Populate new DB', command=truncate_logs('ble_rssi_logs'))

lbl1.place(relx=0.5, y=20, anchor=tk.CENTER)
but1.place(relx=0.5, y=55, anchor=tk.CENTER)
but2.place(relx=0.5, y=95, anchor=tk.CENTER)
but3.place(relx=0.5, y=135, anchor=tk.CENTER)
lbl2.place(relx=0.5, y=190, anchor=tk.CENTER)
but4.place(relx=0.25, y=225, anchor=tk.CENTER)
but5.place(relx=0.7, y=225, anchor=tk.CENTER)
opt1.place(relx=0.5, y=265, anchor=tk.CENTER)
lbl3.place(relx=0.5, y=320, anchor=tk.CENTER)
but6.place(relx=0.25, y=355, anchor=tk.CENTER)
but7.place(relx=0.7, y=355, anchor=tk.CENTER)
opt2.place(relx=0.5, y=395, anchor=tk.CENTER)
lbl4.place(relx=0.5, y=450, anchor=tk.CENTER)
but8.place(relx=0.5, y=485, anchor=tk.CENTER)
but9.place(relx=0.5, y=525, anchor=tk.CENTER)

app.mainloop()