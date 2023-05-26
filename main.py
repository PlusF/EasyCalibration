import os
import tkinter as tk
from tkinter import messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from EasyCalibrator import EasyCalibrator


def Lorentzian(x: np.ndarray, x0: float, a: float, b: float, c: float) -> np.ndarray:
    return a / ((x - x0) ** 2 + b) + c


def update_plot(func):
    def wrapper(*args, **kwargs):
        args[0].ax.clear()
        ret = func(*args, **kwargs)
        args[0].canvas.draw()
        return ret
    return wrapper


class MainWindow(tk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.calibrator = EasyCalibrator()
        self.create_widgets()

    def create_widgets(self) -> None:
        self.width = 400
        self.height = 200
        dpi = 50
        if os.name == 'posix':
            fig = plt.figure(figsize=(self.width / 2/  dpi, self.height / 2 / dpi), dpi=dpi)
        else:
            fig = plt.figure(figsize=(self.width / dpi, self.height / dpi), dpi=dpi)
        self.ax = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, self.master)
        self.canvas.get_tk_widget().grid(row=1, column=1)

        frame_msg = tk.LabelFrame(self.master, text='Message', width=self.width, height=self.height)
        frame_help = tk.LabelFrame(self.master, text='Help', width=self.width, height=self.height * 2)
        frame_listbox = tk.LabelFrame(self.master, text='Data to calibrate', width=self.width, height=self.height)
        frame_ref = tk.LabelFrame(self.master, text='Reference', width=self.width, height=self.height)
        frame_msg.grid(row=0, column=0)
        frame_listbox.grid(row=0, column=1)
        frame_help.grid(row=0, column=2, rowspan=2)
        frame_ref.grid(row=1, column=0)
        frame_msg.pack_propagate(False)
        frame_ref.grid_propagate(False)
        frame_help.pack_propagate(False)
        frame_listbox.pack_propagate(False)

        # frame_msg
        self.msg = tk.StringVar(value='Please drag & drop data files.')
        label_msg = tk.Label(master=frame_msg, textvariable=self.msg)
        label_msg.pack()

        # frame_ref
        self.filename_ref = tk.StringVar()
        entry_ref = tk.Entry(frame_ref, textvariable=self.filename_ref, width=40)
        entry_ref.bind('<Button-1>', lambda e: self.show_spectrum_ref())
        entry_ref.bind('<Button-2>', lambda e: self.delete_spectrum_ref())

        self.measurement = tk.StringVar(value=self.calibrator.get_measurement_list()[0])
        optionmenu_measurement = tk.OptionMenu(frame_ref, self.measurement, *self.calibrator.get_measurement_list(), command=self.update_material)
        self.center = tk.DoubleVar(value=self.calibrator.center)
        combobox_center = ttk.Combobox(frame_ref, textvariable=self.center, values=[500, 630, 760], width=7, justify=tk.CENTER)
        self.material = tk.StringVar(value=self.calibrator.get_material_list()[0])
        self.optionmenu_material = tk.OptionMenu(frame_ref, self.material, *self.calibrator.get_material_list())
        self.dimension = tk.StringVar(value=self.calibrator.get_dimension_list()[0])
        optionmenu_dimension = tk.OptionMenu(frame_ref, self.dimension, *self.calibrator.get_dimension_list())
        self.function = tk.StringVar(value=self.calibrator.get_function_list()[0])
        self.optionmenu_function = tk.OptionMenu(frame_ref, self.function, *self.calibrator.get_function_list())
        self.easy = tk.BooleanVar(value=False)
        checkbutton_easy = tk.Checkbutton(frame_ref, text='easy', variable=self.easy, command=self.switch_easy)
        self.button_calibrate = tk.Button(frame_ref, text='CALIBRATE', command=self.calibrate, state=tk.DISABLED)

        entry_ref.grid(row=0, column=0, columnspan=6)
        optionmenu_measurement.grid(row=1, column=0)
        combobox_center.grid(row=1, column=1)
        self.optionmenu_material.grid(row=1, column=2)
        optionmenu_dimension.grid(row=2, column=0)
        self.optionmenu_function.grid(row=2, column=1)
        checkbutton_easy.grid(row=2, column=2)
        self.button_calibrate.grid(row=3, column=0, columnspan=6)

        # frame_msg
        self.help = tk.StringVar(value='Raman\n'
                                       '  sulfur: 86 ~ 470 cm-1\n'
                                       '  naphthalene: 514 ~ 1576 cm-1\n'
                                       '  1,4-Bis(2-methylstyryl)benzene: 1178 ~ 1627 cm-1\n'
                                       '  acetonitrile: 2254 ~ 2940 cm-1\n\n'
                                       '参照ピークが・・・\n2本か3本しかないとき:Linear\n'
                                       '中心波長付近にあるとき: Quadratic\n'
                                       '全体に分布しているとき: Cubic\n\n'
                                       'easyをonにするとフィッティングなしで\n'
                                       '  最大値抽出でピーク検出が行われます\n\n'
                                       'Rayleighのデータはずれが大きく正しくできない\n'
                                       '  ことがあります。重要なものはSolisを使って\n'
                                       '  キャリブレーションしてください。')
        label_help = tk.Label(master=frame_help, textvariable=self.help, justify=tk.LEFT)
        label_help.pack()

        # frame_before
        self.listbox_before = tk.Listbox(frame_listbox, selectmode="extended", height=8, width=40)
        self.listbox_before.bind('<Button-1>', self.select_spectrum)
        self.listbox_before.bind('<Button-2>', self.delete_spectrum)
        self.button_download = tk.Button(frame_listbox, text='DOWNLOAD', command=self.download, state=tk.DISABLED)
        self.listbox_before.pack()
        self.button_download.pack()

        # canvas_drop
        self.canvas_drop = tk.Canvas(self.master, width=self.width * 3, height=self.height * 2)
        self.canvas_drop.create_rectangle(0, 0, self.width * 3, self.height, fill='lightgray')
        self.canvas_drop.create_rectangle(0, self.height, self.width * 3, self.height * 2, fill='gray')
        self.canvas_drop.create_text(self.width * 3 / 2, self.height / 2, text='Data to Calibrate',
                                     font=('Arial', 30))
        self.canvas_drop.create_text(self.width * 3 / 2, self.height * 3 / 2, text='Reference Data',
                                     font=('Arial', 30))

    def update_material(self, event=None):
        self.optionmenu_material['menu'].delete(0, 'end')
        material_list = self.calibrator.get_material_list()
        for material in material_list:
            self.optionmenu_material['menu'].add_command(label=material, command=tk._setit(self.material, material))
        self.material.set(material_list[0])

    @update_plot
    def calibrate(self) -> None:
        if self.measurement.get() == 'Rayleigh':
            self.calibrator.set_initial_xdata(self.center.get())
        self.calibrator.set_measurement(self.measurement.get())
        self.calibrator.set_material(self.material.get())
        self.calibrator.set_dimension(int(self.dimension.get()[0]))
        self.calibrator.set_function(self.function.get())
        ok = self.calibrator.calibrate(easy=self.easy.get())
        if not ok:
            self.msg.set('Calibration failed.')
            return
        self.update_listbox()
        self.button_download.config(state=tk.ACTIVE)
        self.msg.set('Successfully calibrated.\nYou can now download the calibrated data.')

        self.setattr_to_all_raw('xdata', self.calibrator.xdata)
        self.setattr_to_all_raw('abs_path_ref', self.calibrator.filename_ref)
        self.setattr_to_all_raw('calibration', self.calibrator.calibration_info)

        self.calibrator.show_fit_result(self.ax)

    def setattr_to_all_raw(self, key, value):
        for filename in self.calibrator.filename_raw_list:
            setattr(self.calibrator.spec_dict[filename], key, value)

    @update_plot
    def drop(self, event=None) -> None:
        self.canvas_drop.place_forget()

        master_geometry = list(map(int, self.master.winfo_geometry().split('+')[1:]))

        dropped_place = (event.y_root - master_geometry[1] - 30) / self.height

        if os.name == 'posix':
            threshold = 1
        else:
            threshold = 0.5
        filenames = [f.replace('{', '').replace('}', '') for f in event.data.split('} {')]

        if dropped_place > threshold:  # reference data
            filename = filenames[0]
            if self.calibrator.filename_ref != '':
                self.calibrator.delete_file(self.calibrator.filename_ref)
            self.setattr_to_all_raw('abs_path_ref', filename)
            self.filename_ref.set(filename)
            self.calibrator.load_ref(filename)
            self.show_spectrum(self.calibrator.spec_dict[filename])
            self.check_data_type(filename)
            self.button_calibrate.config(state=tk.ACTIVE)
            self.button_download.config(state=tk.DISABLED)
        else:  # data to calibrate
            self.calibrator.load_raw_list(filenames)
            self.show_spectrum(self.calibrator.spec_dict[filenames[0]])
            self.update_listbox()
            self.button_download.config(state=tk.DISABLED)

    def drop_enter(self, event: TkinterDnD.DnDEvent) -> None:
        self.canvas_drop.place(anchor='nw', x=0, y=0)

    def drop_leave(self, event: TkinterDnD.DnDEvent) -> None:
        self.canvas_drop.place_forget()

    def check_data_type(self, filename):
        if self.calibrator.spec_dict[filename].device == 'Renishaw':
            self.calibrator.set_measurement('Raman')
            self.measurement.set('Raman')
        elif self.calibrator.spec_dict[filename].device in ['Andor', 'CSS']:
            self.calibrator.set_measurement('Rayleigh')
            self.measurement.set('Rayleigh')
            self.material.set(self.calibrator.get_material_list()[0])
            self.easy.set(True)
            self.optionmenu_function.config(state=tk.DISABLED)
        for center in ['500', '630', '760']:
            if center in filename:
                self.center.set(float(center))
        for material in self.calibrator.get_material_list():
            if material in filename:
                self.material.set(material)

    def switch_easy(self):
        if self.easy.get():
            self.optionmenu_function.config(state=tk.DISABLED)
        else:
            self.optionmenu_function.config(state=tk.ACTIVE)

    def update_listbox(self) -> None:
        self.listbox_before.delete(0, tk.END)
        for filename in self.calibrator.filename_raw_list:
            self.listbox_before.insert(0, filename)
        if len(self.calibrator.filename_raw_list) == 0:
            self.button_download.config(state=tk.DISABLED)

    def show_spectrum(self, spec) -> None:
        self.ax.plot(spec.xdata, spec.ydata, color='k')

    @update_plot
    def show_spectrum_ref(self) -> None:
        if self.calibrator.filename_ref == '':
            return
        self.show_spectrum(self.calibrator.spec_dict[self.calibrator.filename_ref])

    @update_plot
    def delete_spectrum_ref(self) -> None:
        if self.calibrator.filename_ref == '':
            return
        ok = messagebox.askyesno('確認', f'Delete {self.filename_ref.get()}?')
        if not ok:
            return
        self.msg.set(f'Deleted {self.filename_ref.get()}.')
        self.filename_ref.set('')
        self.calibrator.delete_file(self.calibrator.filename_ref)
        self.calibrator.filename_ref = ''

    @update_plot
    def select_spectrum(self, event) -> None:
        if len(event.widget.curselection()) == 0:
            return
        key = event.widget.get(event.widget.curselection()[0])

        if event.widget == self.listbox_before:
            self.show_spectrum(self.calibrator.spec_dict[key])

    @update_plot
    def delete_spectrum(self, event) -> None:
        if len(event.widget.curselection()) == 0:
            return
        keys = []
        msg = ''
        for i in event.widget.curselection():
            key = event.widget.get(i)
            keys.append(key)
            msg += f'\n"{key}"'
        ok = messagebox.askyesno('確認', f'Delete {msg}?')
        if not ok:
            return

        if event.widget == self.listbox_before:
            for key in keys:
                self.calibrator.filename_raw_list.remove(key)
                self.calibrator.delete_file(key)

        self.update_listbox()
        self.msg.set(f'Deleted {msg}.')

    def download(self) -> None:
        self.calibrator.save_raw_files()
        msg = 'Successfully downloaded.\n'
        for filename in self.calibrator.filename_raw_list:
            msg += os.path.basename(filename) + '\n'
        self.msg.set(msg)

    def quit(self) -> None:
        self.master.quit()


def main():
    root = TkinterDnD.Tk()
    app = MainWindow(master=root)
    root.protocol('WM_DELETE_WINDOW', app.quit)
    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<DropEnter>>', app.drop_enter)
    root.dnd_bind('<<DropLeave>>', app.drop_leave)
    root.dnd_bind('<<Drop>>', app.drop)
    app.mainloop()


if __name__ == '__main__':
    main()
