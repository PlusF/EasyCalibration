import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from calibrator import Calibrator
from dataloader import DataLoader
from MyTooltip import MyTooltip

font_lg = ('Arial', 24)
font_md = ('Arial', 16)
font_sm = ('Arial', 12)

plt.rcParams['font.family'] = 'Arial'

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25

plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = 35         # 軸ラベルのフォントサイズ
plt.rcParams['axes.linewidth'] = 1.0        # グラフ囲う線の太さ

plt.rcParams['legend.loc'] = 'best'        # 凡例の位置、"best"でいい感じのところ
plt.rcParams['legend.frameon'] = True       # 凡例を囲うかどうか、Trueで囲う、Falseで囲わない
plt.rcParams['legend.framealpha'] = 1.0     # 透過度、0.0から1.0の値を入れる
plt.rcParams['legend.facecolor'] = 'white'  # 背景色
plt.rcParams['legend.edgecolor'] = 'black'  # 囲いの色
plt.rcParams['legend.fancybox'] = False     # Trueにすると囲いの四隅が丸くなる

plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['figure.subplot.top'] = 0.95
plt.rcParams['figure.subplot.bottom'] = 0.15
plt.rcParams['figure.subplot.left'] = 0.1
plt.rcParams['figure.subplot.right'] = 0.95


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
        self.master.bind('<Control-Key-z>', self.undo)

        self.x0, self.y0, self.x1, self.y1 = 0, 0, 0, 0
        self.rectangles = []
        self.ranges = []
        self.drawing = False
        self.rect_drawing = None

        self.dl_raw = DataLoader()
        self.dl_ref = DataLoader()
        self.calibrator = Calibrator(measurement='Raman', material='sulfur', dimension=1)
        self.create_widgets()

    def create_widgets(self) -> None:
        # スタイル設定
        style = ttk.Style()
        style.theme_use('winnative')
        style.configure('TButton', font=font_md, width=14, padding=[0, 4, 0, 4], foreground='black')
        style.configure('R.TButton', font=font_md, width=14, padding=[0, 4, 0, 4], foreground='red')
        style.configure('TLabel', font=font_sm, padding=[0, 4, 0, 4], foreground='black')
        style.configure('Color.TLabel', font=font_lg, padding=[0, 0, 0, 0], width=4, background='black')
        style.configure('TEntry', font=font_md, width=14, padding=[0, 4, 0, 4], foreground='black')
        style.configure('TCheckbutton', font=font_md, padding=[0, 4, 0, 4], foreground='black')
        style.configure('TMenubutton', font=font_md, padding=[20, 4, 0, 4], foreground='black')
        style.configure('TCombobox', font=font_md, padding=[20, 4, 0, 4], foreground='black')
        style.configure('TTreeview', font=font_md, foreground='black')

        self.width = 400
        self.height = 200
        dpi = 50
        if os.name == 'posix':
            fig = plt.figure(figsize=(self.width / 2 / dpi * 2, self.height / 2 / dpi * 3), dpi=dpi)
        else:
            fig = plt.figure(figsize=(self.width / dpi * 2, self.height / dpi * 3), dpi=dpi)

        fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('motion_notify_event', self.draw_preview)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, self.master)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=3)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=3, column=0)

        frame_download = ttk.LabelFrame(self.master, text='Data to calibrate', width=self.width, height=self.height)
        frame_ref = ttk.LabelFrame(self.master, text='Reference', width=self.width, height=self.height)
        frame_msg = ttk.LabelFrame(self.master, text='Message', width=self.width, height=self.height)
        frame_button = ttk.LabelFrame(self.master, text='', width=self.width)
        frame_download.grid(row=0, column=1)
        frame_ref.grid(row=1, column=1)
        frame_msg.grid(row=2, column=1)
        frame_button.grid(row=3, column=1)

        # frame_listbox
        self.treeview = ttk.Treeview(frame_download, height=6, selectmode=tk.EXTENDED)
        self.treeview['columns'] = ['filename']
        self.treeview.column('#0', width=40, stretch=tk.NO)
        self.treeview.column('filename', width=400, anchor=tk.CENTER)
        self.treeview.heading('#0', text='#')
        self.treeview.heading('filename', text='filename')
        self.treeview.bind('<<TreeviewSelect>>', self.select_data)
        self.treeview.bind('<Button-2>', self.delete_data)
        self.treeview.bind('<Button-3>', self.delete_data)

        self.button_download = ttk.Button(frame_download, text='DOWNLOAD', command=self.download, state=tk.DISABLED)
        self.treeview.pack()
        self.button_download.pack()

        # frame_ref
        self.filename_ref = tk.StringVar(value='')
        label_ref = ttk.Label(frame_ref, textvariable=self.filename_ref)
        label_ref.bind('<Button-1>', lambda e: self.show_spectrum_ref())
        label_ref.bind('<Button-2>', lambda e: self.delete_spectrum_ref())
        self.tooltip = MyTooltip(label_ref, '')

        self.measurement = tk.StringVar(value=self.calibrator.get_measurement_list()[0])
        self.material = tk.StringVar(value=self.calibrator.get_material_list()[0])
        self.center = tk.DoubleVar(value=630)
        self.dimension = tk.StringVar(value=self.calibrator.get_dimension_list()[0])
        self.function = tk.StringVar(value=self.calibrator.get_function_list()[0])
        self.easy = tk.BooleanVar(value=False)
        self.optionmenu_function = ttk.OptionMenu(frame_ref, self.function, self.calibrator.get_function_list()[0], *self.calibrator.get_function_list())
        self.optionmenu_function.config(width=10)
        self.optionmenu_function['menu'].config(font=font_sm)
        optionmenu_measurement = ttk.OptionMenu(frame_ref, self.measurement, self.calibrator.get_measurement_list()[0], *self.calibrator.get_measurement_list(), command=self.change_measurement)
        optionmenu_measurement.config(width=10)
        optionmenu_measurement['menu'].config(font=font_sm)
        self.optionmenu_material = ttk.OptionMenu(frame_ref, self.material, self.calibrator.get_material_list()[0], *self.calibrator.get_material_list())
        self.optionmenu_material.config(width=10)
        self.optionmenu_material['menu'].config(font=font_sm)
        self.combobox_center = ttk.Combobox(frame_ref, textvariable=self.center, values=[500, 630, 760], justify=tk.CENTER, state=tk.DISABLED)
        self.combobox_center.config(width=10)
        optionmenu_dimension = ttk.OptionMenu(frame_ref, self.dimension, *self.calibrator.get_dimension_list())
        optionmenu_dimension.config(width=10)
        optionmenu_dimension['menu'].config(font=font_sm)
        checkbutton_easy = ttk.Checkbutton(frame_ref, text='easy', variable=self.easy, command=self.switch_easy)
        self.button_calibrate = ttk.Button(frame_ref, text='CALIBRATE', command=self.calibrate, state=tk.DISABLED)

        label_ref.grid(row=0, column=0, columnspan=6)
        optionmenu_measurement.grid(row=1, column=0)
        self.optionmenu_material.grid(row=1, column=1)
        self.combobox_center.grid(row=1, column=2)
        optionmenu_dimension.grid(row=2, column=0)
        self.optionmenu_function.grid(row=2, column=1)
        checkbutton_easy.grid(row=2, column=2)
        self.button_calibrate.grid(row=3, column=0, columnspan=6)

        # frame_msg
        self.msg = tk.StringVar(value='Please drag & drop data files.')
        label_msg = ttk.Label(master=frame_msg, textvariable=self.msg)
        label_msg.pack()

        # frame_button
        button_reset = ttk.Button(frame_button, text='RESET', command=self.reset)
        button_help = ttk.Button(frame_button, text='HELP', command=self.show_help)
        button_reset.grid(row=0, column=0)
        button_help.grid(row=0, column=1)

        # canvas_drop
        self.canvas_drop = tk.Canvas(self.master, width=self.width * 3, height=self.height * 3)
        self.canvas_drop.create_rectangle(0, 0, self.width * 3, self.height * 1.5, fill='lightgray')
        self.canvas_drop.create_rectangle(0, self.height * 1.5, self.width * 3, self.height * 3, fill='gray')
        self.canvas_drop.create_text(self.width * 3 / 2, self.height * 3 / 4, text='Data to Calibrate',
                                     font=('Arial', 30))
        self.canvas_drop.create_text(self.width * 3 / 2, self.height * 9 / 4, text='Reference Data',
                                     font=('Arial', 30))

    def change_measurement(self, event=None):
        if self.measurement.get() == 'Raman':
            self.combobox_center.config(state=tk.DISABLED)
        elif self.measurement.get() == 'Rayleigh':
            self.combobox_center.config(state=tk.ACTIVE)
        self.calibrator.set_measurement(self.measurement.get())
        # update material
        self.optionmenu_material['menu'].delete(0, 'end')
        material_list = self.calibrator.get_material_list()
        for material in material_list:
            self.optionmenu_material['menu'].add_command(label=material, command=tk._setit(self.material, material))
        self.material.set(material_list[0])

    def assign_peaks(self):
        x_true = self.calibrator.get_true_x()
        found_x_true = []
        for x0, y0, x1, y1 in self.ranges:
            x_mid = (x0 + x1) / 2
            diff = np.abs(x_true - x_mid)
            idx = np.argmin(diff)
            found_x_true.append(x_true[idx])
        return found_x_true

    @update_plot
    def calibrate(self) -> None:
        if len(self.ranges) == 0:
            messagebox.showerror('Error', 'Choose range.')
            self.show_spectrum_ref()
            return
        if self.measurement.get() == 'Rayleigh':
            wavelength_range = 134
            initial_xdata = np.linspace(self.center.get() - wavelength_range / 2,
                                        self.center.get() + wavelength_range / 2,
                                        self.dl_ref.spec_dict[self.filename_ref.get()].xdata.shape[0])
        self.calibrator.set_measurement(self.measurement.get())
        self.calibrator.set_material(self.material.get())
        self.calibrator.set_dimension(int(self.dimension.get()[0]))
        self.calibrator.set_function(self.function.get())
        self.calibrator.set_search_width(50)
        ok = self.calibrator.calibrate(mode='manual', ranges=self.ranges, x_true=self.assign_peaks())
        if not ok:
            self.msg.set('Calibration failed.')
            return
        self.update_treeview()
        self.button_download.config(state=tk.ACTIVE)
        msg = 'Successfully calibrated.\nYou can now download the calibrated data.\n'

        self.setattr_to_all_raw('xdata', self.calibrator.xdata)
        self.setattr_to_all_raw('abs_path_ref', self.filename_ref.get())
        self.setattr_to_all_raw('calibration', self.calibrator.calibration_info)

        msg += 'Found peak, True value\n'
        for i, (fitted_x, true_x) in enumerate(zip(self.calibrator.fitted_x, self.calibrator.found_x_true)):
            msg += f'{fitted_x:.2f}, {true_x:.2f}\n'
        self.msg.set(msg)
        for r in self.rectangles:
            self.ax.add_patch(r)
        self.calibrator.show_fit_result(self.ax)

    def setattr_to_all_raw(self, key, value):
        for spec in self.dl_raw.spec_dict.values():
            setattr(spec, key, value)

    @update_plot
    def drop(self, event=None) -> None:
        self.canvas_drop.place_forget()

        master_geometry = list(map(int, self.master.winfo_geometry().split('+')[1:]))

        dropped_place = (event.y_root - master_geometry[1] - 30) / self.height

        threshold = 3 / 2

        if event.data[0] == '{':
            filenames = list(map(lambda x: x.strip('{').strip('}'), event.data.split('} {')))
        else:
            filenames = event.data.split()

        if dropped_place > threshold:  # reference data
            filename = filenames[0]
            if self.filename_ref.get() != '':
                self.dl_ref.delete_file(self.filename_ref.get())
            self.setattr_to_all_raw('abs_path_ref', filename)
            self.filename_ref.set(filename)
            self.dl_ref.load_file(filename)
            self.rectangles = []
            self.ranges = []
            self.show_spectrum(self.dl_ref.spec_dict[filename])
            self.check_data_type(filename)
            self.button_calibrate.config(state=tk.ACTIVE)
            self.button_download.config(state=tk.DISABLED)
        else:  # data to calibrate
            self.dl_raw.load_files(filenames)
            self.show_spectrum(self.dl_raw.spec_dict[filenames[0]])
            self.update_treeview()
            self.button_download.config(state=tk.DISABLED)

    def drop_enter(self, event: TkinterDnD.DnDEvent) -> None:
        self.canvas_drop.place(anchor='nw', x=0, y=0)

    def drop_leave(self, event: TkinterDnD.DnDEvent) -> None:
        self.canvas_drop.place_forget()

    def check_data_type(self, filename):
        if self.dl_ref.spec_dict[filename].device == 'Renishaw':
            self.calibrator.set_measurement('Raman')
            self.measurement.set('Raman')
        elif self.dl_ref.spec_dict[filename].device in ['Andor', 'CSS']:
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

        self.change_measurement()

    def switch_easy(self):
        if self.easy.get():
            self.optionmenu_function.config(state=tk.DISABLED)
        else:
            self.optionmenu_function.config(state=tk.ACTIVE)

    def update_treeview(self) -> None:
        self.treeview.delete(*self.treeview.get_children())
        for i, filename in enumerate(self.dl_raw.spec_dict.keys()):
            self.treeview.insert(
                '',
                tk.END,
                iid=str(i),
                text=str(i),
                values=[filename],
                open=True,
                )

    def show_spectrum(self, spec) -> None:
        self.ax.plot(spec.xdata, spec.ydata, color='k')

    @update_plot
    def show_spectrum_ref(self) -> None:
        if self.filename_ref.get() == '':
            return
        self.show_spectrum(self.dl_ref.spec_dict[self.filename_ref.get()])

    @update_plot
    def delete_spectrum_ref(self) -> None:
        if self.filename_ref.get() == '':
            return
        ok = messagebox.askyesno('確認', f'Delete {self.filename_ref.get()}?')
        if not ok:
            return
        self.dl_ref.delete_file(self.filename_ref.get())
        self.msg.set(f'Deleted {self.filename_ref.get()}.')
        self.filename_ref.set('')

    @update_plot
    def select_data(self, event) -> None:
        if self.treeview.focus() == '':
            return
        key = self.treeview.item(self.treeview.focus())['values'][0]
        self.show_spectrum(self.dl_raw.spec_dict[key])

    @update_plot
    def delete_data(self, event) -> None:
        if self.treeview.focus() == '':
            return
        key = self.treeview.item(self.treeview.focus())['values'][0]
        ok = messagebox.askyesno('確認', f'Delete {key}?')
        if not ok:
            return
        self.dl_raw.delete_file(key)

        self.update_treeview()
        self.msg.set(f'Deleted {key}.')

    def on_press(self, event):
        if event.xdata is None or event.ydata is None:
            return
        self.x0 = event.xdata
        self.y0 = event.ydata

        self.drawing = True

    def on_release(self, event):
        if event.xdata is None or event.ydata is None:
            return
        # Toolbarのズーム機能を使っている状態では動作しないようにする
        if self.toolbar._buttons['Zoom'].var.get():
            return

        # プレビュー用の矩形を消す
        if self.rect_drawing is not None:
            self.rect_drawing.remove()
            self.rect_drawing = None

        self.drawing = False

        self.x1 = event.xdata
        self.y1 = event.ydata
        if self.x0 == self.x1 or self.y0 == self.y1:
            return
        if self.is_overlapped(self.x0, self.x1):
            messagebox.showerror('Error', 'Overlapped.')
            return
        x0, x1 = sorted([self.x0, self.x1])
        y0, y1 = sorted([self.y0, self.y1])
        r = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor='r',
                              facecolor='none')
        self.ax.add_patch(r)
        self.rectangles.append(r)
        self.ranges.append((x0, y0, x1, y1))
        self.canvas.draw()

    def draw_preview(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if not self.drawing:
            return
        # Toolbarのズーム機能を使っている状態では動作しないようにする
        if self.toolbar._buttons['Zoom'].var.get():
            return
        if self.rect_drawing is not None:
            self.rect_drawing.remove()
        x1 = event.xdata
        y1 = event.ydata
        self.rect_drawing = patches.Rectangle((self.x0, self.y0), x1 - self.x0, y1 - self.y0, linewidth=0.5,
                                              edgecolor='r', linestyle='dashed', facecolor='none')
        self.ax.add_patch(self.rect_drawing)
        self.canvas.draw()

    def is_overlapped(self, x0, x1):
        for x0_, y0_, x1_, y1_ in self.ranges:
            if x0_ <= x0 <= x1_ or x0_ <= x1 <= x1_:
                return True
            if x0 <= x0_ <= x1 or x0 <= x1_ <= x1:
                return True
        return False

    def undo(self, event):
        if len(self.rectangles) == 0:
            return
        self.rectangles[-1].remove()
        self.rectangles.pop()
        self.ranges.pop()
        self.canvas.draw()

    def download(self) -> None:
        for filename in self.dl_raw.spec_dict.keys():
            self.dl_raw.save(filename)
        msg = 'Successfully downloaded.\n'
        for filename in self.calibrator.filename_raw_list:
            msg += os.path.basename(filename) + '\n'
        self.msg.set(msg)

    @update_plot
    def reset(self):
        self.rectangles = []
        self.ranges = []
        self.treeview.delete(*self.treeview.get_children())
        self.filename_ref.set('')
        self.dl_raw.__init__()
        self.dl_ref.__init__()
        self.calibrator.__init__()
        self.button_download.config(state=tk.DISABLED)
        self.button_calibrate.config(state=tk.DISABLED)
        self.msg.set(f'Reset.')

    def show_help(self):
        messagebox.showinfo('HELP', '''
        Raman\n
        sulfur: 86 ~ 470 cm-1\n
          naphthalene: 514 ~ 1576 cm-1\n
          1,4-Bis(2-methylstyryl)benzene: 1178 ~ 1627 cm-1\n
          acetonitrile: 2254 ~ 2940 cm-1\n\n
        参照ピークが・・・\n2本か3本しかないとき:Linear\n
        中心波長付近にあるとき: Quadratic\n
        全体に分布しているとき: Cubic\n\n
        easyをonにするとフィッティングなしで\n
          最大値抽出でピーク検出が行われます\n\n
        Rayleighのデータはずれが大きく正しくできない\n
          ことがあります。重要なものはSolisを使って\n
          キャリブレーションしてください
        ''')

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
