import time
import datetime
import os
from matplotlib import pyplot as plt
from scipy.signal import firwin
from functools import reduce
from matplotlib.colors import ListedColormap
from numpy import array, loadtxt, shape, pi, divide, tile, arange, cos, sin, sqrt, rint, logical_and, ones,linspace, convolve, ndarray

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
from kivy.factory import Factory
from kivy.utils import platform
from kivy.core.window import Window
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

if platform == 'android':
    from android.permissions import request_permissions, Permission
    request_permissions([
        Permission.WRITE_EXTERNAL_STORAGE,
        Permission.READ_EXTERNAL_STORAGE,
        Permission.INTERNET,
    ])

class LoadDialog(FloatLayout):
    # floatlayout指滑动的时候浮动的部分
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

# class SaveDialog(FloatLayout):
#     save = ObjectProperty(None)
#     text_input = ObjectProperty(None)
#     cancel = ObjectProperty(None)

class UploadScreen(Screen):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.7, 0.7))
        self._popup.open()

    def load(self, path, filename):
        print(filename)
        if len(filename) > 0:
            with open(os.path.join(path, filename[0])) as f:
                # 读取全部文件内容f.read(),'w'写入内容
                print(f.read(), file=open("output.txt", "w"))
                # 关闭文件
                f.close()

        self.dismiss_popup()
    pass

class LoadingScreen(Screen):
    pass

class PATScreen(Screen):
    pass

class ScreenManagement(ScreenManager):

    def __init__(self, **kwargs):
        super(ScreenManagement, self).__init__(**kwargs)
        Window.bind(on_keyboard=self.on_key)

    def on_key(self, window, key, *args):
        if key == 27:  # the esc key
            if self.current_screen.name == "upload_screen":
                return False  # exit the app from this page
            elif self.current_screen.name == "pat_screen":
                self.current = 'upload_screen'
                return True

GUI = Builder.load_file("main.kv")

class MainApp(App):
    # _stopper = threading.Event()
    #
    # def start_second_thread(self):
    #     threading.Thread(target=self.second_thread).start()

    def start_reconstruction(self):
        # Clock.schedule_once(self.start_test, 0)
        # self.start_test()
        start_time = time.time()
        filename_pa = 'output.txt'  # input data filename

        fs = 25  # sampling freq (in MHz)

        x_size = 25e-3  # reconstructed image size in m
        y_size = 25e-3
        c_x = 0  # r center of the reconstructed image in m
        c_y = 0
        Npixel_x = 250  # number of pixels in x direction
        Npixel_y = 250  # number of pixels in y direction

        radius_input = float(self.root.ids['upload_screen'].ids['radius_input'].text)
        first_radius = int(radius_input)
        sec_radius = int(first_radius + 1)
        radius = array(range(first_radius, sec_radius))

        angle_step = int(float(self.root.ids['upload_screen'].ids['anglestep_input'].text))

        offset = array([0])  # [%] another parameter to fine tune, 0 = normal reconstruction
        delay1 = 0  # in %
        fcut = array([0.1, 8])  # cutoff freq
        forder = 256  # order of the filter
        soundv = 1500  # sound velocity (in m/s)

        pa_data = loadtxt(filename_pa, dtype='float').transpose()  # PA signal (1)

        # [Nsample, Nstep] = shape(pa_data)
        #
        # if Nstep==600:
        #     pa_data=pa_data.reshape(-1, 3).mean(1).reshape(Nsample, -1)


        [Nsample, Nstep] = shape(pa_data)  # Nstep: no. of steps, Nsample = no. of samples per step
        angle_per_step = 2 * pi / Nstep

        # FILTERING #
        bpf = firwin(numtaps=forder + 1, cutoff=divide(fcut, fs / 2), pass_zero='bandpass')
        pa_data = array([convolve(pa_data[:,i], bpf, 'same') for i in range(0, Nstep)]).transpose()

        # RECONSTRUCTION GRID #
        x_scale = Npixel_x / x_size
        y_scale = Npixel_y / y_size
        c_pixel_x = round(c_x * x_scale)
        c_pixel_y = round(c_y * y_scale)
        x_img = tile((arange(1, Npixel_x + 1) - ((Npixel_x + 1) / 2 + c_pixel_x)) * x_size / (Npixel_x - 1),
                        (250, 1))
        y_img = tile((((Npixel_y + 1) / 2 + c_pixel_y) - arange(1, Npixel_y + 1)) * y_size / (Npixel_y - 1),
                        (250, 1)).transpose()

        print('Please be patient while the PAT/TAT image reconstruction is going on...')

        #delay = tile((delay1 + offset) / (fs * 1e6), (len(radius), 1))
        delay = (delay1 + offset) / (fs * 1e6)  # (2)
        # RECONSTRUCTION #
        #centerpt = tile(radius, (
        #    len(offset), 1)).transpose() + offset  # sample point corresponding to the center of circular scan
        centerpt = radius + offset  # (3)
        R = ndarray.flatten(centerpt / (fs * 1e6) * soundv)

        # RECEIVER POSITION #
        angle_step1 = angle_step / 180 * pi
        x_receive = array([cos(arange(0, Nstep) * angle_per_step + angle_step1) * r for r in
                              R])  # vectorized; includes x_receive for all radius
        y_receive = array([sin(arange(0, Nstep) * angle_per_step + angle_step1) * r for r in
                              R])  # vectorized ; includes y_receive for all radius
        pa_img = np.zeros((250, 250))  # (4)
        for i in range(0, Nstep):
            pa_data_tmp = pa_data[:, i]
            pa_data_tmp = np.r_[pa_data_tmp, np.array([0])]
            idx = rint(
                (sqrt((x_img - x_receive[0, i]) ** 2 + (y_img - y_receive[0, i]) ** 2) / soundv - delay) * fs * 1e6)
            inrange = logical_and(logical_and((idx >= 1), (idx <= Nsample)), logical_and(
                (sqrt(x_img * x_img + y_img * y_img) < (centerpt / (fs * 10 ** 6) * soundv)),
                (sqrt((x_img - x_receive[0, i]) ** 2 + (
                        y_img - y_receive[0, i]) ** 2) > delay * soundv)))
            idx = inrange * idx + (1 - inrange) * (Nsample + 1)
            r0 = sqrt(x_receive[0, i] ** 2 + y_receive[0, i] ** 2)
            dx = x_img - x_receive[0, i]
            dy = y_img - y_receive[0, i]
            rr0 = sqrt(dx ** 2 + dy ** 2)
            angle_correction = (-x_receive[0, i] * dx - y_receive[0, i] * dy) / r0 / (rr0 ** 3) / 4 / pi
            t = np.zeros((250, 250))
            for i in range(250):
                for j, k in enumerate(idx[i, :]):
                    k = int(k - 1)
                    t[i, j] = pa_data_tmp[k]
            pa_img = pa_img + t

        # Colourmap Settings
        N = 64
        vals = ones((N, 3))
        vals[:, 0] = linspace(1, 0, N)
        vals[:, 1] = linspace(1, 0, N)
        vals[:, 2] = linspace(1, 0, N)
        gray2 = ListedColormap(vals)

        plt.figure()
        im = plt.imshow(pa_img.astype(float),cmap=gray2)
        plt.gca()
        plt.xticks([])
        plt.yticks([])
        plt.axis('image')
        plt.colorbar(im,fraction=0.046, pad=0.04)
        plt.suptitle(radius, fontsize=20)
        execution_time = round((time.time() - start_time), 1)
        plt.figtext(0.5, 0.05, 'Reconstruction time: ' + str(execution_time) + ' s', ha = 'center', fontsize=20)
        reconstructed_image = FigureCanvasKivyAgg(plt.gcf())
        self.image = self.root.ids['pat_screen'].ids['image']
        self.image.add_widget(reconstructed_image)
        # time.sleep(2)  # Do some more blocking operations.
        # self._stopper_test()  # Remove some widgets and update some properties in the main thread by decorating the called function with @mainthread.
        print('Reconstruction done')
        self.change_screen('pat_screen')

    # def start_test(self, *args):
    #     l = Label(text='[color=000000]Reconstructing PAT image...[/color]',font_size='10sp', markup=True)
    #     self.anim_box = self.root.ids['pat_screen'].ids['anim_box']
    #     self.anim_box.add_widget(l)


    # @mainthread
    # def _stopper_test(self):
    #     self.anim_box = self.root.ids['pat_screen'].remove_widget(self.anim_box)


    def build(self):
        return GUI
        pass

    def change_screen(self, screen_name):
        # Get the screen manager from the kv file
        screen_manager = self.root.ids['screen_manager']
        screen_manager.current = screen_name

    def clear_data(self):
        plt.cla()
        self.image.clear_widgets()
        self.root.ids['upload_screen'].ids.anglestep_input.text = ''
        self.root.ids['upload_screen'].ids.radius_input.text = ''

    def save_image(self):
        currentDT = datetime.datetime.now()
        currentDT = currentDT.strftime("%Y%m%d%H%M%S.png")
        my_path = '/storage/emulated/0/DCIM/PAT'+ currentDT
        plt.savefig(my_path)


Factory.register('UploadScreen', cls=UploadScreen)
Factory.register('LoadDialog', cls=LoadDialog)
# Factory.register('SaveDialog', cls=SaveDialog)


MainApp().run()