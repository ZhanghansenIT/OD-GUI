import argparse
import time
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
from tkinter import filedialog
from tkinter import ttk
import torch
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
from mmdet.apis import init_detector, inference_detector

import numpy as np

import mmcv
import matplotlib.pyplot as plt
import cv2

# global ---
image_path_g = None


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()


def create_blank_image(width, height, color=(255, 255, 255)):
    return Image.new("RGB", (width, height), color)


def display_original_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((300, 400))
    photo = ImageTk.PhotoImage(image)
    original_label.config(image=photo)
    original_label.image = photo


def display_processed_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((300, 400))
    photo = ImageTk.PhotoImage(image)
    processed_label.config(image=photo)
    processed_label.image = photo


# vedio
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
from tkinter import filedialog
from tkinter import ttk
import torch
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
from mmdet.apis import init_detector, inference_detector

import numpy as np

import mmcv
import matplotlib.pyplot as plt
import cv2


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()


def create_blank_image(width, height, color=(255, 255, 255)):
    return Image.new("RGB", (width, height), color)


def display_original_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((300, 400))
    photo = ImageTk.PhotoImage(image)
    original_label.config(image=photo)
    original_label.image = photo


def display_processed_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((300, 400))
    photo = ImageTk.PhotoImage(image)
    processed_label.config(image=photo)
    processed_label.image = photo


# vedio

def start_vedio_detection(canvas2):
    print("start---")
    # model files
    config_file = './configs/underwater/optics2.py'
    checkpoint_file = './demo/epoch_20.pth'

    output_file_root_path = 'demo/output/'
    output_file_name = 'result6.mp4'
    device = 'cuda:0'
    video_root_path = './video/'
    video_name = 'fish-1.mp4'
    score_thr = 0.5
    model = init_detector(config_file, checkpoint_file, device=device)

    camera = cv2.VideoCapture(video_root_path + video_name)
    # print(camera)
    width = 400
    height = 300
    camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_hight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

    # cv2.resizeWindow('Video',width,height)
    fps = camera.get(cv2.CAP_PROP_FPS)

    video_writer = cv2.VideoWriter(output_file_root_path + output_file_name, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (camera_width, camera_hight))

    count = 0

    # print('Press "Esc", "q" or "Q" to exit.')
    while True:
        torch.cuda.empty_cache()
        ret_val, img = camera.read()
        print(type(img))
        # plt.imshow(img)
        # plt.show()
        if ret_val:
            if count < 0:
                count += 1
                print("Write {} in result Successfuly!".format(count))
                continue

            result = inference_detector(model, img)
            print("*" * 50)
            # print(result)
            print("*" * 50)
            result_int = result[1][0:3]
            result_int = result_int.astype(int)
            print(result_int)
            left_top = (result_int[0][0], result_int[0][1])
            right_bottom = (result_int[0][2], result_int[0][3])

            cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 3)
            # cv2.imshow("img", img)
            # cv2.resizeWindow("img",300,300)
            # cv2.waitKey(0)
            # cv2.destroyWindow("img")

            # ch = cv2.waitKey(1)
            # if ch == 27 or ch == ord('q') or ch == ord('Q'):
            #     break

            frame = model.show_result(img, result, score_thr=score_thr, wait_time=1, show=False)
            # cv2.imshow('frame', frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #
            # # 在此处添加任何对帧的处理操作
            #
            # # 将帧转换为图像对象
            image = Image.fromarray(frame)
            image = image.resize((400, 300))
            #
            # # 在界面上显示图像
            photo = ImageTk.PhotoImage(image)
            canvas2.create_image(0, 0, anchor=tk.NW, image=photo)
            if len(frame) >= 1:
                video_writer.write(frame)
                count += 1
                print("Write {} in result Successfuly!".format(count))

        else:
            print('Load fail!')
            break

    camera.release()
    video_writer.release()
    return output_file_root_path + output_file_name


# def upload_image():
#     image_path = tk.filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
#     if image_path:
#         # image = Image.open(image_path)
#         # image = image.resize((300, 300))
#         # photo = ImageTk.PhotoImage(image)
#
#         # image_label.configure(image=photo)
#         # image_label.image = photo
#         display_original_image(image_path)


# def start_processing(img_path):
#     #
#     print("start---")
#     # model files
#     config_file = './configs/underwater/optics2.py'
#     checkpoint_file = './demo/epoch_20.pth'
#     out_putfile = 'demo/output'
#     device = 'cuda:0'
#
#     if not os.path.exists(out_putfile):
#         os.mkdir(out_putfile)
#     model = init_detector(config_file, checkpoint_file, device=device)
#     # img = 'demo/demo.jpg'
#     result = inference_detector(model, img_path)
#     r = show_result_pyplot(model, img_path, result)
#     class_name = model.CLASSES
#     print(class_name)
#
#     result_path = out_putfile + 'result.jpg'
#     model.show_result(img_path, result, out_file=result_path)
#
#     display_processed_image(result_path)


def import_video():
    video_path = tk.filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if video_path:
        print(video_path)
        #
        # display_path = start_vedio_detection()
        display_path = video_path
        # display_path = '/demo/output/result6.mp4'
        print("detection over----")
        print()
        if display_path:
            print(f'current path{display_path}')
            play_video(display_path, canvas1)


def import_video2():
    video_path = tk.filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if video_path:
        print(video_path)
        #
        # display_path = start_vedio_detection()
        display_path = video_path
        # display_path = '/demo/output/result6.mp4'
        print("detection over----")
        print()
        if display_path:
            print(f'current path{display_path}')
            play_video(display_path, canvas2)


def play_video(video_path, canvas):
    cap = cv2.VideoCapture(video_path)
    video_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 将帧从 BGR 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 在此处添加任何对帧的处理操作

        # 将帧转换为图像对象
        image = Image.fromarray(frame)
        image = image.resize((400, 300))
        video_frames.append(image)
        # 在界面上显示图像
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo
        # video_label.configure(image=photo)
        # video_label.image = photo
        delay_time = get_current_scale_value()
        # print(delay_time)
        time.sleep(float(delay_time) / 20)
        # 更新界面
        # window.after(20)
        window.update()
    final_frame = video_frames[-1]
    final_photo = ImageTk.PhotoImage(final_frame)
    canvas.create_image(0, 0, anchor=tk.NW, image=final_photo)
    canvas.image = final_photo
    # canvas.itemconfig(video_image,image=video_frame[-1])

    # 释放视频捕捉对象
    # cap.release()


def start_vedio_detection1():
    print("start---")
    # model files
    config_file = './configs/underwater/optics2.py'
    checkpoint_file = './demo/epoch_20.pth'

    output_file_root_path = 'demo/output/'
    output_file_name = 'result6.mp4'
    device = 'cuda:0'
    video_root_path = './video/'
    video_name = 'fish-1.mp4'
    score_thr = 0.5
    model = init_detector(config_file, checkpoint_file, device=device)

    camera = cv2.VideoCapture(video_root_path + video_name)
    # print(camera)
    width = 400
    height = 300
    camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_hight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

    # cv2.resizeWindow('Video',width,height)
    fps = camera.get(cv2.CAP_PROP_FPS)

    video_writer = cv2.VideoWriter(output_file_root_path + output_file_name, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (camera_width, camera_hight))

    count = 0

    # print('Press "Esc", "q" or "Q" to exit.')
    while True:
        torch.cuda.empty_cache()
        ret_val, img = camera.read()
        if ret_val:
            if count < 0:
                count += 1
                print("Write {} in result Successfuly!".format(count))
                continue
            result = inference_detector(model, img)
            print("*" * 50)
            # print(result)
            print("*" * 50)
            result_int = result[1][0:3]
            result_int = result_int.astype(int)
            print(result_int)
            left_top = (result_int[0][0], result_int[0][1])
            right_bottom = (result_int[0][2], result_int[0][3])

            cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 3)
            # cv2.imshow("img", img)
            # cv2.resizeWindow("img",300,300)
            # cv2.waitKey(0)
            # cv2.destroyWindow("img")

            # ch = cv2.waitKey(1)
            # if ch == 27 or ch == ord('q') or ch == ord('Q'):
            #     break

            frame = model.show_result(img, result, score_thr=score_thr, wait_time=1, show=False)
            # cv2.imshow('frame', frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #
            # # 在此处添加任何对帧的处理操作
            #
            # # 将帧转换为图像对象
            # image = Image.fromarray(frame)
            # image = image.resize((400, 300))
            #
            # # 在界面上显示图像
            # photo = ImageTk.PhotoImage(image)
            # canvas.create_image(0, 0, anchor=tk.NW, image=photo)

            if len(frame) >= 1:
                video_writer.write(frame)
                count += 1
                print("Write {} in result Successfuly!".format(count))

        else:
            print('Load fail!')
            break
    camera.release()
    video_writer.release()

    return output_file_root_path + output_file_name


def detection_import_video():
    video_path_ = start_vedio_detection(canvas2)
    print(video_path_)
    if video_path_:
        play_video(video_path_, canvas2)


def upload_image():
    global image_path_g
    image_path_g = tk.filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    print(image_path_g, type(image_path_g))
    if image_path_g:
        # image = Image.open(image_path)
        # image = image.resize((300, 300))
        # photo = ImageTk.PhotoImage(image)
        # image_label.configure(image=photo)
        # image_label.image = photo
        display_original_image(image_path_g)

    # print('image: ',image_path_g, type(image_path))


def start_processing():
    #
    global image_path_g

    print("start--- [image detection] ")
    # model files
    config_file = './configs/underwater/optics2.py'
    checkpoint_file = './demo/epoch_20.pth'
    out_putfile = 'demo/output'
    device = 'cuda:0'
    img_path = image_path_g
    if img_path is None:
        print("No image")
    else:
        if not os.path.exists(out_putfile):
            os.mkdir(out_putfile)
        model = init_detector(config_file, checkpoint_file, device=device)
        # img = 'demo/demo.jpg'
        result = inference_detector(model, img_path)
        print(type(result))
        print(result[0])
        print(type(result[0]))
        # for bbox ,score ,label in zip(result[0]['bbox'] ,result[0]['score'],result[0]['label']) :
        #     bbox = np.array(bbox)
        #     print('Bouding Box' ,bbox)
        #     print('Score :' ,score)
        #     print('Label  :',label)
        # r = show_result_pyplot(model, img_path, result)
        class_name = model.CLASSES
        print(class_name)

        result_path = out_putfile + 'result.jpg'
        model.show_result(img_path, result, out_file=result_path)

        display_processed_image(result_path)


def update_value(value):
    decimal_value = float(value) / 20
    value_label.config(text="V:{}".format(decimal_value))


def get_current_scale_value():
    return scale.get()


def button_upload_image():
    # global image_path_g
    img_ = upload_image()
    print("img :  ", img_)
    return img_


# import cv2
import threading
# import tkinter as tk
# from PIL import Image, ImageTk

# import cv2
import threading


# import tkinter as tk
# from PIL import Image, ImageTk

class VideoThread(threading.Thread):
    def __init__(self, video_path, canvas):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.canvas = canvas
        self.video_frames = []
        self.images_list = []
        self.is_running = False

    def run(self):
        self.is_running = True
        cap = cv2.VideoCapture(self.video_path)
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            # 按 q exit()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = image.resize((400, 300))
            self.images_list.append(image)
            self.video_frames.append(image)

            # 在界面上显示原始视频
            photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo

    def stop(self):
        self.is_running = False


class ObjectDetectionThread(threading.Thread):
    def __init__(self, video_thread, canvas):
        threading.Thread.__init__(self)
        self.video_thread = video_thread
        self.canvas = canvas
        self.model = None
        self.score_thr =0.5
        # 加载自定义模型
        config_file = './configs/underwater/optics2.py'
        checkpoint_file = './demo/epoch_20.pth'
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

    def run(self):
        # model = init_detector(config_file, checkpoint_file, device=device)
        while self.video_thread.is_running :

            if(len(self.video_thread.video_frames)) > 0 :
                frame = self.video_thread.video_frames[-1]
                image = self.video_thread.images_list[-1]
                print(type(image))
                image = np.array(image)

                result = inference_detector(self.model, image)

                # print("*" * 50)
                # print(result)
                # print("*" * 50)
                result_int = result[1][0:3]
                result_int = result_int.astype(int)
                # print(result_int)
                left_top = (result_int[0][0], result_int[0][1])
                right_bottom = (result_int[0][2], result_int[0][3])

                cv2.rectangle(image, left_top, right_bottom, (0, 0, 255), 3)

                frame = self.model.show_result(image, result, score_thr=self.score_thr, wait_time=1, show=False)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # # 在此处添加任何对帧的处理操作

                # # 将帧转换为图像对象
                image = Image.fromarray(frame)
                image = image.resize((400, 300))
                #
                # # 在界面上显示图像
                photo = ImageTk.PhotoImage(image)
                canvas2.create_image(0, 0, anchor=tk.NW, image=photo)


    ####

    # # 获取最新的视频帧
    # if len(self.video_thread.video_frames) > 0:
    #     frame = self.video_thread.video_frames[-1]
    #
    #     # 进行目标检测
    #     result = self.detect_objects(frame)
    #     print(result)
    #
    #     # 在界面上显示目标检测结果
    #     self.show_detection_result(result)
    #
    # # 模拟目标检测的延迟
    # time.sleep(0.1)

    def detect_objects(self, frame):
        # 将图像转换为模型可接受的格式
        img = mmcv.imfrombytes(frame.tobytes(), flag='color')

        # 使用模型进行目标检测
        result = inference_detector(self.model, img)

        return result

    def show_detection_result(self, result):
        # 获取目标框的坐标和类别信息
        bboxes = result[0]  # 假设目标框信息在result的第一个元素中
        labels = result[1]  # 假设类别信息在result的第二个元素中

        # 在画布上绘制目标框
        for bbox, label in zip(bboxes, labels):
            x, y, w, h = bbox
            cv2.rectangle(self.canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.canvas, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 更新画布显示
        self.canvas.update()


# # 创建主窗口
# window = tk.Tk()
# window.geometry("800x300")
#
# # 创建Canvas用于显示原始视频
# video_canvas = tk.Canvas(window, width=400, height=300)
# video_canvas.pack(side=tk.LEFT)
#
# # 创建Canvas用于显示目标检测结果
# detection_canvas = tk.Canvas(window, width=400, height=300)
# detection_canvas.pack(side=tk.LEFT)
#
# # 创建视频线程
# video_thread = VideoThread("path_to_video_file", video_canvas)
# video_thread.start()
#
# # 创建目标检测线程
# object_detection_thread = ObjectDetectionThread(video_thread, detection_canvas)
# object_detection_thread.start()

# 关闭窗口时停止线程
def on_closing():
    video_thread.stop()
    video_thread.join()
    object_detection_thread.join()
    # window.destroy()


# window.protocol("WM_DELETE_WINDOW", on_closing)
#
# window.mainloop()



def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title" ,type=str, default="水下目标检测", help="标题")
    parser.add_argument("--window_size", type=str, default="900x600", help="窗口大小")
    parser.add_argument("--config_file", type=str, default="./config/underwater/optics2.py", help="配置文件路径")
    parser.add_argument("--checkpoint_file", type=str, default="./demo/epoch_20.pth", help="")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")

    return parser.parse_args()

if __name__ == '__main__':


  # 创建主窗口
    window = tk.Tk()
    window.title("水下目标检测平台")
    window.geometry("900x600")

    # 创建选项卡
    notebook = tk.ttk.Notebook(window)
    notebook.pack(fill=tk.BOTH, expand=True)

    # 模块1：文本展示
    text_frame = tk.Frame(notebook, width=600, height=500, bg="white")
    text_label = tk.Label(text_frame, text="导入配置文件", font=("Arial", 16), bg="white")
    text_label.pack(pady=20)
    notebook.add(text_frame, text="导入配置文件")

    # 模块2：图片展示
    image_frame = tk.Frame(notebook, width=600, height=500, bg="white")
    image_frame.pack(padx=10, pady=10)

    upload_button = tk.Button(image_frame, text="上传图片", command=upload_image)
    upload_button.pack(pady=10)

    start_button = tk.Button(image_frame, text="开始", command=start_processing)
    start_button.pack(pady=10)

    # image_label = tk.Label(image_frame, bg="white")
    # image_label.pack(pady=20)
    # 原始图像显示标签
    original_image = create_blank_image(300, 400)
    original_photo = ImageTk.PhotoImage(original_image)
    original_label = tk.Label(image_frame, image=original_photo)
    original_label.pack(side=tk.LEFT, padx=10, pady=10)

    # 处理后的图像显示标签
    processed_image = create_blank_image(300, 400)
    processed_photo = ImageTk.PhotoImage(processed_image)
    processed_label = tk.Label(image_frame, image=processed_photo)
    processed_label.pack(side=tk.RIGHT, padx=10, pady=10)

    image_label = tk.Label(image_frame, bg='white')
    image_label.pack(pady=20)
    notebook.add(image_frame, text="水下图像检测")

    ###

    # 模块3：视频展示
    video_frame = tk.Frame(notebook, width=600, height=500, bg="black")
    video_frame.pack(padx=10, pady=10)

    # scrollbar  = tk.Scrollbar(video_frame)
    # scrollbar.pack(side=tk.RIGHT,fill=tk.Y)
    # video_canvas = tk.Canvas(video_frame,width=600,height=500,bg="black",yscrollcommand=scrollbar.set)
    # video_canvas.pack(side=tk.LEFT)
    # scrollbar.config(command=video_canvas.yview)

    import_button = tk.Button(video_frame, text="导入视频", command=import_video)
    import_button.pack(pady=10)

    # video_label = tk.Label(video_frame, bg="black")
    # video_label.pack()

    # create first canvas1
    canvas1 = tk.Canvas(video_frame, width=400, height=300)
    canvas1.pack(side=tk.LEFT)

    # initial_image = ImageTk.PhotoImage(video_frame[0])
    # video_image = canvas1.create_image(0,0,anchor=tk.NW,image=initial_image)

    # 2
    canvas2 = tk.Canvas(video_frame, width=400, height=300)
    canvas2.pack(side=tk.RIGHT)
    # # 创建视频线程
    video_thread = VideoThread("./video/fish-1.mp4", canvas1)
    video_thread.start()
    #
    # # 创建目标检测线程
    object_detection_thread = ObjectDetectionThread(video_thread, canvas2)
    object_detection_thread.start()

    scale = tk.Scale(video_frame, from_=0, to=20, orient=tk.HORIZONTAL, command=update_value)
    scale.pack()
    value_label = tk.Label(video_frame, text='V: 0')
    value_label.pack()
    import_button2 = tk.Button(video_frame, text="start detection", command=detection_import_video)
    import_button2.pack(pady=10)

    # video_label2 = tk.Label(video_frame,bg="white")
    # video_label2.pack()

    notebook.add(video_frame, text="水下视频检测")

    # 运行主循环
    window.mainloop()
    window.protocol("WM_DELETE_WINDOW", on_closing)
