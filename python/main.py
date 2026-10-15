import tkinter as tk
from tkinter import ttk
import serial
import time
import cv2
import math
import os
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
from datetime import datetime

# --- ПОДКЛЮЧЕНИЕ НЕЙРОСЕТИ YOLO11s ---
MODEL_PATH = "best.pt"
model = None
try:
    from ultralytics import YOLO

    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"[AI] Модель {MODEL_PATH} успешно загружена!")
    else:
        print(f"[AI] Файл {MODEL_PATH} не найден!")
except Exception as e:
    print(f"[AI] Ошибка загрузки ultralytics: {e}")

# --- НАСТРОЙКИ ---
COM_PORT = "COM3"
BAUD_RATE = 115200
CAMERA_ID = 1

# --- ОБРЕЗКА КАДРА (ROI) ---
CROP_TOP = 0.0
CROP_BOTTOM = 0.0
CROP_LEFT = 0.0
CROP_RIGHT = 0.0

# --- НАСТРОЙКИ ЛИНИИ И ТРЕКЕРА ---
LINE_Y_RATIO = 0.5
SAME_OBJECT_RADIUS = 0.12
MEMORY_MOVING_SECONDS = 3.0

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
ser = None
cap = None
latest_frame = None
frame_lock = threading.Lock()
is_camera_running = True

is_conveyor_moving = False
last_command = "Остановлен (c:0)"

conveyor_active_time = 0.0
last_loop_time = time.time()

tracked_objects = {}
next_unique_id = 1

# --- СТАТИСТИКА ---
stats = {
    "total": 0,
    "piston": 0,
    "connecting_rod": 0,
    "valve": 0
}

CLASS_MAP = {
    "gvozd": "valve",
    "gvozd_sv": "valve",
    "key": "connecting_rod",
    "key_sb": "connecting_rod",
    "porshen": "piston",
    "porshen_sb": "piston",
    "porshen_sn": "piston"
}

DISPLAY_NAMES = {
    "valve": "Клапан",
    "connecting_rod": "Шатун",
    "piston": "Поршень"
}

try:
    pil_font_main = ImageFont.truetype("arialbd.ttf", 18)
    pil_font_sub = ImageFont.truetype("arial.ttf", 15)
except:
    pil_font_main = ImageFont.load_default()
    pil_font_sub = ImageFont.load_default()

def init_serial():
    global ser
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print(f"Порт {COM_PORT} открыт!")
    except Exception as e:
        print(f"Ошибка порта: {e}")

def send_command(cmd):
    global ser
    if ser and ser.is_open:
        full_cmd = f"{cmd}#"
        ser.write(full_cmd.encode('utf-8'))
        print(f"Отправлено: {full_cmd}")

def update_counter_ui():
    lbl_count_total.config(text=str(stats["total"]))
    lbl_count_piston.config(text=str(stats["piston"]))
    lbl_count_rod.config(text=str(stats["connecting_rod"]))
    lbl_count_valve.config(text=str(stats["valve"]))

def save_report_to_file():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"conveyor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    report_content = f"""========================================
       ОТЧЕТ РАБОТЫ УМНОГО КОНВЕЙЕРА
========================================
Дата и время формирования: {current_time}
Последняя команда управления: {last_command}
----------------------------------------
СТАТИСТИКА ОБЪЕКТОВ:
- Поршень:         {stats['piston']} шт.
- Шатун:           {stats['connecting_rod']} шт.
- Клапан:          {stats['valve']} шт.
----------------------------------------
ОБЩИЙ ИТОГ (Всего деталей): {stats['total']} шт.
========================================
Статус: Успешно завершено / Сохранено автоматически
"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\n[УСПЕХ] Отчет автоматически сохранен в файл: {filename}")
    except Exception as e:
        print(f"\n[ОШИБКА СОХРАНЕНИЯ ОТЧЕТА]: {e}")

# --- УПРАВЛЕНИЕ НАПРАВЛЕНИЕМ ЛЕНТЫ ---
def start_conveyor():
    global last_command, is_conveyor_moving
    send_command("c:2")  # Команда движения вперед
    is_conveyor_moving = True
    last_command = "Пуск Вперед (c:2)"
    status_indicator.config(bg="#2ecc71")
    status_text.config(text="Статус: Едет ВПЕРЕД", fg="#2ecc71")

def reverse_conveyor():
    global last_command, is_conveyor_moving
    send_command("c:1")  # Команда реверса
    is_conveyor_moving = True
    last_command = "Реверс / Назад (c:1)"
    status_indicator.config(bg="#f39c12")
    status_text.config(text="Статус: РЕВЕРС (НАЗАД)", fg="#f39c12")

def stop_conveyor():
    global last_command, is_conveyor_moving
    send_command("c:0")  # Команда стоп
    is_conveyor_moving = False
    last_command = "Стоп / Остановлен (c:0)"
    status_indicator.config(bg="#e74c3c")
    status_text.config(text="Статус: ОСТАНОВЛЕН", fg="#e74c3c")

def crop_frame(frame):
    h, w, _ = frame.shape
    top = int(h * CROP_TOP)
    bottom = int(h * (1.0 - CROP_BOTTOM))
    left = int(w * CROP_LEFT)
    right = int(w * (1.0 - CROP_RIGHT))
    return frame[top:bottom, left:right]

def update_custom_tracker(raw_detections, frame_w, frame_h):
    global next_unique_id, tracked_objects

    line_y = int(frame_h * LINE_Y_RATIO)
    max_dist = frame_h * SAME_OBJECT_RADIUS

    to_delete = []
    for obj_id, obj in tracked_objects.items():
        if (conveyor_active_time - obj['last_active_time']) > MEMORY_MOVING_SECONDS:
            to_delete.append(obj_id)
    for obj_id in to_delete:
        del tracked_objects[obj_id]

    filtered_dets = []
    for (item_type, x1, y1, x2, y2, conf) in sorted(raw_detections, key=lambda x: x[5], reverse=True):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        is_duplicate = False
        for (f_type, fx1, fy1, fx2, fy2, _) in filtered_dets:
            if f_type == item_type:
                fcx = (fx1 + fx2) / 2.0
                fcy = (fy1 + fy2) / 2.0
                if math.hypot(cx - fcx, cy - fcy) < (frame_h * 0.12):
                    is_duplicate = True
                    break
        if not is_duplicate:
            filtered_dets.append((item_type, x1, y1, x2, y2, conf))

    used_obj_ids = set()
    final_boxes = []

    for (item_type, x1, y1, x2, y2, conf) in filtered_dets:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        best_id = None
        best_dist = max_dist

        for obj_id, obj in tracked_objects.items():
            if obj_id in used_obj_ids:
                continue
            if obj['type'] == item_type:
                dist = math.hypot(cx - obj['cx'], cy - obj['cy'])
                if dist < best_dist:
                    best_dist = dist
                    best_id = obj_id

        if best_id is not None:
            prev_cy = tracked_objects[best_id]['cy']
            tracked_objects[best_id]['cx'] = cx
            tracked_objects[best_id]['cy'] = cy
            tracked_objects[best_id]['last_active_time'] = conveyor_active_time
            used_obj_ids.add(best_id)
            curr_id = best_id
        else:
            prev_cy = cy
            curr_id = next_unique_id
            next_unique_id += 1
            tracked_objects[curr_id] = {
                'type': item_type,
                'cx': cx,
                'cy': cy,
                'last_active_time': conveyor_active_time,
                'counted': False
            }
            used_obj_ids.add(curr_id)

        obj_data = tracked_objects[curr_id]

        if is_conveyor_moving and not obj_data['counted']:
            crossed = (prev_cy < line_y <= cy) or (prev_cy > line_y >= cy)
            if crossed:
                obj_data['counted'] = True
                stats[item_type] += 1
                stats["total"] += 1
                root.after(0, update_counter_ui)
                print(f"[+1] Засчитан: {DISPLAY_NAMES[item_type]} (Объект #{curr_id})")

        final_boxes.append((curr_id, item_type, x1, y1, x2, y2, obj_data['counted']))

    return final_boxes

def process_and_draw(rgb_frame, tracked_boxes):
    h, w, _ = rgb_frame.shape
    line_y = int(h * LINE_Y_RATIO)

    line_color = (46, 204, 113) if is_conveyor_moving else (150, 150, 150)
    cv2.line(rgb_frame, (0, line_y), (w, line_y), line_color, 2)

    overlay = rgb_frame.copy()
    cv2.rectangle(overlay, (10, 10), (220, 125), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, rgb_frame, 0.35, 0, rgb_frame)

    pil_img = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_img)

    draw.text((20, 16), f"Всего: {stats['total']}", font=pil_font_main, fill=(243, 156, 18))
    draw.text((20, 44), f"Поршень: {stats['piston']}", font=pil_font_sub, fill=(52, 152, 219))
    draw.text((20, 68), f"Шатун: {stats['connecting_rod']}", font=pil_font_sub, fill=(46, 204, 113))
    draw.text((20, 92), f"Клапан: {stats['valve']}", font=pil_font_sub, fill=(185, 119, 212))

    for (obj_id, item_type, x1, y1, x2, y2, is_counted) in tracked_boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        box_color = (46, 204, 113) if is_counted else (243, 156, 18)
        rus_label = f"{DISPLAY_NAMES[item_type]}"

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 50, 50))

        text_y = max(0, int(y1) - 24)
        draw.rectangle([x1, text_y, x1 + 110, text_y + 22], fill=box_color)
        draw.text((x1 + 6, text_y + 2), rus_label, font=pil_font_sub, fill=(0, 0, 0))

    return pil_img

def camera_capture_thread():
    global cap, latest_frame, is_camera_running, conveyor_active_time, last_loop_time
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    last_loop_time = time.time()

    while is_camera_running:
        now = time.time()
        dt = now - last_loop_time
        last_loop_time = now

        if is_conveyor_moving:
            conveyor_active_time += dt

        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = crop_frame(frame)
                h_orig, w_orig, _ = frame.shape
                raw_detections = []

                if model is not None:
                    results = model.predict(frame, conf=0.60, agnostic_nms=True, verbose=False)[0]
                    if results.boxes is not None and len(results.boxes) > 0:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        clss = results.boxes.cls.int().cpu().tolist()
                        confs = results.boxes.conf.cpu().tolist()

                        for box, cls_id, conf in zip(boxes, clss, confs):
                            raw_name = str(model.names[cls_id]).strip().lower()
                            item_type = CLASS_MAP.get(raw_name, None)
                            if item_type:
                                raw_detections.append((item_type, box[0], box[1], box[2], box[3], conf))

                tracked_boxes = update_custom_tracker(raw_detections, w_orig, h_orig)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with frame_lock:
                    latest_frame = (rgb_frame, tracked_boxes)
            else:
                time.sleep(0.01)
        else:
            time.sleep(0.5)

def update_gui_frame():
    if not is_camera_running:
        return

    data = None
    with frame_lock:
        if latest_frame is not None:
            data = (latest_frame[0].copy(), list(latest_frame[1]))

    if data is not None:
        rgb_frame, tracked_boxes = data
        orig_h, orig_w, _ = rgb_frame.shape

        w = cam_label.winfo_width()
        h = cam_label.winfo_height()
        if w > 10 and h > 10:
            scale_x = w / orig_w
            scale_y = h / orig_h
            rgb_frame = cv2.resize(rgb_frame, (w, h))
            scaled_boxes = [
                (oid, itype, x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, cnt)
                for (oid, itype, x1, y1, x2, y2, cnt) in tracked_boxes
            ]
        else:
            scaled_boxes = tracked_boxes

        pil_img = process_and_draw(rgb_frame, scaled_boxes)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        cam_label.imgtk = imgtk
        cam_label.configure(image=imgtk)

    root.after(20, update_gui_frame)

def on_closing():
    global is_camera_running
    is_camera_running = False
    send_command("c:0")
    if ser and ser.is_open:
        ser.close()
    save_report_to_file()
    time.sleep(0.2)
    if cap and cap.isOpened():
        cap.release()
    root.destroy()

# --- ИНТЕРФЕЙС ---
if __name__ == '__main__':
    init_serial()

    root = tk.Tk()
    root.title("Control Panel — Smart Conveyor ID 12 (1st Place Edition)")
    root.geometry("980x620")
    root.minsize(720, 500)
    root.config(bg="#1e1e1e")

    ui_font = ("Segoe UI", 11)
    bold_font = ("Segoe UI", 11, "bold")
    title_font = ("Segoe UI", 14, "bold")

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=3)
    root.columnconfigure(1, weight=1)

    left_frame = tk.Frame(root, bg="#1e1e1e")
    left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    left_frame.rowconfigure(0, weight=1)
    left_frame.columnconfigure(0, weight=1)

    cam_label = tk.Label(left_frame, bg="#000000", bd=0, highlightthickness=0)
    cam_label.grid(row=0, column=0, sticky="nsew")

    right_container = tk.Frame(root, bg="#1e1e1e")
    right_container.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
    right_container.rowconfigure(2, weight=1)
    right_container.columnconfigure(0, weight=1)

    tk.Label(right_container, text="УПРАВЛЕНИЕ", font=title_font, bg="#1e1e1e", fg="#ffffff").grid(row=0, column=0, pady=(5, 10), sticky="w")

    # Панель кнопок (Пуск, Реверс, Стоп)
    control_frame = tk.LabelFrame(right_container, text="Лента", font=ui_font, bg="#252526", fg="#ffffff", relief=tk.FLAT, padx=10, pady=10)
    control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
    control_frame.columnconfigure(0, weight=1)
    control_frame.columnconfigure(1, weight=1)

    status_frame = tk.Frame(control_frame, bg="#252526")
    status_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")
    status_indicator = tk.Label(status_frame, bg="#e74c3c", width=2, height=1)
    status_indicator.pack(side=tk.LEFT, padx=(0, 8))
    status_text = tk.Label(status_frame, text="Остановлен", font=ui_font, bg="#252526", fg="#e74c3c")
    status_text.pack(side=tk.LEFT)

    # Кнопка ПУСК ВПЕРЕД
    btn_start = tk.Button(control_frame, text="▶ ПУСК", font=bold_font, bg="#27ae60", fg="white", activebackground="#2ecc71", relief=tk.FLAT, command=start_conveyor, cursor="hand2")
    btn_start.grid(row=1, column=0, padx=(0, 5), sticky="ew", ipady=5)

    # Кнопка РЕВЕРС (НАЗАД)
    btn_reverse = tk.Button(control_frame, text="⟲ РЕВЕРС", font=bold_font, bg="#d35400", fg="white", activebackground="#e67e22", relief=tk.FLAT, command=reverse_conveyor, cursor="hand2")
    btn_reverse.grid(row=1, column=1, padx=(5, 0), sticky="ew", ipady=5)

    # Кнопка СТОП (на всю ширину снизу)
    btn_stop = tk.Button(control_frame, text="⬛ СТОП", font=bold_font, bg="#c0392b", fg="white", activebackground="#e74c3c", relief=tk.FLAT, command=stop_conveyor, cursor="hand2")
    btn_stop.grid(row=2, column=0, columnspan=2, pady=(8, 0), sticky="ew", ipady=5)

    stats_frame = tk.LabelFrame(right_container, text="Статистика объектов", font=ui_font, bg="#252526", fg="#ffffff", relief=tk.FLAT, padx=12, pady=12)
    stats_frame.grid(row=2, column=0, sticky="nsew")
    stats_frame.columnconfigure(1, weight=1)

    tk.Label(stats_frame, text="ВСЕГО ОБЪЕКТОВ:", font=bold_font, bg="#252526", fg="#f39c12").grid(row=0, column=0, sticky="w", pady=5)
    lbl_count_total = tk.Label(stats_frame, text="0", font=("Segoe UI", 16, "bold"), bg="#252526", fg="#f39c12")
    lbl_count_total.grid(row=0, column=1, sticky="e")

    ttk.Separator(stats_frame, orient='horizontal').grid(row=1, column=0, columnspan=2, sticky="ew", pady=8)

    def create_stat_row(parent, row, label_text, color):
        tk.Label(parent, text=label_text, font=ui_font, bg="#252526", fg=color).grid(row=row, column=0, sticky="w", pady=4)
        lbl_obj = tk.Label(parent, text="0", font=("Segoe UI", 13, "bold"), bg="#252526", fg="#ffffff")
        lbl_obj.grid(row=row, column=1, sticky="e", pady=4)
        return lbl_obj

    lbl_count_piston = create_stat_row(stats_frame, 2, "• Поршень", "#3498db")
    lbl_count_rod = create_stat_row(stats_frame, 3, "• Шатун", "#2ecc71")
    lbl_count_valve = create_stat_row(stats_frame, 4, "• Клапан", "#9b59b6")

    btn_save = tk.Button(stats_frame, text="Сохранить отчет сейчас", font=("Segoe UI", 9, "bold"), bg="#2980b9", fg="white", relief=tk.FLAT, command=save_report_to_file, cursor="hand2")
    btn_save.grid(row=5, column=0, columnspan=2, pady=(20, 0), sticky="ew", ipady=4)

    cam_thread = threading.Thread(target=camera_capture_thread, daemon=True)
    cam_thread.start()
    root.after(20, update_gui_frame)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
