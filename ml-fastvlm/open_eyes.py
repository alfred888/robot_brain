from flask import Flask, render_template, Response, jsonify
import cv2
import subprocess
import os
from datetime import datetime
import threading
import queue
import time
import sys

app = Flask(__name__)

# FastVLM 模型路径
MODEL_PATH = "/Users/user/workspace/models/llava-fastvithd_0.5b_stage3"
PROMPT = "用简短的语言描述图片内容"

# 全局变量
camera = None
description_queue = queue.Queue()
latest_description = "等待拍摄第一张照片..."
capture_lock = threading.Lock()  # 添加线程锁
is_model_loading = True  # 添加模型加载状态标志

# 创建图片保存目录
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

def print_progress(message):
    """打印带时间戳的进度信息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        # 设置摄像头参数
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        camera.set(cv2.CAP_PROP_EXPOSURE, -4)
        camera.set(cv2.CAP_PROP_GAIN, 100)
        camera.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    return camera

def generate_frames():
    while True:
        camera = get_camera()
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)  # 控制帧率

def describe_image(image_path):
    """调用 FastVLM 模型描述图像内容"""
    global is_model_loading
    
    cmd = [
        "python", "predict.py",
        "--model-path", MODEL_PATH,
        "--image-file", image_path,
        "--prompt", PROMPT
    ]
    
    print_progress(f"🔍 正在描述图片: {image_path}")
    print_progress(f"📝 使用的提示词: {PROMPT}")
    
    if is_model_loading:
        print_progress("⏳ 首次运行，模型正在加载中...")
        is_model_loading = False
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    description = result.stdout.strip()
    error = result.stderr.strip()
    
    if error:
        print_progress(f"❌ 错误信息:")
        print("=" * 50)
        print(error)
        print("=" * 50)
        return f"处理出错: {error}"
    
    if not description:
        print_progress("⚠️ 警告: 模型没有返回任何描述")
        return "模型没有返回任何描述"
    
    print_progress("✨ 描述结果:")
    print("=" * 50)
    print(description)
    print("=" * 50)
    print()
    
    return description

def capture_and_describe():
    while True:
        with capture_lock:  # 使用线程锁确保同一时间只有一个线程在处理
            try:
                camera = get_camera()
                ret, frame = camera.read()
                if not ret:
                    print_progress("❌ 无法读取摄像头画面")
                    time.sleep(1)
                    continue

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_image_{timestamp}.jpg"
                filepath = os.path.join(IMAGE_DIR, filename)
                
                # 保存图片
                cv2.imwrite(filepath, frame)
                print_progress(f"📸 图片已保存到: {filepath}")
                
                # 确保文件存在
                if not os.path.exists(filepath):
                    print_progress(f"❌ 图片文件保存失败: {filepath}")
                    continue
                    
                # 描述图片
                description = describe_image(filepath)
                description_queue.put(description)
                print_progress("✅ 图片描述已添加到队列")
                
                # 删除图片
                try:
                    os.remove(filepath)
                    print_progress(f"🗑️ 临时文件已删除: {filepath}")
                except Exception as e:
                    print_progress(f"⚠️ 删除临时文件失败: {e}")
                
            except Exception as e:
                print_progress(f"❌ 处理过程出错: {e}")
        
        time.sleep(5)  # 每5秒拍摄一次

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_description')
def get_description():
    try:
        description = description_queue.get_nowait()
        return jsonify({"description": description})
    except queue.Empty:
        return jsonify({"description": latest_description})

if __name__ == '__main__':
    print_progress("🚀 启动摄像头服务...")
    print_progress("📡 等待浏览器连接...")
    
    # 启动描述线程
    describe_thread = threading.Thread(target=capture_and_describe, daemon=True)
    describe_thread.start()
    
    # 启动 Flask 应用
    app.run(host='0.0.0.0', port=5000, debug=False)  # 关闭调试模式 