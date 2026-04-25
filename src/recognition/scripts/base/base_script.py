from abc import ABC, abstractmethod
import threading
import multiprocessing
import sys
import time
from typing import final
import cv2
import numpy
from datatype.frame import Frame
from log import send_log
from const import ConstClass
from enum import Enum
import macro
import requests
import json
import os
import re
import shutil
from datetime import datetime
CONFIG_PATH = "feishu_config.json"
# OBS WebSocket 可选依赖，在方法内部导入，避免强制要求安装


class WorkflowEnum(Enum):
    Begin = 0
    Preparation = 1
    Cycle = 2
    AfterCycle = 3
    End = 4


class BaseScript(ABC):
    def __init__(self, script_name, stop_event: multiprocessing.Event, frame_queue: multiprocessing.Queue, controller_input_action_queue: multiprocessing.Queue, paras: dict = dict()):
        # 停止事件
        self._stop_event = stop_event

        # 帧队列
        self._frame_queue = frame_queue
        while not self._frame_queue.empty():
            self._frame_queue.get_nowait()
        # 控制器输入队列
        self._controller_input_action_queue = controller_input_action_queue
        self._script_name = script_name
        self._paras = paras if paras else dict()
        self.set_paras(paras)
        self._my_const = ConstClass()
        self._width = self._my_const.RecognizeVideoWidth
        self._height = self._my_const.RecognizeVideoHeight
        self._fps = self._my_const.RecognizeVideoFps

        # 宏进程
        self._macro_thread = None
        self._macro_stop_event = None
        # 当前帧
        self._current_frame = None
        # 运行状态
        self._running_status = None
        # 运行开始时间
        self._run_start_time_monotonic = 0
        # 准备期帧数
        self._preparation_frame_count = -1
        # 跳出循环后帧数
        self._after_cycle_frame_count = -1
        # 首次循环周期开始时间
        self._first_cycle_begin_time_monotonic = 0
        # 当前循环开始时间
        self._current_cycle_begin_time_monotonic = 0
        # 当前循环帧数
        self._current_cycle_frame_count = -1
        # 当前循环次数
        self._cycle_times = 0
        # 循环重新开始标志
        self._cycle_continue_flag = False
        # 循环跳出标志
        self._cycle_break_flag = False
        # 循环跳出时间
        self._cycle_break_time_monotonic = 0
        # 结束标志
        self._stop_flag = False
        # 上一帧获取时间
        self._last_set_frame_time_monotonic = 0

        self._load_feishu_config()
        self._feishu_token_info = {'access_token': None, 'expire_at': 0}

        

        # ---------- OBS 录制相关 ----------
        self._obs = None
        self._obs_connected = False
        self._obs_pending_action = None
        self._failure_folder = None            # 失败录像存放目录
        self._script_start_time = None         # 在 on_start 中赋值
        self._recipe_name = None               # 在 on_start 中赋值

        # ---------- 通用统计计数器（子类可修改）----------
        self._success_count = 0
        self._target_success_count = 0
        self._invalid_cycle_count = 0


    def _load_feishu_config(self):
        default_config = {
            'app_id': '',
            'app_secret': '',
            'chat_id': '',
            'enable_notification': False,
        }
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        self._feishu_app_config = default_config
    
    @abstractmethod
    def process_frame(self):
        pass

    @abstractmethod
    def on_start(self):
        pass

    @abstractmethod
    def on_stop(self):
        pass

    @abstractmethod
    def on_cycle(self):
        pass

    @abstractmethod
    def on_error(self):
        pass

    # 运行参数
    @final
    @property
    def paras(self) -> dict:
        return self._paras

    # 设置运行参数
    @final
    def set_paras(self, paras: dict):
        if paras is None:
            return
        for p in self._paras.values():
            if p.name in paras:
                self._paras[p.name].set_value(paras[p.name].value)
            else:
                self._paras[p.name].set_value(p.default_value)

    # 获取参数
    @final
    def get_para(self, name: str):
        value = None
        if name in self._paras:
            value = self._paras[name].value
        if value is None:
            value = self._paras[name].default_value
        if value is None:
            raise ValueError("parameter {} not found".format(name))
        return value

    # 当前帧
    @final
    @property
    def current_frame(self) -> cv2.Mat:
        np_array = numpy.frombuffer(
            self._current_frame.bytes(), dtype=numpy.uint8)
        mat = np_array.reshape(
            (self._current_frame.height, self._current_frame.width, self._current_frame.channels))
        if self._current_frame.height != self._height or self._current_frame.width != self._width:
            frame_mat = cv2.resize(
                mat, (self._width, self._height))
        else:
            frame_mat = mat
        return frame_mat

    @final
    @property
    def current_frame_960x540(self) -> cv2.Mat:
        np_array = numpy.frombuffer(
            self._current_frame.bytes(), dtype=numpy.uint8)
        mat = np_array.reshape(
            (self._current_frame.height, self._current_frame.width, self._current_frame.channels))
        if self._current_frame.height != 540 or self._current_frame.width != 960:
            recognize_mat = cv2.resize(
                mat, (960, 540), interpolation=cv2.INTER_AREA)
        else:
            recognize_mat = mat
        return recognize_mat

    # 运行状态
    @final
    @property
    def running_status(self):
        return self._running_status

    @final
    @property
    def current_frame_count(self):
        if self._running_status == WorkflowEnum.Preparation:
            return self._preparation_frame_count
        elif self._running_status == WorkflowEnum.Cycle:
            return self._current_cycle_frame_count
        elif self._running_status == WorkflowEnum.AfterCycle:
            return self._after_cycle_frame_count
        elif self._running_status == WorkflowEnum.End:
            return 0

    # 循环次数
    @final
    @property
    def cycle_times(self) -> float:
        return self._cycle_times

    # 当前循环持续时间
    @final
    @property
    def current_cycle_time_span(self) -> float:
        if self._running_status != WorkflowEnum.Cycle:
            return -1
        return time.monotonic() - self._current_cycle_begin_time_monotonic

    # 首次循环持续时间
    @final
    @property
    def first_cycle_time_span(self) -> float:
        if self._running_status != WorkflowEnum.Cycle:
            return -1
        return time.monotonic() - self._first_cycle_begin_time_monotonic

    # 运行持续时间
    @final
    @property
    def run_time_span(self) -> float:
        return time.monotonic() - self._run_start_time_monotonic

    @final
    def save_temp_image(self, rect=None):
        img = self.current_frame
        if rect is not None:
            img = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        time_str = time.strftime(
            "%Y%m%d%H%M%S", time.localtime())
        cv2.imwrite("./temp_"+time_str+".jpg", img)

    # 运行宏命令
    @final
    def macro_run(self, macro_name, loop=1, paras={}, block: bool = True, timeout: float = None):
        self.macro_stop()

        self._macro_stop_event = threading.Event()
        self._macro_thread = threading.Thread(
            target=macro.run, args=(macro_name, self._macro_stop_event, self._controller_input_action_queue, loop, paras, False))
        self._macro_thread.start()
        if not block:
            return True
        start_monotonic = time.monotonic()
        while self._macro_thread.is_alive():
            if self._stop_event is not None and self._stop_event.is_set():
                raise InterruptedError
            time.sleep(0.1)
            if timeout != None and timeout > 0 and time.monotonic() - start_monotonic > timeout:
                return self.macro_stop(timeout=None)
        self._macro_stop_event = None
        return True

    @final
    def macro_text_run(self, text, summary="", loop=1, paras={}, block: bool = True, timeout: float = None):
        self.macro_stop()
        self._macro_stop_event = threading.Event()
        self._macro_thread = threading.Thread(
            target=macro.run_text, args=(text, self._macro_stop_event, self._controller_input_action_queue, summary, loop, paras, False))
        self._macro_thread.start()
        if not block:
            return True
        start_monotonic = time.monotonic()
        while self._macro_thread.is_alive():
            if self._stop_event is not None and self._stop_event.is_set():
                raise InterruptedError
            time.sleep(0.1)
            if timeout != None and timeout > 0 and time.monotonic() - start_monotonic > timeout:
                return self.macro_stop(timeout=None)
        self._macro_stop_event = None
        return True

    # 宏命令线程是否正在运行
    @final
    @property
    def macro_running(self):
        if self._macro_thread == None:
            return False
        return self._macro_thread.is_alive()

    # 停止宏命令
    @final
    def macro_stop(self, block=True, timeout=None):
        if self._macro_thread != None:
            if self._macro_thread.is_alive():
                if block:
                    try:
                        self._macro_stop_event.set()
                        start_monotonic = time.monotonic()
                        while time.monotonic() - start_monotonic < timeout:
                            if self._stop_event is not None and self._stop_event.is_set():
                                raise InterruptedError
                            time.sleep(0.1)
                            if not self._macro_thread.is_alive():
                                self._macro_thread = None
                                return True
                    except InterruptedError:
                        raise
                    except:
                        pass
                    return False
                else:
                    self._macro_stop_event.set()
                    return False
            else:
                self._macro_thread = None
        return True

    # 发送日志
    @final
    def send_log(self, msg):
        send_log(msg)

    # 开始循环
    @final
    def set_cycle_begin(self):
        if self._first_cycle_begin_time_monotonic == 0:
            self._first_cycle_begin_time_monotonic = time.monotonic()

    # 继续循环
    @final
    def set_cycle_continue(self):
        self._cycle_continue_flag = True

    # 跳出循环
    @final
    def set_cycle_end(self):
        self._cycle_break_flag = True

    # 结束
    @final
    def stop_work(self):
        self._stop_flag = True

    # 运行
    @final
    def run(self):
        self._on_start()
        try:
            while True:
                if self._stop_flag:
                    return
                if self._stop_event.is_set():
                    raise InterruptedError
                frame = None
                while not self._frame_queue.empty():
                    frame = self._frame_queue.get()
                    if self._stop_flag:
                        return
                    if self._stop_event.is_set():
                        raise InterruptedError
                delay = 1.0/self._fps - \
                    (time.monotonic() - self._last_set_frame_time_monotonic)
                if frame is None or delay > 0:
                    if frame is None or delay > 0.001:
                        delay = 0.001
                    time.sleep(delay)
                    continue

                self._last_set_frame_time_monotonic = time.monotonic()
                self._current_frame = frame

                # 设置准备状态
                if self._running_status == WorkflowEnum.Begin:
                    self._running_status = WorkflowEnum.Preparation
                    self._preparation_frame_count = 0
                elif self._running_status == WorkflowEnum.Preparation:
                    self._preparation_frame_count += 1
                elif self._running_status == WorkflowEnum.Cycle:
                    self._current_cycle_frame_count += 1
                elif self._running_status == WorkflowEnum.AfterCycle:
                    self._after_cycle_frame_count += 1

                self.process_frame()

                if self._running_status == WorkflowEnum.Preparation and self._first_cycle_begin_time_monotonic > 0:
                    self._on_cycle()
                    self._first_cycle_begin_time_monotonic = self._current_cycle_begin_time_monotonic
                elif self._running_status == WorkflowEnum.Cycle and self._cycle_continue_flag:
                    self._cycle_continue_flag = False
                    self._on_cycle()
                elif self._running_status == WorkflowEnum.Cycle and self._cycle_break_flag:
                    self._on_cycle_break()

        except InterruptedError:
            return
        except Exception as e:
            self.on_error()
            raise e
        finally:
            self._stop_event = None
            self.macro_stop(timeout=0.5, block=True)
            self._on_stop()

    @final
    def _on_cycle(self):
        if self._running_status == WorkflowEnum.Cycle:
            self.on_cycle()
        self._running_status = WorkflowEnum.Cycle
        self._current_cycle_begin_time_monotonic = time.monotonic()
        self._current_cycle_frame_count = 0
        self._cycle_times += 1

    @final
    def _on_cycle_break(self):
        self._running_status = WorkflowEnum.AfterCycle
        self._cycle_break_time_monotonic = time.monotonic()
        self._after_cycle_frame_count = 0

    @final
    def _on_start(self):
        self._run_start_time_monotonic = time.monotonic()
        self._running_status = WorkflowEnum.Begin
        self.on_start()

    @final
    def _on_stop(self):
        self._running_status = WorkflowEnum.End
        self.on_stop()

    # 清空帧队列
    @final
    def clear_frame_queue(self):
        while not self._frame_queue.empty():
            self._frame_queue.get_nowait()

    # ==================== 飞书通知相关方法（通用） ====================
    def _get_feishu_tenant_access_token(self):
        """获取飞书 tenant_access_token，带缓存"""
        now = time.time()
        if self._feishu_token_info['access_token'] and self._feishu_token_info['expire_at'] > now:
            return self._feishu_token_info['access_token']

        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self._feishu_app_config.get('app_id'),
            "app_secret": self._feishu_app_config.get('app_secret')
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('code') == 0:
                    token = data['tenant_access_token']
                    expire = now + data['expire'] - 60  # 提前60秒刷新
                    self._feishu_token_info['access_token'] = token
                    self._feishu_token_info['expire_at'] = expire
                    self.send_log("获取飞书 tenant_access_token 成功")
                    return token
                else:
                    self.send_log(f"获取飞书 token 失败: {data}")
            else:
                self.send_log(f"获取飞书 token HTTP错误: {resp.status_code}")
        except Exception as e:
            self.send_log(f"获取飞书 token 异常: {e}")
        return None

    def _send_feishu_message(self, title, content_dict):
        """发送飞书卡片消息（使用应用API），自动从实例属性获取运行统计数据"""
        if not self._feishu_app_config.get('enable_notification', True):
            return False
        chat_id = self._feishu_app_config.get('chat_id')
        if not chat_id:
            self.send_log("飞书 chat_id 未配置，无法发送消息")
            return False

        token = self._get_feishu_tenant_access_token()
        if not token:
            self.send_log("无法获取飞书 access token，发送失败")
            return False

        # 从实例获取运行统计
        cycle_times = getattr(self, 'cycle_times', 0)
        #invalid_count = getattr(self, '_invalid_cycle_count', 0)
        success_count = getattr(self, '_success_count', 0)
        #target_count = getattr(self, '_target_success_count', 0)
        run_time_span = getattr(self, 'run_time_span', 0)
        run_time_str = f"{int(run_time_span/3600)}小时{int((run_time_span % 3600)/60)}分{int(run_time_span % 60)}秒"

        # 优先使用传入的 content_dict 中的值，否则使用实例值
        timestamp = content_dict.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        run_time_display = content_dict.get('run_time', run_time_str)
        cycle_times_display = content_dict.get('cycle_times', cycle_times)
        #invalid_count_display = content_dict.get('invalid_count', invalid_count)
        success_count_display = content_dict.get('success_count', success_count)
        #target_count_display = content_dict.get('target_count', target_count)

        # 构建卡片内容
        card_content = {
            "config": {
                "wide_screen_mode": True,
                "enable_forward": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": "blue"
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**时间**：{timestamp}\n"
                                   f"**运行时长**：{run_time_display}\n"
                                   #f"**失败次数**：{invalid_count_display} / {cycle_times_display}\n"
                                   #f"**成功次数**：{success_count_display} / {target_count_display}"
                    }
                }
            ]
        }

        # 添加详细信息：自动显示 content_dict 中的所有键值对（排除已显示在顶部的字段）
        extra_lines = []
        exclude_keys = {'timestamp', 'run_time'}  # 这些已在顶部显示，不再重复
        for key, value in content_dict.items():
            if key in exclude_keys:
                continue
            # 格式化：键名加粗，值直接显示
            extra_lines.append(f"**{key}**：{value}")
        if extra_lines:
            details_text = "\n".join(extra_lines)
            card_content["elements"].append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": details_text
                }
            })

        # 发送请求
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        params = {
            "receive_id_type": "open_id"
        }
        payload = {
            "receive_id": chat_id,
            "msg_type": "interactive",
            "content": json.dumps(card_content)
        }
        try:
            resp = requests.post(url, headers=headers, params=params, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('code') == 0:
                    self.send_log(f"飞书卡片消息发送成功: {title}")
                    return True
                else:
                    self.send_log(f"飞书卡片消息发送失败: {data}")
            else:
                self.send_log(f"飞书卡片消息发送HTTP错误: {resp.status_code}")
        except Exception as e:
            self.send_log(f"飞书卡片消息发送异常: {e}")
        return False

    def _send_feishu_webhook(self, msg_type, title, content_dict):
        """统一飞书通知接口，实际调用应用消息接口"""
        # 不再使用 webhook，直接使用应用消息接口
        return self._send_feishu_message(title, content_dict)

    def _upload_feishu_image(self, image_path: str) -> str:
        """上传图片到飞书，返回 image_key，失败返回 None"""
        token = self._get_feishu_tenant_access_token()
        if not token:
            self.send_log("无法获取飞书 access token，上传图片失败")
            return None

        url = "https://open.feishu.cn/open-apis/im/v1/images"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f, 'image/png')}
                data = {'image_type': 'message'}
                resp = requests.post(url, headers=headers, files=files, data=data, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                if result.get('code') == 0:
                    image_key = result.get('data', {}).get('image_key')
                    self.send_log(f"飞书图片上传成功，image_key: {image_key}")
                    return image_key
                else:
                    self.send_log(f"飞书图片上传失败: {result}")
            else:
                self.send_log(f"飞书图片上传HTTP错误: {resp.status_code}")
        except Exception as e:
            self.send_log(f"飞书图片上传异常: {e}")
        return None

    def _send_feishu_card_with_image(self, title: str, image_key: str, content_dict: dict = None):
        """发送带图片的飞书卡片消息"""
        token = self._get_feishu_tenant_access_token()
        if not token:
            self.send_log("无法获取飞书 access token，发送图片消息失败")
            return False

        chat_id = self._feishu_app_config.get('chat_id')
        if not chat_id:
            self.send_log("飞书 chat_id 未配置，无法发送消息")
            return False

        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        params = {"receive_id_type": "open_id"}

        # 构建卡片内容
        card_content = {
            "config": {"wide_screen_mode": True, "enable_forward": True},
            "header": {"title": {"tag": "plain_text", "content": title}, "template": "blue"},
            "elements": [
                {"tag": "img", "img_key": image_key},
                {"tag": "div",
                 "text": {"tag": "lark_md", "content": f"**时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}}
            ]
        }
        # 添加额外信息
        if content_dict:
            details = "\n".join([f"**{k}**：{v}" for k, v in content_dict.items()])
            card_content["elements"].append({"tag": "div", "text": {"tag": "lark_md", "content": details}})

        payload = {
            "receive_id": chat_id,
            "msg_type": "interactive",
            "content": json.dumps(card_content)
        }
        try:
            resp = requests.post(url, headers=headers, params=params, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('code') == 0:
                    self.send_log(f"飞书带图片卡片消息发送成功: {title}")
                    return True
                else:
                    self.send_log(f"飞书带图片卡片消息发送失败: {data}")
            else:
                self.send_log(f"飞书带图片卡片消息发送HTTP错误: {resp.status_code}")
        except Exception as e:
            self.send_log(f"飞书带图片卡片消息发送异常: {e}")
        return False

    # ==================== OBS 录制相关方法（通用） ====================
    def _obs_connect(self):
        """连接 OBS WebSocket 并注册事件，所有异常被捕获"""
        if self._obs is not None:
            return
        try:
            from obswebsocket import obsws, events
            from obswebsocket import requests as obs_requests
            # IPv6 地址直接使用，无需特殊处理
            self._obs = obsws("127.0.0.1", 4455, "gpobrc8Rue20x09q")
            self._obs.connect()
            self._obs_connected = True
            self._obs.register(self._on_replay_saved, events.ReplayBufferSaved)
            self.send_log("OBS WebSocket 连接成功")
        except Exception as e:
            self.send_log(f"OBS WebSocket 连接失败: {e}")
            self._obs = None
            self._obs_connected = False

    def _on_replay_saved(self, event):
        """回放缓冲区保存完成时的回调，处理文件移动/重命名"""
        try:
            saved_path = event.datain.get('savedReplayPath')
            if not saved_path or not self._obs_pending_action:
                return
            action = self._obs_pending_action
            if action['type'] == 'failure':
                if self._failure_folder is None:
                    return
                dest_path = os.path.join(self._failure_folder, os.path.basename(saved_path))
                shutil.move(saved_path, dest_path)
                self.send_log(f"失败录像已保存至: {dest_path}")
            elif action['type'] == 'success':
                # 重命名文件：拼接 OCR 文本（清理非法字符）
                safe_parts = []
                for text in [action.get('ocr1', ''), action.get('ocr2', ''), action.get('ocr3', '')]:
                    clean = re.sub(r'[\\/*?:"<>|\n\r]', '_', text.strip())
                    safe_parts.append(clean)
                base_name = "_".join(safe_parts) + ".mp4"
                dest_dir = os.path.dirname(saved_path)
                dest_path = os.path.join(dest_dir, base_name)
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(base_name)
                    dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                    counter += 1
                shutil.move(saved_path, dest_path)
                self.send_log(f"成功录像已重命名为: {dest_path}")
        except Exception as e:
            self.send_log(f"处理录像文件时出错: {e}")
        finally:
            self._obs_pending_action = None

    def _trigger_obs_save(self, action_type, **kwargs):
        """触发 OBS 保存回放缓冲区，并记录上下文，所有异常被捕获"""
        try:
            if not self._obs_connected:
                self._obs_connect()
            if not self._obs_connected:
                self.send_log("无法连接 OBS，跳过保存")
                return

            # 如果是失败类型且文件夹未创建，则创建
            if action_type == 'failure' and self._failure_folder is None:
                if self._recipe_name is None or self._script_start_time is None:
                    self.send_log("缺少 recipe_name 或 script_start_time，无法创建失败录像文件夹")
                    return
                recipe_safe = self._recipe_name.replace('/', '-')
                folder_name = f"{self._script_start_time}_{recipe_safe}"
                base_dir = os.path.join(r'G:\Captures', "obs_recordings")
                self._failure_folder = os.path.join(base_dir, folder_name)
                os.makedirs(self._failure_folder, exist_ok=True)
                self.send_log(f"失败录像将保存至: {self._failure_folder}")

            self._obs_pending_action = {'type': action_type, **kwargs}
            try:
                from obswebsocket import requests as obs_requests
                self._obs.call(obs_requests.SaveReplayBuffer())
                self.send_log("已触发 OBS 保存回放")
            except Exception as e:
                self.send_log(f"触发 OBS 保存失败: {e}")
                self._obs_pending_action = None
        except Exception as e:
            self.send_log(f"OBS 操作整体异常: {e}")
            self._obs_pending_action = None