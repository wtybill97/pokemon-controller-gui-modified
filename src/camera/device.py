# FIXED: 添加缺失的导入
import platform
import subprocess
import re
import locale          # FIXED: 处理编码问题
import imageio_ffmpeg
from typing import List, Optional, Tuple   # FIXED: 添加类型提示


class CameraDevice(object):
    def __init__(self, id, name, width, height, pixelFormat, min_fps, max_fps):
        self._id = id
        self._name = name
        self._width = width
        self._height = height
        self._pixelFormat = pixelFormat
        self._min_fps = min_fps
        self._max_fps = max_fps
        self._fps = max_fps

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def fps(self):
        return self._fps

    @property
    def min_fps(self):
        return self._min_fps

    @property
    def max_fps(self):
        return self._max_fps

    def setFps(self, fps: int):
        if fps < self._min_fps:
            self._fps = self._min_fps
        elif fps > self._max_fps:
            self._fps = self._max_fps
        else:
            self._fps = fps

    # 列出当前设备的所有摄像头
    @staticmethod
    def list_device() -> List['CameraDevice']:   # FIXED: 添加返回类型提示
        cameras = []
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        system = platform.system()

        try:
            if system == 'Darwin':
                # FIXED: macOS 使用 dummy 避免环境变量问题，并处理编码
                cmd = [ffmpeg_exe, '-f', 'avfoundation', '-list_devices', 'true', '-i', 'dummy']
                result = subprocess.run(cmd, stderr=subprocess.PIPE, text=False)  # 获取 bytes

                # FIXED: 多编码尝试解码 stderr
                for encoding in ('utf-8', locale.getpreferredencoding(), 'gbk', 'macroman'):
                    try:
                        lines = result.stderr.decode(encoding).splitlines()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    lines = result.stderr.decode('utf-8', errors='ignore').splitlines()

                is_video = False
                for line in lines:
                    if "AVFoundation video devices:" in line:
                        is_video = True
                        continue
                    if "AVFoundation audio devices:" in line:
                        break
                    if is_video:
                        # FIXED: 更健壮的正则匹配
                        match = re.search(r'\[(\d+)\]\s+(.+)$', line.strip())
                        if match:
                            dev_id = match.group(1)
                            name = match.group(2).strip()
                            # FIXED: 可扩展调用 ffprobe 获取实际格式，此处先使用默认值
                            cameras.append(CameraDevice(dev_id, name, 1920, 1080, None, 30, 60))

            elif system == 'Windows':
                # 步骤1：列举设备（使用 dummy）
                cmd = [ffmpeg_exe, '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy']
                result = subprocess.run(cmd, stderr=subprocess.PIPE, text=False)

                # FIXED: 多编码解码 stderr
                raw_stderr = result.stderr
                system_encoding = locale.getpreferredencoding()
                stderr_text = None
                for enc in (system_encoding, 'utf-8', 'gbk', 'cp1252'):
                    try:
                        stderr_text = raw_stderr.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if stderr_text is None:
                    stderr_text = raw_stderr.decode('utf-8', errors='ignore')

                lines = stderr_text.splitlines()
                device_names = []

                # FIXED: 正确解析 Windows 设备名（仅视频设备，去重）
                for line in lines:
                    if '(video)' in line:
                        match = re.search(r'"([^"]+)"', line)
                        if match:
                            name = match.group(1).strip()
                            if name and name not in device_names:
                                device_names.append(name)

                # FIXED: 对每个设备查询最佳格式（新增 _get_best_dshow_format）
                for idx, name in enumerate(device_names):
                    dev_id = f"video={name}"
                    best_format = CameraDevice._get_best_dshow_format(ffmpeg_exe, name)

                    if best_format:
                        width, height, min_fps, max_fps, pix_fmt = best_format
                        cameras.append(CameraDevice(dev_id, name, width, height, pix_fmt, min_fps, max_fps))
                        print(f"[CameraDevice] Added device: {name} with {width}x{height} @ {max_fps}fps")
                    else:
                        # 降级：使用默认值
                        print(f"[CameraDevice] No format info for {name}, using default 1920x1080@30")
                        cameras.append(CameraDevice(dev_id, name, 1920, 1080, None, 30, 60))
                        # 可选：添加索引形式的 ID 备用
                        alt_id = str(idx)
                        cameras.append(CameraDevice(alt_id, f"{name} (index)", 1920, 1080, None, 30, 60))

        except Exception as e:
            print(f"[CameraDevice] Error listing devices: {e}")
            import traceback
            traceback.print_exc()

        return cameras

    # FIXED: 新增方法 _get_best_dshow_format，用于查询设备支持的格式
    @staticmethod
    def _get_best_dshow_format(ffmpeg_exe: str, device_name: str) -> Optional[Tuple[int, int, int, int, str]]:
        """
        查询 dshow 设备支持的格式，返回最佳 (width, height, min_fps, max_fps, pixel_format)
        """
        try:
            cmd = [ffmpeg_exe, '-f', 'dshow', '-list_options', 'true', '-i', f'video={device_name}']
            result = subprocess.run(cmd, stderr=subprocess.PIPE, text=False, timeout=5)

            system_encoding = locale.getpreferredencoding()
            try:
                output = result.stderr.decode(system_encoding)
            except UnicodeDecodeError:
                output = result.stderr.decode('utf-8', errors='ignore')

            # 多种模式匹配分辨率、帧率
            patterns = [
                r'(\d+)x(\d+)[,\s]+fps:\s*(\d+(?:\.\d+)?)',
                r'(\d+)x(\d+)@(\d+(?:\.\d+)?)',
                r'(\d+)x(\d+)[,\s]+(\d+(?:\.\d+)?)\s*fps',
            ]
            candidates = []
            for line in output.splitlines():
                for pat in patterns:
                    match = re.search(pat, line, re.IGNORECASE)
                    if match:
                        w = int(match.group(1))
                        h = int(match.group(2))
                        fps = float(match.group(3))
                        # 应用过滤条件：宽度 720~1920，16:9 宽高比，帧率 29~61
                        if w < 720 or w > 1920:
                            continue
                        if abs(h - w * 9 / 16) > 1:
                            continue
                        if fps < 29 or fps > 61:
                            continue
                        candidates.append((w, h, fps))
                        break

            if not candidates:
                # 如果没有匹配到任何格式，尝试提取任何分辨率（放宽条件）
                fallback_pattern = r'(\d+)x(\d+)'
                for line in output.splitlines():
                    match = re.search(fallback_pattern, line)
                    if match:
                        w = int(match.group(1))
                        h = int(match.group(2))
                        if 640 <= w <= 1920:
                            candidates.append((w, h, 30.0))

            if not candidates:
                print(f"[CameraDevice] No valid format found for {device_name}")
                return None

            # 选择最高分辨率，同分辨率选最高帧率
            candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
            best_w, best_h, best_fps = candidates[0]
            max_fps = int(round(best_fps))
            min_fps = 30 if max_fps > 30 else max_fps
            return (best_w, best_h, min_fps, max_fps, None)

        except subprocess.TimeoutExpired:
            print(f"[CameraDevice] Timeout querying formats for {device_name}")
        except Exception as e:
            print(f"[CameraDevice] Error in _get_best_dshow_format: {e}")
        return None