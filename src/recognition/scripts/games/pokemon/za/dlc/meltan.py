# meltan.py
# 放置在 recognition/scripts/games/pokemon/za/dlc/meltan.py
# 脚本逻辑：重启游戏 → 执行美录坦捕捉宏 → 检测特定区域文字，识别到"你替我捉住"即成功并停止

from enum import Enum
import multiprocessing
import time
from recognition.scripts.parameter_struct import ScriptParameter
from recognition.scripts.base.base_script import BaseScript, WorkflowEnum
import cv2
import numpy as np
from recognition.ocr.rapidocr import RapidOCR
from datetime import datetime

class Meltan(BaseScript):
    def __init__(self, stop_event: multiprocessing.Event, frame_queue: multiprocessing.Queue,
                 controller_input_action_queue: multiprocessing.Queue, paras: dict = None):
        super().__init__(Meltan.script_name(), stop_event,
                         frame_queue, controller_input_action_queue, Meltan.script_paras())
        self._prepare_step_index = -1
        self._cycle_step_index = -1
        self._jump_next_frame = False

        # 默认参数
        self._loop = self.get_para("loop")
        self._durations = self.get_para("durations")
        self._ns1 = self.get_para("ns1") if paras and "ns1" in paras else False

        # OCR 引擎初始化
        self.ocr_engine = RapidOCR(
            upscale=2.5,
            enable_preprocess=True,
        )

        # 目标文字（子串匹配）
        self._target_substring = "你替我捉住"

        # 成功标志（成功一次即停止）
        self._success = False

        self.set_paras(paras)

    @staticmethod
    def script_name() -> str:
        return "宝可梦-ZA-DLC-美录坦"

    @staticmethod
    def script_paras() -> dict:
        paras = dict()
        paras["loop"] = ScriptParameter(
            "loop", int, -1, "运行次数（-1表示无限）")
        paras["durations"] = ScriptParameter(
            "durations", float, -1, "运行时长（分钟）")
        paras["ns1"] = ScriptParameter(
            "ns1", bool, True, "是否使用NS1")
        return paras

    # ---------- 主流程 ----------
    def process_frame(self):
        if self._check_durations():
            return
        if self._check_cycles():
            return

        if self.running_status == WorkflowEnum.Preparation:
            if self._prepare_step_index >= 0:
                if self._prepare_step_index >= len(self._prepare_step_list):
                    self.set_cycle_begin()
                    self._cycle_step_index = 0
                    return
                self._prepare_step_list[self._prepare_step_index]()
            return

        if self.running_status == WorkflowEnum.Cycle:
            if self.current_frame_count == 1:
                self._cycle_init()
            if self._jump_next_frame:
                self.clear_frame_queue()
                self._jump_next_frame = False
                return
            if self._cycle_step_index >= 0 and self._cycle_step_index < len(self._cycle_step_list):
                self._cycle_step_list[self._cycle_step_index]()
            else:
                self.macro_stop()
                self.set_cycle_continue()
                self._cycle_step_index = 0
            return

        if self.running_status == WorkflowEnum.AfterCycle:
            self.stop_work()
            return

    def on_start(self):
        self._prepare_step_index = 0
        self._script_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.send_log(f"开始运行 {Meltan.script_name()} 脚本")

    def on_cycle(self):
        run_time_span = self.run_time_span
        log_txt = (f"[{Meltan.script_name()}] 脚本运行中，已运行 {self.cycle_times} 轮，耗时 "
                   f"{int(run_time_span/3600)}小时{int((run_time_span % 3600)/60)}分{int(run_time_span % 60)}秒")
        self.send_log(log_txt)

    def on_stop(self):
        run_time_span = self.run_time_span
        self.send_log(f"[{Meltan.script_name()}] 脚本停止，"
                      f"总运行时间：{int(run_time_span/3600)}小时{int((run_time_span % 3600)/60)}分{int(run_time_span % 60)}秒")

    def on_error(self):
        pass

    @property
    def _prepare_step_list(self):
        return [self.prepare_step_0]

    def prepare_step_0(self):
        self._prepare_step_index += 1

    @property
    def _cycle_step_list(self):
        return [self.step_0, self.step_1]

    def _cycle_init(self):
        pass

    def _finished_process(self):
        """成功捕获后停止脚本"""
        run_time_span = self.run_time_span
        self.macro_stop(block=True)
        self.macro_run("common.switch_sleep",
                       loop=1, paras={"ns1": str(self._ns1)}, block=True, timeout=10)
        self.send_log(f"[{Meltan.script_name()}] 脚本完成，"
                      f"已运行 {self.cycle_times - 1} 轮，"
                      f"耗时 {int(run_time_span/3600)}h {int((run_time_span % 3600)/60)}m {int(run_time_span % 60)}s")
        self.stop_work()

    # ---------- 步骤实现 ----------
    def step_0(self):
        """1. 重启游戏（不恢复备份）  2. 执行美录坦捕捉宏"""
        self.macro_run("pokemon.za.common.restart_game",
                       loop=1,
                       paras={"ns1": str(self._ns1), "restore_backup": False},
                       block=True,
                       timeout=None)
        time.sleep(0.1)
        current_frame = self.current_frame
        if current_frame is not None:
            error_region = (380, 300, 130, 42)   # x, y, w, h
            error_results = self.ocr_engine.batch_recognize_regions(current_frame, [error_region])
            error_text = ""
            if error_results and len(error_results) > 0:
                text_obj = error_results[0]
                if text_obj and isinstance(text_obj, dict):
                    error_text = text_obj.get('text', "")
                    if error_text is None:
                        error_text = ""
                    error_text = error_text.strip()
            if error_text and "发生错误" in error_text:
                self.send_log(f"❌ 检测到错误文本：{error_text}，脚本停止")
                error_content = {"reason": "switch发生错误，已黑屏", "ocr_text": error_text}
                self.send_notification(
                title='⚠️ 脚本异常停止',
                feishu_content=error_content,
                meow_title="⚠️ 脚本异常停止",
                meow_content=f"switch发生错误，已黑屏\n识别文本：{error_text}"
                )
                self._finished_process()
                return

        self.macro_run("recognition.pokemon.za.dlc.meltan.meltan",
                       loop=1,
                       block=True,
                       timeout=None)

        self._jump_next_frame = True
        self._cycle_step_index += 1

    def step_1(self):
        """OCR 检测指定区域，若包含目标文字则成功并停止"""
        current_frame = self.current_frame
        if current_frame is None:
            self.send_log("当前帧为空，等待下一帧")
            self._cycle_step_index += 1
            return

        region = (534, 941, 534, 60)
        x, y, w, h = region
        roi = current_frame[y:y+h, x:x+w]
        if roi.size == 0:
            self.send_log(f"裁剪区域无效: {region}")
            self._cycle_step_index += 1
            return

        # 使用 batch_recognize_regions 识别该区域
        results = self.ocr_engine.batch_recognize_regions(current_frame, [region])
        recognized_text = ""
        if results and len(results) > 0:
            text_obj = results[0]
            if text_obj and isinstance(text_obj, dict):
                recognized_text = text_obj.get('text', "")
                if recognized_text is None:
                    recognized_text = ""
                recognized_text = recognized_text.strip()
        self.send_log(f"OCR 识别结果: {recognized_text}")

        # 若识别到目标文字
        if self._target_substring in recognized_text:
            self._success = True
            self.send_log(f"✅ 成功识别到目标文字！停止脚本")
            self.macro_run("recognition.pokemon.za.dlc.donut.capture", loop=1, block=True, timeout=None)
            # 发送通知
            success_content = {
                "recognized_text": recognized_text,
                "region": region,
                "recipe": "美录坦捕捉"
            }
            self.send_notification(
                title="✅ 美录坦捕捉成功",
                feishu_content=success_content,
                meow_title="🎉 美录坦捕捉成功",
                meow_content=f"识别文字：{recognized_text}"
            )

            self._finished_process()
        else:
            self.send_log("未识别到目标文字，重新开始循环")
            self._cycle_step_index += 1  # 进入下一轮循环

    # ---------- 辅助检查方法 ----------
    def _check_durations(self):
        if self._durations <= 0:
            return False
        if self.run_time_span >= self._durations * 60:
            self.send_log("运行时间已到达设定值，脚本停止")
            stop_content = {"reason": "运行时间到达设定值", "run_time_seconds": self.run_time_span}
            self.send_notification(
                title=f"🕒 运行时间到达 {self._durations} 分钟",
                feishu_content=stop_content,
                meow_title=f"🕒 运行时间到达 {self._durations} 分钟",
                meow_content=f"运行时间到达 {self._durations} 分钟"
            )
            self._finished_process()
            return True
        return False

    def _check_cycles(self):
        if self._loop <= 0:
            return False
        if self.cycle_times > self._loop:
            self.send_log("运行次数已到达设定值，脚本停止")
            stop_content = {"reason": "运行次数到达设定值", "total_cycles": self.cycle_times - 1}
            self.send_notification(
                title=f"🔢 运行次数到达 {self._loop} 次",
                feishu_content=stop_content,
                meow_title="运行次数到达设定值",
                meow_content=f"运行次数到达 {self._loop} 次"
            )
            self._finished_process()
            return True
        return False