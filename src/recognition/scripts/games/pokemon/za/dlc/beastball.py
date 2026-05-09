# beastball.py
# 放置在 recognition/scripts/games/pokemon/za/dlc/beastball.py
# 脚本逻辑：重启游戏 → 执行对应宝可梦的捕捉宏 → 检测特定区域文字，识别到指定字符串即成功并停止
# 参数 pokemon_type: "玛夏多" 或 "美录坦" 或 "波尔凯尼恩"（下拉菜单选择）
# 参数 ball_type: 球种名称（下拉菜单选择）
# 美录坦：路卡利欧近身战A秒杀后捕捉
# 玛夏多：命玉X鹿月爆A仙闪X月爆A大概率击杀后捕捉（取决于是否有暗影偷盗躲技能）
# 波尔凯尼恩：命玉X鹿强化十万伏特A*3大概率击杀后捕捉

from enum import Enum
import multiprocessing
import time
from recognition.scripts.parameter_struct import ScriptParameter
from recognition.scripts.base.base_script import BaseScript, WorkflowEnum
import cv2
import numpy as np
from recognition.ocr.rapidocr import RapidOCR
from datetime import datetime

# 球种名称到数值的映射
BALLS = {
    "究极球": 1,
    "狩猎球": 2,
    "竞赛球": 3,
    "梦境球": 4,
    "月亮球": 5,
    "甜蜜球": 6,
    "沉重球": 7,
    "等级球": 8,
    "诱饵球": 9,
    "友友球": 10,
    "速度球": 11,
}

class PokemonCatch(BaseScript):
    @staticmethod
    def script_name() -> str:
        return "宝可梦-ZA-DLC-幻兽球种"

    def __init__(self, stop_event: multiprocessing.Event, frame_queue: multiprocessing.Queue,
                 controller_input_action_queue: multiprocessing.Queue, paras: dict = None):
        # 使用固定脚本名初始化父类（不包含宝可梦名称）
        super().__init__(self.script_name(), stop_event,
                         frame_queue, controller_input_action_queue, PokemonCatch.script_paras())
        self._prepare_step_index = -1
        self._cycle_step_index = -1
        self._jump_next_frame = False

        # 默认参数（会在 set_paras 后生效）
        self._loop = self.get_para("loop")
        self._durations = self.get_para("durations")
        self._ns1 = self.get_para("ns1") if paras and "ns1" in paras else False

        # OCR 引擎初始化
        self.ocr_engine = RapidOCR(
            upscale=2.5,
            enable_preprocess=True,
        )

        # 待初始化的属性（将在 on_start 中通过 get_para 设置）
        self._pokemon_display_name = None
        self._pokemon_type = None
        self._target_substring = None
        self._ocr_region = None
        self._macro_name = None
        self._success_title = None
        self._obs_save_title = None
        self._meow_content_prefix = None
        self._ball_name = None      # 球种名称（用于日志）
        self._ball_value = 1     # 球种数值（传递给宏）

        self._success = False

        self.set_paras(paras)

    def _convert_display_to_internal(self, display_name: str) -> str:
        """将显示名（中文）转换为内部标识"""
        if display_name == "美录坦":
            return "meltan"
        elif display_name == "玛夏多":
            return "marshadow"
        elif display_name == "波尔凯尼恩":
            return "volcanion"
        else:
            # 默认返回 marshadow
            return "marshadow"


    @staticmethod
    def script_paras() -> dict:
        paras = dict()
        paras["loop"] = ScriptParameter("loop", int, -1, "运行次数（-1表示无限）")
        paras["durations"] = ScriptParameter("durations", float, -1, "运行时长（分钟）")
        paras["ns1"] = ScriptParameter("ns1", bool, True, "是否使用NS1")
        # 宝可梦类型下拉菜单
        paras["pokemon_type"] = ScriptParameter(
            "pokemon_type", str, "波尔凯尼恩", "宝可梦类型", ["玛夏多", "美录坦", "波尔凯尼恩"]
        )
        # 球种下拉菜单：选项为球种名称列表，参数类型为字符串，内部再转换为数值
        ball_names = list(BALLS.keys())
        paras["ball_type"] = ScriptParameter(
            "ball_type", str, "究极球", "选择球种", ball_names
        )
        return paras
    
    def _setup_by_pokemon(self):
        """根据宝可梦类型配置识别文字、区域、宏名称等"""
        if self._pokemon_type == "meltan":
            self._target_substring = "你替我捉住"
            self._ocr_region = (534, 941, 534, 60)
            self._macro_name = "recognition.pokemon.za.dlc.meltan.meltan"
            self._success_title = f"✅ 成功用{self._ball_name}捕捉{self._pokemon_display_name}！"
            self._obs_save_title = self._pokemon_display_name
            self._meow_content_prefix = "识别文字"
        elif self._pokemon_type == "marshadow":
            self._target_substring = "原来如此"
            self._ocr_region = (536, 876, 408, 56)
            self._macro_name = "recognition.pokemon.za.dlc.marshadow.marshadow"
            self._success_title = f"✅ 成功用{self._ball_name}捕捉{self._pokemon_display_name}！"
            self._obs_save_title = self._pokemon_display_name
            self._meow_content_prefix = "识别文字"
        elif self._pokemon_type == "volcanion":
            self._target_substring = "漂亮地捉住"
            self._ocr_region = (533, 942, 574, 56)
            self._macro_name = "recognition.pokemon.za.dlc.volcanion.volcanion"
            self._success_title = f"✅ 成功用{self._ball_name}捕捉{self._pokemon_display_name}！"
            self._obs_save_title = self._pokemon_display_name
            self._meow_content_prefix = "识别文字"

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
        # 获取宝可梦类型
        self._pokemon_display_name = self.get_para("pokemon_type")
        self._pokemon_type = self._convert_display_to_internal(self._pokemon_display_name)
        self._setup_by_pokemon()
        # 获取球种
        self._ball_name = self.get_para("ball_type")
        self._ball_value = BALLS.get(self._ball_name, 1)  # 默认究极球
        full_name = f"{self.script_name()}-{self._ball_name}-{self._pokemon_display_name}"
        self.send_log(f"开始运行 {full_name} 脚本")

    def on_cycle(self):
        run_time_span = self.run_time_span
        full_name = f"{self.script_name()}-{self._ball_name}-{self._pokemon_display_name}"
        log_txt = (f"[{full_name}] 脚本运行中，已运行 {self.cycle_times} 轮，耗时 "
                   f"{int(run_time_span/3600)}小时{int((run_time_span % 3600)/60)}分{int(run_time_span % 60)}秒")
        self.send_log(log_txt)

    def on_stop(self):
        run_time_span = self.run_time_span
        full_name = f"{self.script_name()}-{self._ball_name}-{self._pokemon_display_name}"
        self.send_log(f"[{full_name}] 脚本停止，"
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
        run_time_span = self.run_time_span
        self.macro_stop(block=True)
        full_name = f"{self.script_name()}-{self._ball_name}-{self._pokemon_display_name}"
        self.send_log(f"[{full_name}] 脚本完成，"
                      f"已运行 {self.cycle_times - 1} 轮，"
                      f"耗时 {int(run_time_span/3600)}小时{int((run_time_span % 3600)/60)}分{int(run_time_span % 60)}秒")
        self.stop_work()

    # ---------- 步骤实现 ----------
    def step_0(self):
        """重启游戏并执行对应宝可梦的捕捉宏"""
        self.macro_run("pokemon.za.common.restart_game",
                       loop=1,
                       paras={"ns1": str(self._ns1), "restore_backup": False},
                       block=True,
                       timeout=None)
        time.sleep(0.1)
        current_frame = self.current_frame
        if current_frame is not None:
            error_region = (380, 300, 130, 42)
            error_results = self.ocr_engine.batch_recognize_regions(current_frame, [error_region])
            error_text = ""
            if error_results and len(error_results) > 0:
                text_obj = error_results[0]
                if text_obj and isinstance(text_obj, dict):
                    error_text = text_obj.get('text', "") or ""
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

        # 调用捕捉宏，并传递球种参数
        macro_paras = {"number": self._ball_value}  # 假设宏需要的参数名为 "ball"
        self.macro_run(self._macro_name,
                       loop=1,
                       paras=macro_paras,
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

        x, y, w, h = self._ocr_region
        roi = current_frame[y:y+h, x:x+w]
        if roi.size == 0:
            self.send_log(f"裁剪区域无效: {self._ocr_region}")
            self._cycle_step_index += 1
            return

        results = self.ocr_engine.batch_recognize_regions(current_frame, [self._ocr_region])
        recognized_text = ""
        if results and len(results) > 0:
            text_obj = results[0]
            if text_obj and isinstance(text_obj, dict):
                recognized_text = text_obj.get('text', "") or ""
                recognized_text = recognized_text.strip()
        self.send_log(f"OCR 识别结果: {recognized_text}")

        if self._target_substring in recognized_text:
            self._success = True
            self.send_log(f"✅ 成功识别到目标文字！停止脚本")
            self.macro_run("recognition.pokemon.za.dlc.donut.capture", loop=1, block=True, timeout=None)
            success_content = {"识别文本": recognized_text}
            self.send_notification(
                title=self._success_title,
                feishu_content=success_content,
                meow_title=self._success_title,
                meow_content=f"{self._meow_content_prefix}：{recognized_text}"
            )
            self._trigger_obs_save(self._pokemon_type, title=self._obs_save_title, cycle_times=self.cycle_times)
            self._finished_process()
        else:
            self.send_log("未识别到目标文字，重新开始循环")
            self._cycle_step_index += 1

    # ---------- 辅助检查 ----------
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