from enum import Enum
import multiprocessing
import time
from recognition.scripts.parameter_struct import ScriptParameter
from recognition.scripts.base.base_script import BaseScript, WorkflowEnum
import cv2
import numpy as np
from recognition.ocr.rapidocr import RapidOCR
import requests
from datetime import datetime
import json
import os
import re
import shutil

# 合并后的配方字典（副本所有配方 + 原始文件中独有的“闪耀力(捕获) - 混合”）
ZaDlcDonutRecipes = {
    # 副本中的详细配方
    "闪耀力 - 彩虹 - 1.59%/0.242%": [(5, 6), (4, 1), (5, 1),],
    "闪耀力 - 混合 - 2.13%/0.118%": [(5, 6), (3, 2)],
    "闪耀力 - 扁樱果 - 6.2%": [(8, 8)],
    "道具力 - 混合 - 1.17%": [(1, 6), (5, 2)],
    "道具力 - 佛柑果 - 3.1%": [(6, 8)],
    "树果3多多1 - 1.30%": [(1, 2), (2, 1),(1,1), (33, 4)],
    "树果3多多1 - 霹霹果 - 1.30%": [(1, 1), (2, 3), (34, 4)],
    "树果3多多1 - 刺耳果 - 1.30%": [(1, 1), (3, 3), (33, 4)],
    "树果3多多1 - 佛柑果 - 1.30%": [(6, 3), (31, 5)],
    "树果3多多1 - 福禄果 - 1.30%": [(1, 3), (2, 1),(6,1),(28,3)],
    "树果3多多1 - 草蚕果 - 1.30%": [(1, 2), (3, 2), (3, 1),(30,3)],
    "树果3多多2 - 霹霹果 - 0.74%": [(1, 2), (2, 3), (34, 3)],
    "树果3多多2 - 刺耳果 - 0.74%": [(1, 2), (3, 3), (33, 3)],
    "树果3多多2 - 棱瓜果 - 1.4%": [(1, 5), (9, 1), (49, 1)],
    "树果3多多2 - 2棱瓜果 - 2.16%": [(1, 2), (5, 4), (4, 2)],
    "树果3多多2 - 2灯浆果 - 2.16%": [(1, 2), (1, 2), (4, 4)],
    "树果3多多2 - 2草蚕果 - 2.16%": [(1, 3), (5, 3), (1, 2)],
    "树果3多多2 - 2福禄果 - 2.16%": [(1, 5), (5, 1), (3, 2)],
    "树果3多多2 - 草蚕果 - 1.4%": [(1, 2), (2, 4), (4, 1)],
    "树果3多多2 - 灯浆果 - 1.4%": [(1, 5), (1, 1), (44, 1)],
    "树果3多多2 - 洛玫果 - 0.74%": [(1, 4), (36, 4)],
    "树果1多多1 - 刺耳果 - 5.56%": [(37, 8)],
    "捕获力": [(9, 8)]
}

ZaDlcDonutPowerLevels1 = [0, 1, 2, 3]
ZaDlcDonutPowerLevels2 = [-1, 0, 1, 2, 3]


class ZaDlcDonutPowerType(Enum):
    Sparkling = "闪耀力"
    Catching = "捕获力"
    Alpha = "头目力"
    Humungo = "大大力"
    Teensy = "小小力"
    Item = "道具力"
    BigHaul = "多多力"


class ZaDlcDonutItemType(Enum):
    Berries = "树果"
    Balls = "球"
    Specials = "特别"


class ZaDlcTypeType(Enum):
    All = "全属性"
    Normal = "一般"
    Flying = "飞行"
    Fire = "火"
    Psychic = "超能力"
    Water = "水"
    Bug = "虫"
    Electric = "电"
    Rock = "岩石"
    Grass = "草"
    Ghost = "幽灵"
    Ice = "冰"
    Dragon = "龙"
    Fighting = "格斗"
    Dark = "恶"
    Poison = "毒"
    Steel = "钢"
    Ground = "地面"
    Fairy = "妖精"


class ZaDlcTypeType2(Enum):
    All = "所有属性"
    Normal = "一般"
    Flying = "飞行"
    Fire = "火"
    Psychic = "超能力"
    Water = "水"
    Bug = "虫"
    Electric = "电"
    Rock = "岩石"
    Grass = "草"
    Ghost = "幽灵"
    Ice = "冰"
    Dragon = "龙"
    Fighting = "格斗"
    Dark = "恶"
    Poison = "毒"
    Steel = "钢"
    Ground = "地面"
    Fairy = "妖精"


class ZaDlcDonut(BaseScript):
    def __init__(self, stop_event: multiprocessing.Event, frame_queue: multiprocessing.Queue, controller_input_action_queue: multiprocessing.Queue, paras: dict = None):
        super().__init__(ZaDlcDonut.script_name(), stop_event,
                         frame_queue, controller_input_action_queue, ZaDlcDonut.script_paras())
        self._prepare_step_index = -1
        self._cycle_step_index = -1
        self._jump_next_frame = False

        

        # OBS 相关变量由基类初始化，无需重复

        # 成功计数器（基类已有 _success_count 和 _target_success_count）
        self._target_success_count = 20          # 默认20次，可通过参数覆盖
        self._target_reached = False

        # 昼夜切换相关
        self._no_valid_data_count = 0
        self._invalid_cycle_count = 0
        self._max_no_valid_data_count = 3
        self._need_daynight = False
        self._consecutive_daynight_count = 0
        self._max_consecutive_daynight = 2

        self.set_paras(paras)

        # 获取脚本参数（覆盖默认值）
        self._loop = self.get_para("loop")
        self._durations = self.get_para("durations")
        self._ns1 = self.get_para("ns1") if paras and "ns1" in paras else False
        self._target_success_count = self.get_para("target_success_count") if paras and "target_success_count" in paras else 20

        self._recipe = [(8, 8), (0, 0), (0, 0), (0, 0)]
        if paras and "Recipe" in paras:
            recipe_items = ZaDlcDonutRecipes.get(self.get_para("Recipe"))
            if recipe_items:
                for idx, item in enumerate(recipe_items[:len(self._recipe)]):
                    self._recipe[idx] = item

        self._sparkling_power_level = self.get_para("SparklingPowerLevel") if paras and "SparklingPowerLevel" in paras else 0
        self._sparkling_power_type_list = self.get_para("SparklingPowerType") if paras and "SparklingPowerType" in paras else [e.value for e in ZaDlcTypeType]
        self._catching_power_level = self.get_para("CatchingPowerLevel") if paras and "CatchingPowerLevel" in paras else 0
        self._alpha_power_level = self.get_para("AlphaPowerLevel") if paras and "AlphaPowerLevel" in paras else 0
        self._humungo_power_level = self.get_para("HumungoPowerLevel") if paras and "HumungoPowerLevel" in paras else 0
        self._teensy_power_level = self.get_para("TeensyPowerLevel") if paras and "TeensyPowerLevel" in paras else 0
        self._item_power_level = self.get_para("ItemPowerLevel") if paras and "ItemPowerLevel" in paras else 0
        self._item_power_type_list = self.get_para("ItemPowerType") if paras and "ItemPowerType" in paras else [e.value for e in ZaDlcDonutItemType]
        self._big_haul_power_level = self.get_para("BigHaulPowerLevel") if paras and "BigHaulPowerLevel" in paras else 0

        # 检查特殊条件（副本中的逻辑）
        self._special_sparkling_condition = False
        if self._sparkling_power_type_list and isinstance(self._sparkling_power_type_list, list):
            if "全属性" in self._sparkling_power_type_list and len(self._sparkling_power_type_list) <= 3:
                self._special_sparkling_condition = True
                self.send_log("启用特殊条件：闪耀力属性列表包含全属性且不超过3个属性")
                self.send_log(f"允许的属性列表：{self._sparkling_power_type_list}")

        self.ocr_engine = RapidOCR(
            upscale=2.5,
            enable_preprocess=True,
        )

    @staticmethod
    def script_name() -> str:
        return "宝可梦-ZA-DLC-刷三明治"

    @staticmethod
    def script_paras() -> dict:
        paras = dict()
        paras["target_success_count"] = ScriptParameter(
            'target_success_count', int, 20, "甜甜圈个数")
        paras["loop"] = ScriptParameter(
            "loop", int, -1, "运行次数")
        paras["durations"] = ScriptParameter(
            "durations", float, -1, "运行时长（分钟）")
        paras["ns1"] = ScriptParameter(
            "ns1", bool, True, "是否使用NS1")
        recipes = list(ZaDlcDonutRecipes.keys())
        paras["Recipe"] = ScriptParameter(
            "Recipe", str, recipes[0], "使用三明治配方", recipes)
        paras["SparklingPowerLevel"] = ScriptParameter(
            "SparklingPowerLevel", int, 3, "闪耀力等级", ZaDlcDonutPowerLevels1)
        paras["SparklingPowerType"] = ScriptParameter(
            "SparklingPowerType", list, [e.value for e in ZaDlcTypeType], "闪耀力属性", [e.value for e in ZaDlcTypeType])
        paras["CatchingPowerLevel"] = ScriptParameter(
            "CatchingPowerLevel", int, 0, "捕获力等级(属性与制作完成甜甜圈闪耀力属性相同)", ZaDlcDonutPowerLevels1)
        paras["AlphaPowerLevel"] = ScriptParameter(
            "AlphaPowerLevel", int, ZaDlcDonutPowerLevels2[0], "头目力等级(-1时排除这个条件)", ZaDlcDonutPowerLevels2)
        paras["HumungoPowerLevel"] = ScriptParameter(
            "HumungoPowerLevel", int, ZaDlcDonutPowerLevels2[0], "(或)大大力等级(-1时排除这个条件)", ZaDlcDonutPowerLevels2)
        paras["TeensyPowerLevel"] = ScriptParameter(
            "TeensyPowerLevel", int, ZaDlcDonutPowerLevels2[0], "(或)小小力等级(-1时排除这个条件)", ZaDlcDonutPowerLevels2)
        paras["ItemPowerLevel"] = ScriptParameter(
            "ItemPowerLevel", int, 0, "道具力等级", ZaDlcDonutPowerLevels1)
        paras["ItemPowerType"] = ScriptParameter(
            "ItemPowerType", list, ['树果'], "道具力类型", [e.value for e in ZaDlcDonutItemType])
        paras["BigHaulPowerLevel"] = ScriptParameter(
            "BigHaulPowerLevel", int, 0, "多多力等级", ZaDlcDonutPowerLevels1)
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
        self._check_pokemon_index = -1
        self._script_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._recipe_name = self.get_para("Recipe")
        self.send_log(f"开始运行{ZaDlcDonut.script_name()}脚本，目标成功次数：{self._target_success_count}")

    def on_cycle(self):
        run_time_span = self.run_time_span
        log_txt = f"[{ZaDlcDonut.script_name()}] 脚本运行中，耗时{int(run_time_span/3600)}小时{int((run_time_span % 3600)/60)}分{int(run_time_span % 60)}秒，当前成功次数：{self._success_count}/{self._target_success_count}"
        self.send_log(log_txt)
        self._check_pokemon_index = -1

    def on_stop(self):
        run_time_span = self.run_time_span
        stop_reason = "正常停止"
        if self._consecutive_daynight_count >= self._max_consecutive_daynight:
            stop_reason = "因连续识别失败停止"
        self.send_log("[{}] {}，实际运行{}次，成功{}次，耗时{}小时{}分{}秒".format(
            ZaDlcDonut.script_name(), stop_reason, self.cycle_times, self._success_count,
            int(run_time_span/3600), int((run_time_span % 3600)/60), int(run_time_span % 60)))

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

    def _finished_process(self):
        run_time_span = self.run_time_span
        self.macro_stop(block=True)
        self.macro_run("common.switch_sleep",
                       loop=1, paras={"ns1": str(self._ns1)}, block=True, timeout=10)
        self.send_log("[{}] 脚本完成，已运行{}次，成功{}次，耗时{}小时{}分{}秒".format(
            ZaDlcDonut.script_name(), self.cycle_times - 1, self._success_count,
            int(run_time_span/3600), int((run_time_span % 3600)/60), int(run_time_span % 60)))
        self.stop_work()

    def _re_cycle(self):
        pass

    def _cycle_init(self):
        pass

    def step_0(self):
        self.macro_run("pokemon.za.common.restart_game",
                       loop=1, paras={"ns1": str(self._ns1), "restore_backup": True}, block=True, timeout=None)

        # 昼夜切换处理（副本逻辑）
        if self._need_daynight:
            if self._consecutive_daynight_count >= self._max_consecutive_daynight:
                self.send_log(f"连续{self._max_consecutive_daynight}轮都未识别到有效数据，停止工作")
                stats = f"失败次数：{self._invalid_cycle_count}/{self.cycle_times}\n成功次数：{self._success_count}/{self._target_success_count}"
                error_content = {
                    "reason": "昼夜切换后连续识别失败",
                    "details":stats,
                    "consecutive_failures": self._consecutive_daynight_count
                }
                self._send_feishu_webhook('script_error', '⚠️ 脚本异常停止', error_content)
                self.stop_work()
                return
            self.send_log("执行昼夜切换操作...")
            self.macro_run("recognition.pokemon.za.dlc.donut.daynight", loop=1, block=True, timeout=None)
            daynight_content = {"reason": "昼夜切换"}
            self._send_feishu_webhook('daynight', '⚠️ 昼夜切换', daynight_content)
            self._need_daynight = False

        paras = {
            "berry1_position": self._recipe[0][0],
            "berry1_count": self._recipe[0][1],
            "berry2_position": self._recipe[1][0],
            "berry2_count": self._recipe[1][1],
            "berry3_position": self._recipe[2][0],
            "berry3_count": self._recipe[2][1],
            "berry4_position": self._recipe[3][0],
            "berry4_count": self._recipe[3][1],
        }
        self.macro_run("recognition.pokemon.za.dlc.donut.donut",
                       loop=1, paras=paras, block=True, timeout=None)
        self._jump_next_frame = True
        self._cycle_step_index += 1

    def step_1(self):
        current_frame = self.current_frame
        text1, text2, text3, text4 = self._ocr_power_text(current_frame)
        power1 = self._split_ocr_power_text(text1)
        power2 = self._split_ocr_power_text(text2)
        power3 = self._split_ocr_power_text(text3)

        p1, s1, l1 = power1
        p2, s2, l2 = power2
        p3, s3, l3 = power3

        self._log_special_condition_info()
        if text4 and '发生错误' in text4:
            self.send_log(f"switch黑屏，停止工作")
            error_content = {"reason": "switch发生错误，已黑屏"}
            self._send_feishu_webhook('script_error', '⚠️ 脚本异常停止', error_content)
            self.stop_work()
            return

        # 检查三行都未识别到有效数据（昼夜切换逻辑）
        if not text1 and not text2 and not text3:
            self._no_valid_data_count += 1
            self._invalid_cycle_count += 1
            self.send_log(f"三行都未识别到有效数据！连续次数：{self._no_valid_data_count}/{self._max_no_valid_data_count}")

            # 保存失败时的截图
            try:
                captures_dir = "./Captures"
                if not os.path.exists(captures_dir):
                    os.makedirs(captures_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{captures_dir}/failure_{timestamp}.jpg"
                cv2.imwrite(filename, self.current_frame)
                self.send_log(f"失败截图已保存至: {filename}")
            except Exception as e:
                self.send_log(f"保存失败截图时出错: {e}")

            self._trigger_obs_save('failure', timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"), recipe=self.get_para("Recipe"))

            if self._no_valid_data_count >= self._max_no_valid_data_count:
                self._no_valid_data_count = 0
                self._need_daynight = True
                self._consecutive_daynight_count += 1
                self.send_log(f"连续{self._max_no_valid_data_count}次未识别到有效数据，将在下次循环执行昼夜切换")
                self.send_log(f"连续需要昼夜切换的轮数：{self._consecutive_daynight_count}/{self._max_consecutive_daynight}")
        else:
            self._no_valid_data_count = 0
            if self._consecutive_daynight_count > 0:
                self.send_log("成功识别到数据，重置连续昼夜切换计数")
                self._consecutive_daynight_count = 0

        self.send_log(f"失败轮次 {self._invalid_cycle_count}/{self.cycle_times}")

        # 输出识别结果
        for i, (power, subPower, lv) in enumerate([power1, power2, power3], 1):
            if power is not None:
                sub_text = f": {subPower.value}" if subPower is not None else ""
                self.send_log(f"行{i}：{power.value}{sub_text} Lv.{lv}")
            else:
                self.send_log(f"行{i}：未匹配到有效数据")
        if any([text1, text2, text3]):
            self.send_log("---------原始文本---------")
            if text1:
                self.send_log(text1)
            if text2:
                self.send_log(text2)
            if text3:
                self.send_log(text3)
            self.send_log("--------------------------")

        # 检测闪耀力并获取其属性
        sparkling_type = None
        for power, subPower, lv in [power1, power2, power3]:
            if self._check_sparkling_power(power, subPower, lv):
                sparkling_type = subPower
                break
        if sparkling_type is None:
            self.send_log("闪耀力检测未通过")
            self._cycle_step_index += 1
            return

        # 捕获力检测
        catch_ok = False
        for power, subPower, lv in [power1, power2, power3]:
            if self._check_catching_power(sparkling_type, power, subPower, lv):
                catch_ok = True
                if power == ZaDlcDonutPowerType.Catching:
                    if self._special_sparkling_condition:
                        self.send_log(f"捕获力检测通过：属性={subPower.value}，在允许的属性列表中")
                    else:
                        self.send_log(f"捕获力检测通过：属性={subPower.value}，等级={lv}")
                break
        if not catch_ok:
            self.send_log("捕获力检测未通过")
            if sparkling_type and self._special_sparkling_condition:
                self.send_log(f"当前闪耀力属性：{sparkling_type.value}")
                allow_types = self._sparkling_power_type_list if isinstance(self._sparkling_power_type_list, list) else [self._sparkling_power_type_list]
                self.send_log(f"允许的属性列表：{allow_types}")
            self._cycle_step_index += 1
            return

        # 头目力/大大力/小小力检测（或关系）
        alpha_humungo_teensy_ok = (self._alpha_power_level == -1 and self._humungo_power_level == -1 and self._teensy_power_level == -1) or any([
            (self._alpha_power_level != -1 and (self._check_alpha_power(p1, s1, l1) or self._check_alpha_power(p2, s2, l2) or self._check_alpha_power(p3, s3, l3))),
            (self._humungo_power_level != -1 and (self._check_humungo_power(p1, s1, l1) or self._check_humungo_power(p2, s2, l2) or self._check_humungo_power(p3, s3, l3))),
            (self._teensy_power_level != -1 and (self._check_teensy_power(p1, s1, l1) or self._check_teensy_power(p2, s2, l2) or self._check_teensy_power(p3, s3, l3)))
        ])
        if not alpha_humungo_teensy_ok:
            self.send_log("头目力、大大力、小小力检测未通过")
            self._cycle_step_index += 1
            return

        # 道具力检测
        item_ok = any([self._check_item_power(p1, s1, l1), self._check_item_power(p2, s2, l2), self._check_item_power(p3, s3, l3)])
        if not item_ok:
            self.send_log("道具力检测未通过")
            self._cycle_step_index += 1
            return

        # 多多力检测
        big_haul_ok = any([self._check_big_haul_power(p1, s1, l1), self._check_big_haul_power(p2, s2, l2), self._check_big_haul_power(p3, s3, l3)])
        if not big_haul_ok:
            self.send_log("多多力检测未通过")
            self._cycle_step_index += 1
            return

        # 所有检测通过，成功计数
        self._success_count += 1
        success_content = {}
        success_content["recipe"] = self.get_para('Recipe')
        # 添加统计信息到 details 字段（基类会显示在卡片底部）
        stats = f"失败次数：{self._invalid_cycle_count}/{self.cycle_times}\n成功次数：{self._success_count}/{self._target_success_count}"
        success_content["details"] = stats
        if text1:
            success_content["power1"] = text1
        if text2:
            success_content["power2"] = text2
        if text3:
            success_content["power3"] = text3

        self._send_feishu_webhook('donut_success', f'✅ 甜甜圈成功 #{self._success_count}', success_content)
        self.send_log(f"✓ 成功找到符合条件的甜甜圈！当前成功次数：{self._success_count}/{self._target_success_count}")

        # 特定配方触发快捷键
        if self.get_para("Recipe") in ["闪耀力 - 混合 - 2.13%/0.118%", "闪耀力 - 彩虹 - 1.59%/0.242%"]:
            self.macro_run("recognition.pokemon.za.dlc.donut.capture", loop=1, block=True, timeout=None)
        self.macro_run("recognition.pokemon.za.dlc.donut.stop", loop=1, block=True, timeout=None)
        if self.get_para("Recipe") in ["闪耀力 - 混合 - 2.13%/0.118%", "闪耀力 - 彩虹 - 1.59%/0.242%"]:
            ocr1 = text1.replace('\n', ' ').strip() if text1 else ""
            ocr2 = text2.replace('\n', ' ').strip() if text2 else ""
            ocr3 = text3.replace('\n', ' ').strip() if text3 else ""
            self._trigger_obs_save('success', ocr1=ocr1, ocr2=ocr2, ocr3=ocr3)

        if self._success_count >= self._target_success_count:
            self.send_log(f"🎉 已达到目标成功次数 {self._target_success_count}，脚本停止")
            stop_content = {
                "reason": "正常完成",
                "total_cycles": self.cycle_times - 1
            }
            self._send_feishu_webhook('script_stop', '🛑 脚本已停止', stop_content)
            self._finished_process()
        else:
            self.send_log(f"继续下一次循环，还需 {self._target_success_count - self._success_count} 次成功")
            self._cycle_step_index = 0

    def _ocr_power_text(self, img):
        regions = [
            (250, 820, 300, 42),
            (250, 870, 300, 42),
            (250, 920, 300, 42),
            (380, 300, 130, 42),
        ]
        results = self.ocr_engine.batch_recognize_regions(img, regions)
        texts = []
        for result in results:
            text = result['text'] if result['text'] else ""
            text = " ".join(text.split())
            texts.append(text)
        return tuple(texts)

    def _split_ocr_power_text(self, text: str):
        text = text.replace('：', ':')
        powerStr, subPowerStr, lv = None, None, 0
        try:
            lindex = text.rindex('L')
        except ValueError:
            return None, None, 0

        lvStr = text[lindex:]
        text = text[:lindex]
        splits = text.split(':')
        powerStr = splits[0].strip()
        if len(splits) >= 2:
            subPowerStr = splits[1].strip()
        try:
            power = ZaDlcDonutPowerType(powerStr)
        except ValueError:
            power = None

        if power == ZaDlcDonutPowerType.Sparkling:
            try:
                subPower = ZaDlcTypeType(subPowerStr)
            except ValueError:
                subPower = None
        elif power == ZaDlcDonutPowerType.Catching:
            try:
                subPower = ZaDlcTypeType2(subPowerStr)
            except ValueError:
                subPower = None
        elif power == ZaDlcDonutPowerType.Item:
            try:
                subPower = ZaDlcDonutItemType(subPowerStr)
            except ValueError:
                subPower = None
        else:
            subPower = None

        if power:
            lvStr = lvStr.strip("LvV.:")
            try:
                lv = int(lvStr)
            except ValueError:
                lv = 0
        return power, subPower, lv

    def _check_sparkling_power(self, power: ZaDlcDonutPowerType, subPower, lv: int):
        if self._sparkling_power_level <= 0:
            return True
        if power != ZaDlcDonutPowerType.Sparkling or lv < self._sparkling_power_level:
            return False
        if not isinstance(subPower, ZaDlcTypeType):
            return False
        allow_types = self._sparkling_power_type_list if isinstance(self._sparkling_power_type_list, list) else [self._sparkling_power_type_list]
        if not allow_types:
            return True
        if subPower == ZaDlcTypeType.All:
            return True
        if subPower.value in allow_types:
            return True
        return False

    def _check_catching_power(self, sparkling_type: ZaDlcTypeType, power: ZaDlcDonutPowerType, subPower, lv: int):
        if self._catching_power_level <= 0:
            return True
        if power != ZaDlcDonutPowerType.Catching or lv < self._catching_power_level:
            return False
        if not isinstance(subPower, ZaDlcTypeType2):
            return False

        if self._special_sparkling_condition:
            allow_types = self._sparkling_power_type_list if isinstance(self._sparkling_power_type_list, list) else [self._sparkling_power_type_list]
            sub_value = subPower.value
            if sub_value == "所有属性":
                sub_value = "全属性"
            return sub_value in allow_types
        else:
            if subPower == ZaDlcTypeType2.All:
                return True
            if sparkling_type == ZaDlcTypeType.All:
                allow_types = [t for t in self._sparkling_power_type_list if t != ZaDlcTypeType.All.value]
                return subPower.value in allow_types
            else:
                return subPower.value == sparkling_type.value

    def _check_alpha_power(self, power: ZaDlcDonutPowerType, subPower, lv: int):
        if self._alpha_power_level <= 0:
            return True
        if power != ZaDlcDonutPowerType.Alpha or lv < self._alpha_power_level:
            return False
        return True

    def _check_humungo_power(self, power: ZaDlcDonutPowerType, subPower, lv: int):
        if self._humungo_power_level <= 0:
            return True
        if power != ZaDlcDonutPowerType.Humungo or lv < self._humungo_power_level:
            return False
        return True

    def _check_teensy_power(self, power: ZaDlcDonutPowerType, subPower, lv: int):
        if self._teensy_power_level <= 0:
            return True
        if power != ZaDlcDonutPowerType.Teensy or lv < self._teensy_power_level:
            return False
        return True

    def _check_item_power(self, power: ZaDlcDonutPowerType, subPower, lv: int):
        if self._item_power_level <= 0:
            return True
        if power != ZaDlcDonutPowerType.Item or lv < self._item_power_level:
            return False
        if not isinstance(subPower, ZaDlcDonutItemType):
            return False
        allow_types = self._item_power_type_list if isinstance(self._item_power_type_list, list) else [self._item_power_type_list]
        if not allow_types:
            return True
        if subPower.value in allow_types:
            return True
        return False

    def _check_big_haul_power(self, power: ZaDlcDonutPowerType, subPower, lv: int):
        if self._big_haul_power_level <= 0:
            return True
        if power != ZaDlcDonutPowerType.BigHaul or lv < self._big_haul_power_level:
            return False
        return True

    def _log_special_condition_info(self):
        if self._special_sparkling_condition:
            allow_types = self._sparkling_power_type_list if isinstance(self._sparkling_power_type_list, list) else [self._sparkling_power_type_list]
            self.send_log(f"允许的属性列表：{allow_types}")

    def _check_durations(self):
        if self._durations <= 0:
            return False
        if self.run_time_span >= self._durations * 60:
            self.send_log("运行时间已到达设定值，脚本停止")
            stats = f"失败次数：{self._invalid_cycle_count}/{self.cycle_times}\n成功次数：{self._success_count}/{self._target_success_count}"
            stop_content = {
                "reason": "运行时间到达设定值",
                "details":stats,
                #"total_cycles": self.cycle_times,
                "run_time_seconds": self.run_time_span,
                #"success_count": self._success_count
            }
            self._send_feishu_webhook('script_stop', f'🕒 运行时间到达 {self._durations} 分钟', stop_content)
            self._finished_process()
            return True
        return False

    def _check_cycles(self):
        if self._loop <= 0:
            return False
        if self.cycle_times > self._loop:
            self.send_log("运行次数已到达设定值，脚本停止")
            stats = f"失败次数：{self._invalid_cycle_count}/{self.cycle_times}\n成功次数：{self._success_count}/{self._target_success_count}"
            stop_content = {
                "reason": "运行次数到达设定值",
                "details": stats,
                "total_cycles": self.cycle_times,
                "success_count": self._success_count
            }
            self._send_feishu_webhook('script_stop', f'🔢 运行次数到达 {self._loop} 次', stop_content)
            self._finished_process()
            return True
        return False