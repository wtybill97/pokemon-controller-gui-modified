from enum import Enum
import multiprocessing
import time
from datetime import datetime
import cv2
import os
from recognition.scripts.games.pokemon.swsh.dynamax_adventures.step01_start import SWSHDAStart
from recognition.scripts.games.pokemon.swsh.dynamax_adventures.step02_choose_path import SWSHDAChoosePath
from recognition.scripts.games.pokemon.swsh.dynamax_adventures.step03_battle import SWSHDABattle, SWSHDABattleResult
from recognition.scripts.games.pokemon.swsh.dynamax_adventures.step04_catch import SWSHDACatch, SWSHDACatchResult
from recognition.scripts.games.pokemon.swsh.dynamax_adventures.step05_switch_pokemon import SWSHDASwitchPokemon
from recognition.scripts.games.pokemon.swsh.dynamax_adventures.step06_shiny_keep import SWSHDAShinyKeep, SWSHDAShinyKeepResult
from recognition.scripts.games.pokemon.swsh.dynamax_adventures.step07_finish import SWSHDAFinish
from recognition.scripts.parameter_struct import ScriptParameter
from recognition.scripts.base.base_script import BaseScript, WorkflowEnum
from recognition.scripts.base.base_sub_step import SubStepRunningStatus

TRACE_LOG = False
DYNITE_ORE_PENALTY_LIMIT = 4


class SWSHDABallType(Enum):
    NotCatch = "不捕捉"
    PokeBall = "精灵球"
    GreatBall = "超级球"
    UltraBall = "高级球"
    MasterBall = "大师球"
    SafariBall = "狩猎球"
    LevelBall = "等级球"
    MoonBall = "月亮球"
    LureBall = "诱饵球"
    FriendBall = "友友球"
    LoveBall = "甜蜜球"
    FastBall = "速度球"
    HeavyBall = "重量球"
    PremierBall = "纪念球"
    RepeatBall = "重复球"
    TimerBall = "计时球"
    NestBall = "巢穴球"
    NetBall = "捕网球"
    DiveBall = "潜水球"
    LuxuryBall = "豪华球"
    HealBall = "治愈球"
    QuickBall = "先机球"
    DuskBall = "黑暗球"
    SportBall = "竞赛球"
    DreamBall = "梦境球"
    BeastBall = "究极球"


class SWSHDAWhenRestart(Enum):
    Never = "不重启"
    NotShiny_And_WonLegendary = "未闪光，击败传说宝可梦重启(寻找最佳路线，保留过程闪光宝可梦)"
    NotShiny_And_WonLegendary_Limit_Restart = f"未闪光，击败传说宝可梦重启(寻找最佳路线，保留过程闪光宝可梦，极矿石惩罚超过{DYNITE_ORE_PENALTY_LIMIT}时限制重启)"
    NotShinyLegendary_And_WonLegendary = "传说未闪光，击败传说宝可梦重启(寻找最佳路线)"
    FindCatchLegendaryBestRoad_And_NotShinyLegendary = "寻找击败传说宝可梦最佳路线，找到最佳路线后传说宝可梦未闪光重启"
    NotShinyLegendary = "传说宝可梦未闪光重启(已确认最佳路线)"


class SwshDynamaxAdventures(BaseScript):
    def __init__(self, stop_event: multiprocessing.Event, frame_queue: multiprocessing.Queue, controller_input_action_queue: multiprocessing.Queue, paras: dict = None):
        super().__init__(SwshDynamaxAdventures.script_name(),
                         stop_event, frame_queue, controller_input_action_queue, SwshDynamaxAdventures.script_paras())
        self._prepare_step_index = -1
        self._cycle_step_index = -1
        self._jump_next_frame = False
        self._dynite_ore_penalty_count = 0

        self._shiny_count = 0
        self._win_count = 0
        self._win_streaks_count = 0

        # ---------- ADDED: 极矿石和捕获计数 ----------
        self._dynite_ore_total = 0          # 累计极矿石总数
        self._ore_gained_this_cycle = 0     # 本轮循环中获得的极矿石（用于传说非闪光时清零）
        self._total_catch_count = 0         # 累计捕获宝可梦总数（包括小怪和传说）
        # ------------------------------------------

        self.set_paras(paras)

        

        # 获取脚本参数
        self._loop = self.get_para("loop")
        self._durations = self.get_para("durations")
        self._secondary = self.get_para(
            "secondary") if "secondary" in paras else False
        self._use_record = self.get_para(
            "use_record") if "use_record" in paras else 1
        self._save_record = self.get_para(
            "save_record") if "save_record" in paras else 1
        restart_flag_value = self.get_para(
            "not_keep_restart") if "not_keep_restart" in paras else SWSHDAWhenRestart.Never.value
        self._restart_flag = SWSHDAWhenRestart(restart_flag_value)
        self._current_restart_flag = self._restart_flag
        self._only_keep_shiny_legendary = self.get_para(
            "only_keep_shiny_legendary") if "only_keep_shiny_legendary" in paras else False
        self._choose_path = [self.get_para("choose_path_1") if "choose_path_1" in paras else 0,
                             self.get_para(
                                 "choose_path_2") if "choose_path_2" in paras else 0,
                             self.get_para("choose_path_3") if "choose_path_3" in paras else 0]
        self._path_enter_event = [True,
                                  self.get_para(
                                      "path_enter_event_2") if "path_enter_event_2" in paras else True,
                                  self.get_para(
                                      "path_enter_event_3") if "path_enter_event_3" in paras else True,
                                  self.get_para("path_event_4") if "path_event_4" in paras else True]
        self._path_leave_event = [True, self.get_para("path_leave_event_2") if "path_leave_event_2" in paras else True,
                                  self.get_para("path_leave_event_3") if "path_leave_event_3" in paras else True]
        self._disable_dynamax = [self.get_para("disable_dynamax_1") if "disable_dynamax_1" in paras else False,
                                 self.get_para(
            "disable_dynamax_2") if "disable_dynamax_2" in paras else False,
            self.get_para(
            "disable_dynamax_3") if "disable_dynamax_3" in paras else True,#默认禁用第三场极巨化
            self.get_para("disable_dynamax_4") if "disable_dynamax_4" in paras else False]
        catch_ball_value = [self.get_para("catch_ball_1") if "catch_ball_1" in paras else SWSHDABallType.PokeBall.value,
                            self.get_para(
                                "catch_ball_2") if "catch_ball_2" in paras else SWSHDABallType.PokeBall.value,
                            self.get_para(
                                "catch_ball_3") if "catch_ball_3" in paras else SWSHDABallType.PokeBall.value,
                            self.get_para("catch_ball_4") if "catch_ball_4" in paras else SWSHDABallType.BeastBall.value]
        self._catch_ball = [SWSHDABallType(value)
                            for value in catch_ball_value]
        self._switch_pokemon = [self.get_para("switch_pokemon_1") if "switch_pokemon_1" in paras else True,
                                self.get_para(
                                    "switch_pokemon_2") if "switch_pokemon_2" in paras else True,
                                self.get_para("switch_pokemon_3") if "switch_pokemon_3" in paras else True]

    @staticmethod
    def script_name() -> str:
        return "宝可梦-剑盾-极巨大冒险(自测3个月毕业)"

    @staticmethod
    def script_paras() -> dict:
        paras = dict()
        paras["loop"] = ScriptParameter(
            "loop", int, -1, "运行次数")
        paras["durations"] = ScriptParameter(
            "durations", float, -1, "运行时长（分钟）")
        paras["only_keep_shiny_legendary"] = ScriptParameter(
            "only_keep_shiny_legendary", bool, "False", "只带走闪光传说宝可梦", ["False", "True"])
        paras["not_keep_restart"] = ScriptParameter(
            "not_keep_restart", str, SWSHDAWhenRestart.NotShiny_And_WonLegendary_Limit_Restart.value, "重启游戏选项（有极矿石惩罚）", [e.value for e in SWSHDAWhenRestart])
        paras["secondary"] = ScriptParameter(
            "secondary", bool, "False", "副设备", ["False", "True"])

        paras["use_record"] = ScriptParameter(
            "use_record", int, 1, "使用记录（0:不使用 1-3:使用记录1-3）", ["1", "2", "3", "0"])
        paras["save_record"] = ScriptParameter(
            "save_record", int, 1, "保存并覆盖原有记录（0:不保存 1-3:覆盖记录位置1-3）", ["1", "2", "3", "0"])

        paras["choose_path_1"] = ScriptParameter(
            "choose_path_1", int, 0, "战斗1 选择路径（0:默认路径，负数:向左移动，正数:向右移动，数字:移动次数）")
        paras["disable_dynamax_1"] = ScriptParameter(
            "disable_dynamax_1", bool, "False", "战斗1 禁用极巨化", ["False", "True"])
        paras["catch_ball_1"] = ScriptParameter(
            "catch_ball_1", str, SWSHDABallType.PokeBall.value, "战斗1 捕捉球种", [e.value for e in SWSHDABallType])
        paras["switch_pokemon_1"] = ScriptParameter(
            "switch_pokemon_1", bool, "True", "战斗1 是否更换使用宝可梦（未捕捉跳过此步骤）", ["False", "True"])

        paras["choose_path_2"] = ScriptParameter(
            "choose_path_2", int, 0, "战斗2 选择路径（0:默认路径，负数:向左移动，正数:向右移动，数字:移动次数）")
        paras["path_enter_event_2"] = ScriptParameter(
            "path_enter_event_2", bool, "True", "战斗2 进入战斗路径事件（True:连点A False:连点B）", ["False", "True"])
        paras["disable_dynamax_2"] = ScriptParameter(
            "disable_dynamax_2", bool, "False", "战斗2 禁用极巨化", ["False", "True"])
        paras["catch_ball_2"] = ScriptParameter(
            "catch_ball_2", str, SWSHDABallType.PokeBall.value, "战斗2 捕捉球种", [e.value for e in SWSHDABallType])
        paras["switch_pokemon_2"] = ScriptParameter(
            "switch_pokemon_2", bool, "True", "战斗2 是否更换使用宝可梦（未捕捉跳过此步骤）", ["False", "True"])
        paras["path_leave_event_2"] = ScriptParameter(
            "path_leave_event_2", bool, "True", "战斗2 离开战斗路径事件（True:连点A False:连点B）", ["False", "True"])

        paras["choose_path_3"] = ScriptParameter(
            "choose_path_3", int, 0, "战斗3 选择路径（0:默认路径，负数:向左移动，正数:向右移动，数字:移动次数）")
        paras["path_enter_event_3"] = ScriptParameter(
            "path_enter_event_3", bool, "True", "战斗3 进入战斗路径事件（True:连点A False:连点B）", ["False", "True"])
        paras["disable_dynamax_3"] = ScriptParameter(
            "disable_dynamax_3", bool, "True", "战斗3 禁用极巨化", ["False", "True"])#默认战斗3禁用极巨化
        paras["catch_ball_3"] = ScriptParameter(
            "catch_ball_3", str, SWSHDABallType.PokeBall.value, "战斗3 捕捉球种", [e.value for e in SWSHDABallType])
        paras["switch_pokemon_3"] = ScriptParameter(
            "switch_pokemon_3", bool, "True", "战斗3 是否更换使用宝可梦（未捕捉跳过此步骤）", ["False", "True"])
        paras["path_leave_event_3"] = ScriptParameter(
            "path_leave_event_3", bool, "True", "战斗2 离开战斗路径事件（True:连点A False:连点B）", ["False", "True"])

        paras["disable_dynamax_4"] = ScriptParameter(
            "disable_dynamax_4", bool, "False", "Boss战 禁用极巨化", ["False", "True"])
        paras["catch_ball_4"] = ScriptParameter(
            "catch_ball_4", str, SWSHDABallType.BeastBall.value, "BOSS战 捕捉球种", [e.value for e in SWSHDABallType])
        return paras

    def _check_durations(self):
        if self._durations <= 0:
            return False
        if self.run_time_span >= self._durations * 60:
            self.send_log("运行时间已到达设定值，脚本停止")
            self._finished_process()   # 正常停止，不传错误信息
            return True
        return False

    def _check_cycles(self):
        if self._loop <= 0:
            return False
        if self.cycle_times > self._loop:
            self.send_log("运行次数已到达设定值，脚本停止")
            self._finished_process()   # 正常停止，不传错误信息
            return True
        return False

    # MODIFIED: 增加 error_msg 参数，并添加极矿石和捕获计数到通知
    def _finished_process(self, error_msg: str = None):
        if error_msg:
            # 构造统计信息字符串
            stats = (
            #f"错误信息：{error_msg}\n"
            f"累计闪光数：{self._shiny_count}/{self._total_catch_count}\n"
            #f"本轮轮次：{self.cycle_times}\n"
            f"成功攻略次数：{self._win_count}/{self.cycle_times}\n"
            f"累计极矿石：{self._dynite_ore_total}\n"
            #f"累计捕获宝可梦：{self._total_catch_count}"
        )
            feishu_content = {
                'error': error_msg,
                'details': stats
            }
            self.send_notification(
                title='⚠️ 极巨大冒险脚本异常停止',
                feishu_content=feishu_content,
                meow_title="⚠️ 极巨大冒险脚本异常停止",
                meow_content=f"{error_msg}\n{stats}"
            )
            self._trigger_obs_save('error_stop', error=error_msg)
        run_time_span = self.run_time_span
        self.macro_stop(block=True)
        self.send_log("[{}] 脚本完成，已运行{}次，耗时{}小时{}分{}秒".format(SwshDynamaxAdventures.script_name(), self.cycle_times - 1, int(
            run_time_span/3600), int((run_time_span % 3600) / 60), int(run_time_span % 60)))
        self.stop_work()

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
        self.send_log(f"开始运行{SwshDynamaxAdventures.script_name()}脚本")
        if self._restart_flag == SWSHDAWhenRestart.FindCatchLegendaryBestRoad_And_NotShinyLegendary:
            self._current_restart_flag = SWSHDAWhenRestart.NotShiny_And_WonLegendary
            self.send_log("开始寻找击败传说宝可梦最佳路线")
        if self._restart_flag == SWSHDAWhenRestart.NotShinyLegendary or self._restart_flag == SWSHDAWhenRestart.NotShinyLegendary_And_WonLegendary:
            self._only_keep_shiny_legendary = True

    def on_cycle(self):
        run_time_span = self.run_time_span
        log_txt = ""
        if self._current_restart_flag == SWSHDAWhenRestart.NotShinyLegendary:
            log_txt += f"脚本运行中，已确认最佳路线，已经运行{self.cycle_times}次，成功攻略大冒险{self._win_count}次，带回闪光宝可梦{self._shiny_count}只"
        else:
            if self._restart_flag == SWSHDAWhenRestart.FindCatchLegendaryBestRoad_And_NotShinyLegendary:
                log_txt += f"脚本运行中，寻找最佳路线中，已经运行{self.cycle_times}次，成功攻略大冒险{self._win_count}次，带回闪光宝可梦{self._shiny_count}只"
            else:
                log_txt += f"脚本运行中，已经运行{self.cycle_times}次，成功攻略大冒险{self._win_count}次，带回闪光宝可梦{self._shiny_count}只"
        if self._dynite_ore_penalty_count >= 3:
            log_txt += f"，极矿石惩罚数量为{self._dynite_ore_penalty_count}"
        log_txt += f"\n累计极矿石: {self._dynite_ore_total}，累计捕获宝可梦: {self._total_catch_count}"
        log_txt += f"\n耗时{int(run_time_span/3600)}小时{int((run_time_span % 3600) / 60)}分{int(run_time_span % 60)}秒"
        self.send_log(log_txt)

    def on_stop(self):
        run_time_span = self.run_time_span
        self.send_log("[{}] 脚本停止，实际运行{}次，成功攻略大冒险{}次，带回闪光宝可梦{}只，累计极矿石{}，累计捕获宝可梦{}，耗时{}小时{}分{}秒".format(SwshDynamaxAdventures.script_name(
        ), self.cycle_times, self._win_count, self._shiny_count, self._dynite_ore_total, self._total_catch_count,
            int(run_time_span/3600), int((run_time_span % 3600) / 60), int(run_time_span % 60)))

    def on_error(self):
        pass

    @property
    def _prepare_step_list(self):
        return [
            self.prepare_step_0,
        ]

    def prepare_step_0(self):
        self._prepare_step_index += 1

    @property
    def _cycle_step_list(self):
        return [
            self.step_1_start,
            self.step_2_choose_path,
            self.step_3_battle,
            self.step_4_catch,
            self.step_5_switch_pokemon,
            self.step_6_shiny_keep,
            self.step_7_finish,
        ]

    def _re_cycle(self):
        self.macro_stop()
        self.set_cycle_continue()
        self._cycle_step_index = 0
        self._battle_index = 0
        self._legendary_caught = False
        self._ore_gained_this_cycle = 0
        self._swsh_da_start = SWSHDAStart(
            self, save_record=self._save_record, choose_record=self._use_record, timeout=90)
        self._swsh_da_choose_path = None
        self._swsh_da_battle = None
        self._swsh_da_catch = None
        self._swsh_da_switch_pokemon = None
        self._swsh_da_shiny_keep = None
        self._swsh_da_finish = SWSHDAFinish(self)

    def _cycle_init(self):
        self._battle_index = 0
        self._legendary_caught = False
        consecutive = self._win_streaks_count
        consecutive_restart = self._dynite_ore_penalty_count
        if consecutive_restart >= 3:
            deduct = min(consecutive_restart, 10)
            self._dynite_ore_total = self._dynite_ore_total - deduct  # 去掉 max(0, ...)
            self.send_log(f"连续重启 {consecutive_restart} 次，扣除 {deduct} 个极矿石，剩余极矿石: {self._dynite_ore_total}")
        self._ore_gained_this_cycle = 0
        self._swsh_da_start = SWSHDAStart(
            self, save_record=self._save_record, choose_record=self._use_record, timeout=90)
        self._swsh_da_choose_path = None
        self._swsh_da_battle = None
        self._swsh_da_catch = None
        self._swsh_da_switch_pokemon = None
        self._swsh_da_shiny_keep = None
        self._swsh_da_finish = SWSHDAFinish(self)

    def step_1_start(self):
        status = self._swsh_da_start.run()
        if status == SubStepRunningStatus.Running:
            return
        elif status == SubStepRunningStatus.OK:
            self._swsh_da_choose_path = SWSHDAChoosePath(
                self, True, True, battle_index=self._battle_index, path=self._choose_path[self._battle_index],)
            self._cycle_step_index += 1
            if TRACE_LOG:
                self.send_log("开始选择路径")
        else:
            error_msg = f"swsh_da_start 函数返回状态为 {status.name}"
            self.send_log(error_msg)
            self._finished_process(error_msg)

    def step_2_choose_path(self):
        status = self._swsh_da_choose_path.run()
        if status == SubStepRunningStatus.Running:
            return
        elif status == SubStepRunningStatus.OK:
            self._swsh_da_battle = SWSHDABattle(
                self, battle_index=self._battle_index, disable_dynamax=self._disable_dynamax[self._battle_index])
            self._cycle_step_index += 1
            if TRACE_LOG:
                self.send_log("选择路径完成，准备战斗")
        else:
            error_msg = f"swsh_da_choose_path 函数返回状态为 {status.name}"
            self.send_log(error_msg)
            self._finished_process(error_msg)

    # ==================== MODIFIED: BOSS战惩罚上限时强制使用精灵球 ====================
    def step_3_battle(self):
        status = self._swsh_da_battle.run()
        if status == SubStepRunningStatus.Running:
            return
        elif status == SubStepRunningStatus.Timeout:
            error_msg = "swsh_da_battle 函数返回状态为 Timeout"
            self.send_log(error_msg)
            self._finished_process(error_msg)
        elif status == SubStepRunningStatus.OK:
            if self._swsh_da_battle.battle_status == SWSHDABattleResult.Won:
                if self._battle_index >= 3:
                    self._legendary_caught = True
                # ==================== MODIFIED START ====================
                # 每场战斗胜利增加极矿石（原为捕获时增加）
                self._dynite_ore_total += 1
                self._ore_gained_this_cycle += 1
                # ==================== MODIFIED END ====================
                catch_flag = (
                    self._catch_ball[self._battle_index] != SWSHDABallType.NotCatch)
                # ---------- MODIFIED START ----------
                target_ball = self._catch_ball[self._battle_index].value
                # 如果是 BOSS 战且极矿石惩罚已达到上限，强制使用精灵球
                if self._battle_index >= 3 and self._dynite_ore_penalty_count >= DYNITE_ORE_PENALTY_LIMIT:
                    target_ball = SWSHDABallType.PokeBall.value
                    self.send_log("极矿石惩罚已达上限，本轮 BOSS 战强制使用精灵球")
                # ---------- MODIFIED END ----------
                self._swsh_da_catch = SWSHDACatch(
                    self, battle_index=self._battle_index, catch=catch_flag, target_ball=target_ball)
                self._cycle_step_index += 1
                if TRACE_LOG:
                    self.send_log("胜利，准备捕捉")
                return

            self._win_streaks_count = 0

            if self._swsh_da_battle.battle_status == SWSHDABattleResult.Lost3:
                if self._restart_flag == SWSHDAWhenRestart.FindCatchLegendaryBestRoad_And_NotShinyLegendary:
                    self._only_keep_shiny_legendary = False
                    self._current_restart_flag = SWSHDAWhenRestart.NotShiny_And_WonLegendary
                self._re_cycle()
                if TRACE_LOG:
                    self.send_log("失败3，重启开始下一轮大冒险")
                return
            if self._current_restart_flag == SWSHDAWhenRestart.NotShinyLegendary:
                if TRACE_LOG:
                    self.send_log("未击败传说宝可梦，重启开始下一轮大冒险")
                self.restart_game()
                return
            if self._swsh_da_battle.battle_status == SWSHDABattleResult.Lost1:
                self._cycle_step_index = 5
                self._swsh_da_shiny_keep = SWSHDAShinyKeep(
                    self, only_keep_shiny_legendary=self._only_keep_shiny_legendary, legendary_caught=self._legendary_caught)
                if TRACE_LOG:
                    self.send_log("失败1，带走宝可梦")
                return
            elif self._swsh_da_battle.battle_status == SWSHDABattleResult.Lost2:
                self._cycle_step_index = 6
                if TRACE_LOG:
                    self.send_log("失败2，准备结束")
                return
            elif self._swsh_da_battle.battle_status == SWSHDABattleResult.Lost3:
                self._re_cycle()
                if TRACE_LOG:
                    self.send_log("失败3，重启开始下一轮大冒险")
                return
        else:
            error_msg = f"swsh_da_battle 函数返回状态为 {status.name}"
            self.send_log(error_msg)
            self._finished_process(error_msg)
    # ==================== MODIFIED END ====================

    def step_4_catch(self):
        status = self._swsh_da_catch.run()
        if status == SubStepRunningStatus.Running:
            return
        elif status == SubStepRunningStatus.Timeout:
            error_msg = "swsh_da_catch 函数返回状态为 Timeout"
            self.send_log(error_msg)
            self._finished_process(error_msg)
        elif status == SubStepRunningStatus.OK:
            if self._swsh_da_catch.catch_result == SWSHDACatchResult.Caught:
                if self._legendary_caught:
                    self._win_count += 1
                    self._win_streaks_count += 1
                    if TRACE_LOG:
                        self.send_log("传说宝可梦捕捉成功")

                    if self._win_streaks_count >= 3 and self._restart_flag == SWSHDAWhenRestart.FindCatchLegendaryBestRoad_And_NotShinyLegendary and self._current_restart_flag != SWSHDAWhenRestart.NotShinyLegendary:
                        self._only_keep_shiny_legendary = True
                        self._current_restart_flag = SWSHDAWhenRestart.NotShinyLegendary
                        self.send_log("连续3次击败传说宝可梦，已确认最佳路线")

                    self._cycle_step_index = 5
                    self._swsh_da_shiny_keep = SWSHDAShinyKeep(
                        self, only_keep_shiny_legendary=self._only_keep_shiny_legendary, legendary_caught=self._legendary_caught)
                    return
                else:
                    switch_flag = self._switch_pokemon[self._battle_index]
                    self._swsh_da_switch_pokemon = SWSHDASwitchPokemon(
                        self, battle_index=self._battle_index, switch=switch_flag)
                    self._cycle_step_index += 1
                    if TRACE_LOG:
                        self.send_log("捕捉成功，准备切换宝可梦")
            else:
                # ==================== MODIFIED START ====================
                # 未捕捉的情况
                # 如果是 BOSS 战（battle_index >= 3），直接结束本次冒险
                if self._battle_index >= 3:
                    self._cycle_step_index = 6  # 跳转到 step_7_finish
                    return
                # 非 BOSS 战：进入下一场战斗
                self._battle_index += 1
                # 如果下一场是 BOSS 战（battle_index == 3），则直接创建战斗，跳过路径选择
                if self._battle_index == 3:
                    self._swsh_da_battle = SWSHDABattle(
                        self, battle_index=self._battle_index, disable_dynamax=self._disable_dynamax[self._battle_index])
                    self._cycle_step_index = 2  # 直接跳转到战斗步骤
                    if TRACE_LOG:
                        self.send_log("未捕捉，进入 BOSS 战")
                else:
                    self._swsh_da_choose_path = SWSHDAChoosePath(
                        self, leave_event=self._path_leave_event[self._battle_index - 1], enter_event=self._path_enter_event[self._battle_index], battle_index=self._battle_index, path=self._choose_path[self._battle_index],)
                    self._cycle_step_index = 1
                    if TRACE_LOG:
                        self.send_log("未捕捉，准备重新选择路径")
            return
            # ==================== MODIFIED END ====================
        else:
            error_msg = f"swsh_da_catch 函数返回状态为 {status.name}"
            self.send_log(error_msg)
            self._finished_process(error_msg)

    def step_5_switch_pokemon(self):
        status = self._swsh_da_switch_pokemon.run()
        if status == SubStepRunningStatus.Running:
            return
        elif status == SubStepRunningStatus.Timeout:
            error_msg = "swsh_da_switch_pokemon 函数返回状态为 Timeout"
            self.send_log(error_msg)
            self._finished_process(error_msg)
        elif status == SubStepRunningStatus.OK:
            self._battle_index += 1
            self._swsh_da_choose_path = SWSHDAChoosePath(
                self, leave_event=self._path_leave_event[self._battle_index - 1], enter_event=self._path_enter_event[self._battle_index], battle_index=self._battle_index, path=0,)
            self._cycle_step_index = 1
            if TRACE_LOG:
                self.send_log("切换宝可梦成功，准备重新选择路径")
            return
        else:
            error_msg = f"swsh_da_switch_pokemon 函数返回状态为 {status.name}"
            self.send_log(error_msg)
            self._finished_process(error_msg)

    # ==================== MODIFIED: 惩罚上限时放弃传说闪光且不重启 ====================
    def step_6_shiny_keep(self):
        status = self._swsh_da_shiny_keep.run()
        if status == SubStepRunningStatus.Running:
            return
        elif status == SubStepRunningStatus.OK:
            # 惩罚上限标志
            penalty_limit = self._dynite_ore_penalty_count >= DYNITE_ORE_PENALTY_LIMIT

            if self._swsh_da_shiny_keep.kept_result == SWSHDAShinyKeepResult.KeptLegendary:
                # 传说闪光（正常情况下会停止脚本）
                self._shiny_count += 1
                self.send_log("检测到传说宝可梦闪光，请手动确认")
                # 发送通知
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"shiny_{timestamp}.png"
                    cv2.imwrite(screenshot_path, self.current_frame_960x540)
                    self.send_log(f"闪光宝可梦截图已保存: {screenshot_path}")
                    # 构造飞书内容字典
                    feishu_content = {
                        '闪光宝可梦类型': '传说宝可梦',
                        '累计闪光数': f"{self._shiny_count}/{self._total_catch_count}",
                        '成功攻略次数': f"{self._win_count}/{self.cycle_times}",
                        '累计极矿石': self._dynite_ore_total,
                    }
                    
                    # 统一发送通知
                    self.send_notification(
                        title='✨ 捕获到闪光传说宝可梦！',
                        feishu_content=feishu_content,
                        image_path=screenshot_path,
                        meow_title="✨ 捕获到闪光传说宝可梦！",
                        meow_content=f"累计闪光数: {self._shiny_count}/{self._total_catch_count}\n成功攻略次数: {self._win_count}/{self.cycle_times}\n累计极矿石: {self._dynite_ore_total}"
                    )
                    self._trigger_obs_save('legendary_shiny', shiny_count=self._shiny_count)
                    os.remove(screenshot_path)
                except Exception as e:
                    self.send_log(f"发送闪光截图失败: {e}")

                # ---------- MODIFIED START ----------
                if penalty_limit:
                    # 惩罚上限：不停止脚本，继续下一轮（但传说闪光已经在子步骤中被放弃了，这里实际上不会进入）
                    # 但为了安全，如果子步骤没有放弃，这里强制不停止
                    self.send_log("极矿石惩罚已达上限，传说闪光宝可梦已放弃，脚本继续运行")
                    self._cycle_step_index += 1
                    return
                else:
                    self._finished_process()
                    return
                # ---------- MODIFIED END ----------

            if self._current_restart_flag == SWSHDAWhenRestart.NotShinyLegendary:
                self.send_log("传说宝可梦未检测到闪光，重启开始下一轮大冒险")
                self.restart_game()
                return
            if self._swsh_da_shiny_keep.kept_result == SWSHDAShinyKeepResult.Kept:
                # 非传说闪光宝可梦保留成功
                self._cycle_step_index += 1
                return
            if TRACE_LOG:
                self.send_log("未检测到闪光宝可梦")

            # 传说捕获但未闪光，清零极矿石
            if self._legendary_caught and self._swsh_da_shiny_keep.kept_result == SWSHDAShinyKeepResult.NotKept:
                if penalty_limit:
                    self._ore_gained_this_cycle += 2
                    self._dynite_ore_total = self._dynite_ore_total+2
                else:
                    self._dynite_ore_total = self._dynite_ore_total - self._ore_gained_this_cycle
                    self.send_log(f"传说宝可梦非闪光，清零本轮获得的 {self._ore_gained_this_cycle} 个极矿石，剩余极矿石: {self._dynite_ore_total}")
                    self._ore_gained_this_cycle = 0

            # ---------- MODIFIED START ----------
            # 惩罚上限时，跳过所有重启逻辑，直接继续循环
            if penalty_limit:
                self.send_log("极矿石惩罚已达上限，不重启，继续下一轮大冒险")
                self._cycle_step_index += 1
                return
            # ---------- MODIFIED END ----------

            # 以下为原有的重启判断逻辑
            if (self._current_restart_flag == SWSHDAWhenRestart.NotShiny_And_WonLegendary
                or self._current_restart_flag == SWSHDAWhenRestart.NotShinyLegendary_And_WonLegendary
                or (self._current_restart_flag == SWSHDAWhenRestart.NotShiny_And_WonLegendary_Limit_Restart and self._dynite_ore_penalty_count < DYNITE_ORE_PENALTY_LIMIT)) and self._legendary_caught:
                self.send_log("成功击败传说宝可梦，未检测到闪光宝可梦，重启开始下一轮大冒险")
                self.restart_game()
                return
            if self._current_restart_flag == SWSHDAWhenRestart.NotShiny_And_WonLegendary_Limit_Restart and self._dynite_ore_penalty_count >= DYNITE_ORE_PENALTY_LIMIT:
                self.send_log(f"成功击败传说宝可梦，未检测到闪光宝可梦，极矿石惩罚数量超过{DYNITE_ORE_PENALTY_LIMIT}，继续开始下一轮大冒险")
                self._cycle_step_index += 1
                return
            self._win_streaks_count = 0
            self._cycle_step_index += 1
            return
        else:
            error_msg = f"swsh_da_shiny_keep 函数返回状态为 {status.name}"
            self.send_log(error_msg)
            self._finished_process(error_msg)
    # ==================== MODIFIED END ====================

    def step_7_finish(self):
        status = self._swsh_da_finish.run()
        self._dynite_ore_penalty_count = 0
        if status == SubStepRunningStatus.Running:
            return
        elif status == SubStepRunningStatus.OK:
            self._cycle_step_index += 1
            return
        else:
            error_msg = f"swsh_da_finish 函数返回状态为 {status.name}"
            self.send_log(error_msg)
            self._finished_process(error_msg)

    def restart_game(self):
        self.macro_run("recognition.pokemon.swsh.common.restart_game",
                       1, {"secondary": str(self._secondary)}, True, None)
        self._dynite_ore_penalty_count += 1
        self._re_cycle()