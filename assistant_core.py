# assistant_core.py  —— 纯分析，不再绘图
import importlib, traceback
from pathlib import Path
from typing   import Dict, Any

import pandas as pd

# ============================ 常量 ===============================
DATA_ROOT = Path(r"E:\数据平台案例汇总\图像生成模块\data")   # 原始 CSV 所在目录

# ============================ 懒加载缓存 =========================
_agent        = None      # DeepSeek‑Agent 缓存
_cell_solver  = None      # 基站侧小模型函数缓存
_ue_solver    = None      # UE  侧小模型函数缓存
# ----------------------------------------------------------------
# ① 懒加载大模型 Agent
# ----------------------------------------------------------------
def _lazy_import_large_agent():
    """首次调用时才 import + 实例化 Agent；后续直接复用全局 _agent。"""
    global _agent
    if _agent is None:
        from agent_sql_r1_pingpong_knowledge_memory_6000_smallmodelact_select import Agent
        _agent = Agent(
            deepseek_api_key="sk-d9c23ab8c0944d38bc9b177748d99f87",
            mysql_host     ="localhost",
            mysql_user     ="root",
            mysql_password ="1234",
            mysql_database ="wireless_optimize_db",
        )
    return _agent
# ----------------------------------------------------------------
# ② 懒加载小模型
# ----------------------------------------------------------------
def _lazy_import_small_models(category: str):
    """
    返回对应的 solve_problem() 函数：
      • baseStation → wireless_cell_test.solve_problem
      • user        → wireless_ue_test.solve_problem
    成功  -> (callable, None)
    失败/未知 -> (None, 错误信息)
    """
    global _cell_solver, _ue_solver
    try:
        if category == "baseStation":
            if _cell_solver is None:
                mod = importlib.import_module("wireless_cell_test")
                _cell_solver = mod.solve_problem
            return _cell_solver, None

        elif category == "user":
            if _ue_solver is None:
                mod = importlib.import_module("wireless_ue_test")
                _ue_solver = mod.solve_problem
            return _ue_solver, None

        else:
            return None, f"未知 category={category}"

    except Exception as e:
        return None, f"小模型导入失败: {e}"
# ----------------------------------------------------------------
# ③ 主入口：/api/assistant 调用这里
# ----------------------------------------------------------------
def analyse_and_reply(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    前端请求结构:
    {
      "message": "...",
      "context": {...},
      "modelOptions": {...}
    }
    """
    try:
        # ---------- 0. 拆请求 ----------
        user_msg = req.get("message", "")
        ctx      = req.get("context", {}) or {}
        mdl_opt  = req.get("modelOptions", {}) or {}

        # ---------- 1. 大模型 ----------
        if mdl_opt.get("useLargeModel"):
            agent    = _lazy_import_large_agent()
            reply    = agent.respond(user_msg)      # 你的 Agent.respond() 返回字符串
        else:
            reply = "默认回复：未启用大模型"

        # ---------- 2. 小模型 ----------
        category = ctx.get("category")
        solver, err_msg = _lazy_import_small_models(category)

        # if solver:
        #     prob_id   = ctx.get("subOption", 1)
        #     data_path = str(DATA_ROOT / ("CellReports.csv" if category=="baseStation"
        #                                  else "UEReports.csv"))
        #     sm_text   = solver(prob_id, data_path)
        # else:
        #     sm_text   = err_msg or f"未知 category={category}"
        #
        # reply += f"\n\n[小模型输出]\n{sm_text}"

        if solver:
            prob_id = ctx.get("subOption", 1)

            # 1) 根据 category 选对应 CSV 文件
            csv_path = DATA_ROOT / ("CellReports.csv" if category == "baseStation"
                                    else "UEReports.csv")

            # 2) **此处真正加载 DataFrame**
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                return {
                    "success": False,
                    "code": 500,
                    "reply": f"加载数据失败: {e}",
                    "data": {}
                }

            # 3) 把 DataFrame 传给 solve_problem
            sm_text = solver(prob_id, df)
        else:
            sm_text = err_msg or f"未知 category={category}"

        # ---------- 3. 返回 ----------
        return {
            "success": True,
            "code"   : 200,
            "reply"  : reply,
            "data"   : {}          # 以后若要返回结构化结果可填这里
        }

    # ---------- 兜底异常 ----------
    except Exception as e:
        return {
            "success" : False,
            "code"    : 500,
            "reply"   : f"服务器异常: {e}",
            "thinking": traceback.format_exc(),
            "data"    : {}
        }




# # assistant_core.py
# import importlib, traceback
# from pathlib import Path
# from typing import Dict, Any
#
# DATA_ROOT = Path(r"E:\数据平台案例汇总\图像生成模块\data")   # <<< 你的 CSV 目录
#
# # ---------- 1. 统一懒加载 ----------
# _agent        = None
# _cell_solver  = None
# _ue_solver    = None
#
# # ----------------------------------------------------------------
# # ① 懒加载大模型 Agent
# # ----------------------------------------------------------------
# def _lazy_import_large_agent():
#     """
#     返回全局唯一的 Agent 实例
#     * 第一次调用时才真正 `import Agent` 并创建连接
#     * 后续直接使用缓存的 `_agent`
#     """
#     global _agent
#     if _agent is None:                      # 只初始化一次
#         # 运行时才 import，避免启动阶段就连数据库 / 加载模型
#         from agent_sql_r1_pingpong_knowledge_memory_6000_smallmodelact_select import Agent
#
#         # 这些连接信息改成你的实际配置
#         _agent = Agent(
#             deepseek_api_key="sk-d9c23ab8c0944d38bc9b177748d99f87",
#             mysql_host="localhost",
#             mysql_user="root",
#             mysql_password="1234",
#             # mysql_database="wireless_optimize_db2"
#             mysql_database="wireless_optimize_db"
#         )
#     return _agent
#
# # ----------------------------------------------------------------
# # ② 懒加载小模型脚本
# # ----------------------------------------------------------------
# def _lazy_import_small_models(category: str):#输入是类别，基站/用户
#     """
#     根据 category 返回对应的 `solve_problem()`
#     返回: (solver_func | None, 错误信息 | str)
#     """
#     global _cell_solver, _ue_solver
#
#     try:
#         # ---------- 基站侧 ----------
#         if category == "baseStation":
#             if _cell_solver is None:                                   # 首次导入
#                 mod = importlib.import_module("wireless_cell_test")    # 动态导入模块
#                 _cell_solver = mod.solve_problem                       # 缓存函数
#             return _cell_solver, None
#
#         # ---------- UE  侧 ----------
#         elif category == "user":
#             if _ue_solver is None:
#                 mod = importlib.import_module("wireless_ue_test")
#                 _ue_solver = mod.solve_problem
#             return _ue_solver, None
#
#         # ---------- 未知 ----------
#         else:
#             return None, f"未知 category={category}"
#
#     except Exception as e:
#         # import 过程中出错 —— 把异常信息返回给调用方
#         return None, f"小模型导入失败: {e}"
#
# # ======================================================================
# # 3. 主入口 —— 前端 /api/assistant 直接调用这里
# # ======================================================================
# def analyse_and_reply(req: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     参数 req 结构（来自前端）::
#         {
#           "message": "<用户输入>",
#           "context": {...},        # 可选
#           "modelOptions": {...}    # 可选
#         }
#     返回 dict 结构  ⇢  app.py 会 jsonify() 后给前端
#     """
#     try:
#         # ---------- 解析请求 -----------------
#         user_msg = req.get("message", "")               # 当前对话内容
#         ctx      = req.get("context", {}) or {}         # 上下文
#         mdl_opt  = req.get("modelOptions", {}) or {}    # 模型开关
#
#         # =================================================================
#         # ①  大模型（DeepSeek Agent）—— 只有在 useLargeModel=true 时才调用
#         # =================================================================
#         if mdl_opt.get("useLargeModel"):
#             agent = _lazy_import_large_agent()          # 首次调用才 import & init
#             # thinking, reply = agent.respond(user_msg)   # 约定 respond 返回 (思考, 答复)
#             reply = agent.respond(user_msg)  # 约定 respond 返回 (思考, 答复)
#         else:
#             thinking = "未启用大模型"
#             reply    = "默认回复：未启用大模型"
#
#         # === ② 小模型 ==================================================
#         category = ctx.get("category")
#         solver, err = _lazy_import_small_models(category)
#
#         if solver:
#             # --------------------------------------------
#             # 1) 取 subOption（默认为 1）
#             prob_id = ctx.get("subOption", 1)
#
#             # 2) 根据 category 选正确的 CSV 路径
#             if category == "baseStation":
#                 data_path = str(DATA_ROOT / "CellReports.csv")
#             else:  # user
#                 data_path = str(DATA_ROOT / "UEReports.csv")
#
#             # 3) 调用小模型
#             sm_text = solver(prob_id, data_path)  # <<< 只改这一行 & 上面 4 行
#             # --------------------------------------------
#         else:
#             sm_text = err or f"未知 category={category}"
#
#         reply += f"\n\n[小模型输出]\n{sm_text}"
#
#         # =================================================================
#         # ③  生成热力图（如果 context.imageTypes 不为空）
#         # =================================================================
#         # from gen_core import generate_images            # 延迟 import 防循环依赖
#         # images = []
#         # if ctx.get("imageTypes"):
#         #     img_req = {
#         #         "imageTypes"    : ctx["imageTypes"],
#         #         "category"      : category,
#         #         "subOption"     : ctx.get("subOption"),
#         #         "dataSource"    : ctx.get("dataSource"),
#         #         "downloadParams": ctx.get("downloadParams", {})
#         #     }
#         #     img_resp = generate_images(img_req)
#         #     if img_resp.get("success"):
#         #         images = img_resp["images"]
#
#         # =================================================================
#         # ④  组织并返回结果
#         # =================================================================
#         return {
#             "success" : True,
#             "code"    : 200,
#             # "thinking": thinking,   # 大模型“思考过程” or 占位
#             "reply"   : reply,      # 综合回复
#             # "images"  : images,     # 可能为空列表
#             "data"    : {}          # 预留：结构化结果
#         }
#
#     # ------------------ 兜底异常 ------------------
#     except Exception as e:
#         # 把完整 traceback 放进 thinking，方便调试
#         return {
#             "success" : False,
#             "code"    : 500,
#             "reply"   : f"服务器异常: {e}",
#             "thinking": traceback.format_exc(),
#             "images"  : [],
#             "data"    : {}
#         }
