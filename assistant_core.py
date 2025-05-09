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
                mod = importlib.import_module("wireless_UE_test")
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

            sm_text = solver(prob_id, df,to_str=True)
        else:
            sm_text = err_msg or f"未知 category={category}"

        reply += f"\n\n[小模型输出]\n{sm_text}"

        # ---------- 3. 返回 ----------
        return {
            "success": True,
            "code"   : 200,
            "reply"  : reply,
            "smallmodel":sm_text,
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
