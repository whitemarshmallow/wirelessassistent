# assistant_core.py
import importlib, traceback
from typing import Dict, Any

# ---------- 1. 统一懒加载 ----------
_agent        = None
_cell_solver  = None
_ue_solver    = None

def _lazy_import_large_agent():
    global _agent
    if _agent is None:                       # 只初始化一次
        from agent_sql_r1_pingpong_knowledge_memory_6000_smallmodelact_select import Agent
        _agent = Agent(
            deepseek_api_key = "...",
            mysql_host       = "...",
            mysql_user       = "...",
            mysql_password   = "...",
            mysql_database   = "wireless_optimize_db"
        )
    return _agent

def _lazy_import_small_models(category: str):
    global _cell_solver, _ue_solver
    try:
        if category == "baseStation" and _cell_solver is None:
            mod = importlib.import_module("wireless_cell_test")
            _cell_solver = mod.solve_problem
        elif category == "user" and _ue_solver is None:
            mod = importlib.import_module("wireless_ue_test")
            _ue_solver = mod.solve_problem
    except Exception as e:
        return None, f"小模型导入失败: {e}"
    return (_cell_solver if category=="baseStation" else _ue_solver), ""

# ---------- 2. 主分析 ----------
def analyse_and_reply(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    req = {
      "message": "...",
      "context": {...},
      "modelOptions": {...}
    }
    """
    try:
        user_msg = req.get("message", "")
        ctx      = req.get("context", {}) or {}
        mdl_opt  = req.get("modelOptions", {}) or {}

        # === ① 大模型 ==================================================
        if mdl_opt.get("useLargeModel"):
            agent = _lazy_import_large_agent()          # 现在才真正 import / 连接 MySQL
            thinking, reply = agent.respond(user_msg)   # 你的 Agent.respond() 要返回 (thought, answer)
        else:
            thinking = "未启用大模型"
            reply    = "默认回复：未启用大模型"

        # === ② 小模型 ==================================================
        cat = ctx.get("category")
        solver, err = _lazy_import_small_models(cat)
        if solver:
            sm_text = solver(ctx.get("subOption", 1), None)  # 这里根据实际签名传 df 或 file
        else:
            sm_text = err or f"未知 category={cat}"
        reply += f"\n\n[小模型输出]\n{sm_text}"

        # === ③ 画图 ====================================================
        from gen_core import generate_images   # 避免顶层循环依赖
        images = []
        if ctx.get("imageTypes"):
            img_req = {
                "imageTypes": ctx["imageTypes"],
                "category":   cat,
                "subOption":  ctx.get("subOption"),
                "dataSource": ctx.get("dataSource"),
                "downloadParams": ctx.get("downloadParams", {})
            }
            img_resp = generate_images(img_req)
            if img_resp.get("success"):
                images = img_resp["images"]

        # === ④ 返回 =====================================================
        return {
            "success": True,
            "code":    200,
            "thinking": thinking,
            "reply":    reply,
            "images":   images,
            "data":     {}
        }

    except Exception as e:
        return {
            "success": False,
            "code":    500,
            "reply":   f"服务器异常: {e}",
            "thinking": traceback.format_exc(),
            "images": [],
            "data":   {}
        }
