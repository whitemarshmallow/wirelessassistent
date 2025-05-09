#!/usr/bin/env python3

# ──────────────────────────────────────────────────────────────
# 项目结构 & 职责一览【调测速查】
# ──────────────────────────────────────────────────────────────
#
#  当前目录：
#  ├─ app.py                            ← Flask 启动入口，只管 HTTP/JSON             (你大多只改路由)
#  ├─ assistant_core.py                 ← “/api/assistant” 的业务编排：               (经常改)
#  │                                       ‑ 拆前端 JSON
#  │                                       ‑ 调大模型 Agent  (可关)
#  │                                       ‑ 懒加载小模型    (cell / UE)
#  │                                       ‑ 组装统一回复
#  │
#  ├─ gen_core.py                       ← 画热力图核心：读 CSV → Matplotlib → PNG   (已稳定)
#  ├─ data_filter.py                    ← 下载 Viavi CSV/ZIP 的裁剪打包逻辑          (偶尔改)
#  │
#  ├─ wireless_cell_test.py             ← “基站侧” 7 个小模型示例                    (算法实验)
#  ├─ wireless_UE_test.py               ← “UE 侧” 7 个小模型示例                    (算法实验)
#  │
#  ├─ agent_sql_r1_pingpong_… .py       ← 大模型 + MySQL 记忆 + 小模型混合 Agent     (可选开关)
#  └─ test.py                           ← 零散脚本/一次性调试                       (自由发挥)
#
#  数据目录（与代码同级或自行修改路径）：
#  ├─ data/                             ← UEReports.csv / CellReports.csv … 原始数据
#  └─ generated_images/                 ← gen_core 生成的 PNG 将自动写入
#
#  调试小贴士
#  ----------------------------------------------------------------
#  1. 仅改小模型算法 → 改 wireless_cell_test.py / wireless_UE_test.py，记得
#     solve_problem(..., to_str=True) 能把 print 捕获回 assistant_core。
#  2. 调接口：
#       • /api/assistant            → 业务主流程（文字 + 小模型）
#       • /api/assistant/image      → 画图
#       • /api/download‑viaviData   → 数据裁剪/下载
#  3. 若前端 500，大多是 assistant_core 捕获到异常；看返回里的
#     "thinking" 或终端 traceback 定位。
#  4. 屏蔽大模型：前端传 {"modelOptions":{"useLargeModel":false}}
# ──────────────────────────────────────────────────────────────


# app.py  —— 直接  python app.py  即可启动
import os, io, json, zipfile, pathlib
import matplotlib
matplotlib.use("Agg")
import traceback

from flask import Flask, request, jsonify, send_file, Response, abort

from assistant_core import analyse_and_reply
# from assistant_core import analyse_and_reply
# 将核心算法导入
from gen_core import generate_images, DATA_DIR
from data_filter import filter_viavi



# from assistant_core import analyse_and_reply

app = Flask(__name__)

# ---------- ① 生成热力图接口 ---------------------------------
@app.route("/api/assistant/image", methods=["POST"])
def api_generate_image():
    try:
        # Flask 帮你把 HTTP‑Body 按 JSON 解析成 Python dict。
        # force=True 表示：即便请求头里没写 Content‑Type: application/json，也强行按 JSON 解析。
        # 如果前端发来的不是合法 JSON，这里会抛异常 → 进入 except 分支。
        req_dict = request.get_json(force=True)   # 保证 UTF‑8
    except Exception as e:
        # 解析失败：直接返回一个 JSON，字段保持和前端约定一致（success/msg/images）
        return jsonify({
            "success": False,
            "msg": f"JSON 解析失败: {e}",
            "images": []
        })

    # 走到这说明 JSON 解析成功，字典在 req_dict 里
    # generate_images() 是你在 gen_core.py 里写的“核心算法”，
    #   负责：读 CSV → 画图 → 把结果打包成 {"success":true, "images":[...]}。
    resp = generate_images(req_dict)

    # jsonify 会把 Python dict 转成 JSON，并自动带上 Content‑Type: application/json
    return jsonify(resp)


# ---------- ② Viavi 数据下载接口 -----------------------------------
@app.route("/api/download-viaviData", methods=["POST"])
def api_download_viavi():
    try:
        req = request.get_json(force=True)   # 同上：拿到前端 JSON 字典

        # filter_viavi() 在 data_filter.py 里：
        #   负责：① 读 UEReports.csv   ② 根据时间 & feature 裁剪
        #        ③ 按 format=csv|zip 生成一个“可下载对象”
        # 返回值三元组：
        #   obj  : ① BytesIO (zip 压缩包) ② 或 str 路径 (单个 CSV)
        #   mime : 浏览器用来识别文件类型的 Content‑Type
        #   fname: 前端保存时看到的文件名
        obj, mime, fname = filter_viavi(req)

        # Flask 内置 send_file：
        #   • 可以把磁盘文件 / 内存 BytesIO 当作文件流返回
        #   • as_attachment=True  ⇒  强行触发“下载”而不是在浏览器预览
        #   • download_name       ⇒  Content‑Disposition 指定的文件名
        return send_file(
            obj,
            mimetype=mime,
            as_attachment=True,
            download_name=fname
        )

    except Exception as e:
        # 出错时把 traceback 打到日志里，返回 400 给前端
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 400

# ----------------- ③ 模型分析接口 -----------------------------
@app.route("/api/assistant", methods=["POST"])
def api_assistant():
    try:
        req_json = request.get_json(force=True)  # JSON → dict
    except Exception as e:
        return jsonify({"success": False, "code": 400, "msg": f"JSON 解析失败: {e}"}), 400

    try:
        resp = analyse_and_reply(req_json)  # 调业务核心
        return jsonify(resp)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"success": False, "code": 500, "msg": str(e)}), 500

# ---------- 入口 ----------
if __name__ == "__main__":
    # 0.0.0.0 方便 Docker / 局域网访问，生产环境请配合 gunicorn、nginx 等
    app.run(host="0.0.0.0", port=5000, debug=True)
