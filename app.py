#!/usr/bin/env python3
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
