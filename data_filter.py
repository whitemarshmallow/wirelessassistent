# data_filter.py
# ───────────────────────────────────────────────────────────────
# 功能：把 UEReports.csv 按『时间窗口 + feature 列』切分成若干 CSV，
#       或者打包成 ZIP 供前端下载。
# 对外只暴露一个函数：filter_viavi(req_dict) → (bytesIO | filepath, mimetype, filename)
# ───────────────────────────────────────────────────────────────
import io
import zipfile
import tempfile
import os
import pathlib
import pandas as pd

from pathlib import Path

# ----------------------- 路径配置 ------------------------------
# DATA_DIR = <当前文件夹>/data
DATA_DIR = pathlib.Path(__file__).parent / "data"

# 原始 UEReports.csv 的全路径
CSV_FILE = DATA_DIR / "UEReports.csv"

# -------------------- feature → 对应列 -------------------------
# 如果以后要扩展，只在这里加映射即可
COL_MAP = {
    "信号质量测量": [
        'Viavi.UE.Rsrp', 'Viavi.UE.Rsrq',
        'Viavi.UE.RsSinr', 'Viavi.Cell.beam'
    ],
    "吞吐": [
        'DRB.UECqiDl', 'DRB.UEThpDl',
        'RRU.PrbUsedDl', 'RRU.PrbUsedUl'
    ],
    "位置信息": [
        'Viavi.Geo.x', 'Viavi.Geo.y', 'Viavi.Geo.z'
    ],
}

# -------------------- 读 CSV 的小工具 ---------------------------
def _load_csv() -> pd.DataFrame:
    """安全地加载 UEReports.csv，没有就报错"""
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"原始文件缺失: {CSV_FILE}")
    return pd.read_csv(CSV_FILE)

# =================================================================
#  主函数：供 app.py 调用
# =================================================================
def filter_viavi(req: dict):
    """
    参数 req —— 必须跟前端约定完全一致，例如：
    {
        "mainCategory": "urban_converage",
        "features": ["信号质量测量", "吞吐"],
        "startTime": "1739505002000",
        "endTime":   "1739505459000",
        "format":    "zip"          # 或 "csv"
    }

    返回值：(obj, mimetype, download_name)
      • obj       : BytesIO（zip）或 本地 CSV 路径字符串
      • mimetype  : 浏览器 Content‑Type
      • download_name : 前端收到后保存的文件名
    """

    # -------- 1. 基本字段校验 ------------------------------------
    need_keys = {"mainCategory", "features", "startTime", "endTime", "format"}
    miss = need_keys - req.keys()
    if miss:
        raise ValueError(f"缺少字段: {', '.join(miss)}")

    # features 必须是非空 list
    features = req["features"]
    if not isinstance(features, list) or not features:
        raise ValueError("features 必须为非空数组")

    # format 只能是 csv / zip
    if req["format"] not in ("csv", "zip"):
        raise ValueError("format 仅支持 csv / zip")

    # 时间戳转 int，方便比较
    start_ms = int(req["startTime"])
    end_ms   = int(req["endTime"])
    timerange = (start_ms, end_ms)
    tcol = 'time (ms)'                      # CSV 里时间列名

    # -------- 2. 读 CSV 并过滤时间 -------------------------------
    df = _load_csv()

    if tcol not in df.columns:
        raise RuntimeError(f"CSV 缺少列 {tcol}")

    df = df[(df[tcol] >= start_ms) & (df[tcol] <= end_ms)]

    # -------- 3. 针对每个 feature 生成一个子 CSV ----------------
    tmpdir = tempfile.TemporaryDirectory()  # 用系统临时目录托管子文件
    out_paths = []                          # 记录生成的 csv 路径

    for feat in features:
        cols = COL_MAP.get(feat)
        if not cols:           # 如果映射表里没有就跳过
            continue

        # 子 DataFrame：时间列 + 该 feature 对应列
        sub = df[[tcol] + cols]

        # 文件名示例： 信号质量测量_1739..._1739....csv
        fname = f"{feat}_{start_ms}_{end_ms}.csv"
        fpath = os.path.join(tmpdir.name, fname)

        sub.to_csv(fpath, index=False)
        out_paths.append(fpath)

    if not out_paths:
        raise ValueError("无合法 feature 或列映射")

    # -------- 4a. format == 'csv'，只能单文件 --------------------
    if req["format"] == "csv":
        if len(out_paths) != 1:
            raise ValueError("format=csv 时 features 只能包含一个元素")

        csv_buf = io.BytesIO()
        with open(out_paths[0], "rb") as f:
            csv_buf.write(f.read())
        csv_buf.seek(0)

        # 临时目录可以安全释放，因为我们已经把内容读入内存
        return (
            csv_buf,  # BytesIO 对象
            "text/csv; charset=utf-8",  # MIME
            os.path.basename(out_paths[0])  # 下载文件名
        )

    # -------- 4b. format == 'zip'，把多个 CSV 打包 ----------------
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_paths:
            # arcname 保留文件名，不带临时目录
            zf.write(p, arcname=os.path.basename(p))
    buf.seek(0)   # 指针回到开头，方便 Flask send_file

    zipname = f"{req['mainCategory']}_{start_ms}_{end_ms}.zip"
    return buf, "application/zip", zipname



# ==========================================================
# 本地测试用例：python data_filter.py
# ==========================================================
# if __name__ == "__main__":
    # import pprint, os, sys
    #
    # # ------- 1. 组装一个“前端请求” -------
    # req_demo = {
    #     "mainCategory": "urban_converage",          # 随便填，zip 文件名里会用到
    #     # "features": ["信号质量测量", "吞吐"],        # <= 确保都在 COL_MAP 里
    #     "features": ["信号质量测量"],  # <= 确保都在 COL_MAP 里
    #     "startTime": "1739505002000",               # 根据你的 CSV 调整
    #     "endTime":   "1739505459000",
    #     # "format":    "zip"                          # 改成 "csv" 只取一个 feature
    #     "format": "csv"
    # }
    #
    # print("\n=== 发起本地测试请求 ===")
    # pprint.pprint(req_demo, width=80)
    #
    # # ------- 2. 调用核心函数 -------
    # try:
    #     obj, mime, fname = filter_viavi(req_demo)
    # except Exception as e:
    #     print("\n[❌] filter_viavi 抛异常：", e)
    #     sys.exit(1)
    #
    # # ------- 3. 打印调试信息 -------
    # print("\n[✅] filter_viavi 返回成功：")
    # print("• mimetype      :", mime)
    # print("• download_name :", fname)
    #
    # if isinstance(obj, io.BytesIO):
    #     size_kb = len(obj.getvalue()) / 1024
    #     print(f"• BytesIO 大小 : {size_kb:.1f} KB")
    #     # 可

if __name__ == "__main__":
    # ====== 1. 准备一个模拟请求 ======
    req = {
        "mainCategory": "urban_converage",
        # "features": ["信号质量测量", "吞吐"],
        "features": ["信号质量测量"],
        # "startTime": "1739505002000",
        # "endTime": "1739505459000",
        "startTime": "1739758277000",
        "endTime": "1739758583000",
        "format": "csv"  # or "csv"
    }

    # ====== 2. 调用核心函数 ======
    obj, mime, fname = filter_viavi(req)
    print("MIME      :", mime)
    print("Down‑name :", fname)

    # ====== 3. 如果是 BytesIO → 写到当前目录，方便查看 ======
    if isinstance(obj, io.BytesIO):
        out_path = Path(fname)
        out_path.write_bytes(obj.getvalue())
        print(f"已把内容写到本地: {out_path.resolve()}")
    else:
        # obj 就是磁盘路径字符串（只可能出现在 format='csv' 且你改过实现）
        print("生成文件路径:", Path(obj).resolve())

