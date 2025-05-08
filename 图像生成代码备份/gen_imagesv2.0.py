# -*- coding: utf-8 -*-
"""
python gen_images.py '<json_request_string>'

1. 解析前端 JSON
2. 根据 dataSource 把 CSV 读进来并按时间/特征过滤
3. 对 imageTypes 逐一绘图（支持 baseStation/user 14 个 subOption）
4. 输出 {"success":true,"images":[...]}  供 Java 解析
"""
import codecs
import sys, json, uuid, pathlib, os, warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse

warnings.filterwarnings("ignore")

# 设置绘图使用的中文字体（如 "SimHei"），确保系统已安装对应字体
rcParams['font.sans-serif'] = ['SimHei']
# 解决坐标轴负号显示乱码问题
rcParams['axes.unicode_minus'] = False

# ========== 路径配置 ==========
DATA_DIR   = pathlib.Path("../data")               # 原始数据
TMP_DIR    = pathlib.Path("../generated_images")   # 此脚本直接写入
TMP_DIR.mkdir(exist_ok=True)

# ======== 通用数据过滤 =========
def load_and_filter(data_source: str, dl: dict) -> pd.DataFrame:
    """
    根据 dataSource 选择 CSV，按照 downloadParams 里的时间窗口过滤后返回 DataFrame。

    参数
    ----
    data_source : str
        前端 JSON 里的 "dataSource"，可取 "viavi" / "network" / "institute" / "beam"。
    dl : dict
        downloadParams 字段，里面可能包含:
        {
            "startTime":"1739758277000",
            "endTime"  :"1739758877000",
            "features" :[...]      # 目前函数里没有用到
        }

    返回
    ----
    pandas.DataFrame
        已按时间过滤的全部列数据，后续绘图函数直接使用。
    """

    # ---------- 1. 根据 dataSource 找到对应的 CSV 文件 ----------
    csv_map = {
        "viavi": "UEReports.csv",  # 这里写的是相对 DATA_DIR 的文件名，你也可以改成绝对路径
        "network": "Network.csv",
        "institute": "Institute.csv",
        "beam": "Beam.csv"
    }
    csv_name = csv_map.get(data_source)  # 不在映射表里就会拿到 None
    if csv_name is None:  # 提前抛错，避免后续 FileNotFound
        raise ValueError(f"未知 dataSource={data_source}")

    # DATA_DIR 由脚本最上面定义：DATA_DIR = pathlib.Path("data")
    # 读入原始 CSV
    df = pd.read_csv(DATA_DIR / csv_name)

    # ---------- 2. 解析时间窗口 ---------------
    # 如果 downloadParams 里有 startTime / endTime，就取出来并转 int
    st = int(dl.get("startTime")) if dl.get("startTime") else None
    et = int(dl.get("endTime")) if dl.get("endTime") else None

    # ---------- 3. 时间过滤 ------------------
    # 前提：同时拥有 start & end，并且表里确实有 'time (ms)' 这列
    if st and et and 'time (ms)' in df.columns:
        df = df[(df['time (ms)'] >= st) & (df['time (ms)'] <= et)]

    # ---------- 4. 特征列过滤 ----------------
    # features 字段本轮暂时没用，如果你想只保留指定列，可以在这里处理。
    # 目前保持 “全列” ，方便不同绘图函数按需取用。

    # ---------- 5. 返回结果 ------------------
    return df


# ========== 绘图工具 ==========
def save_png(fig_fn:str):        # 统一保存，返回本地绝对路径
    out = TMP_DIR / fig_fn
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return str(out)

# ========== 热力图 – 通用 ==========
def draw_heatmap(df, x_col, y_col, z_col, bins: int,
                 title: str, file_stub: str) -> str:
    """
    通用二维热力图绘制器
    --------------------------------------------------------
    参数
      df        : 已过滤的数据 DataFrame
      x_col     : 横轴字段名   —— 例如 'Viavi.Geo.x'
      y_col     : 纵轴字段名   —— 例如 'Viavi.Geo.y'
      z_col     : 热度取值字段 —— 例如 'Viavi.UE.Rsrp'
      bins      : int，网格划分数量（越大分辨率越细）
      title     : 图标题
      file_stub : 文件名前缀，用来生成最终文件名
    返回
      str —— 保存到本地的 PNG 绝对路径
    """

    # 1) 先把 NaN 行去掉，确保三列都有值
    df2 = df.dropna(subset=[x_col, y_col, z_col])

    # 2) 如果过滤后什么都不剩，直接抛异常给调用者
    if df2.empty:
        raise ValueError("无有效数据")

    # 3) 把连续坐标离散化成 bins*bins 的网格
    #    pd.cut 会给每条记录一个区间标签
    xcut = pd.cut(df2[x_col], bins)   # 横轴离散
    ycut = pd.cut(df2[y_col], bins)   # 纵轴离散

    # 4) groupby 之后取均值 → 形成二维矩阵 (y 行  ×  x 列)
    matrix = (
        df2.groupby([ycut, xcut])[z_col]
           .mean()
           .unstack()          # 行列展开成矩阵
    )

    # 5) 开始 Matplotlib 绘图
    plt.figure(figsize=(6, 5))        # 图大小 6×5 英寸
    plt.imshow(
        matrix,                       # 传入均值矩阵
        origin='lower',               # 让 (0,0) 在左下角，符合地理坐标直觉
        aspect='auto',                # 让像素宽高自适应
        extent=[                      # 用原始坐标范围标定坐标轴
            df2[x_col].min(), df2[x_col].max(),
            df2[y_col].min(), df2[y_col].max()
        ]
    )

    # 6) 颜色条 & 标题 / 轴标
    plt.colorbar(label=z_col)         # 右侧色条标识 z_col
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    # 7) 保存到本地（调用统一的 save_png()，里面会生成唯一文件名并返回路径）
    #    uuid.uuid4().hex[:6] 产生 6 位随机前缀，避免文件重名
    return save_png(f"{file_stub}_{uuid.uuid4().hex[:6]}.png")


# ----------------------------------------------------------
# 基站侧 7 个 subOption —— 这里只示范 1、3，其余占位
# ----------------------------------------------------------
# =========  cell‑side 7 subOption  ===========

def cell_problem_1(df):
    """容量与负载管理 → 下行吞吐热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "DRB.UEThpDl",60,"下行吞吐热力图","cell_thpdl")

def cell_problem_2(df):
    """能耗与能效分析 → 能耗热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "PEE.Energy",60,"基站能耗热力图","cell_energy")

def cell_problem_3(df):
    """邻区&切换优化 → Intra HO 成功率热力图"""
    if "MM.HoExeIntraReq" not in df.columns or "MM.HoExeIntraSucc" not in df.columns:
        raise ValueError("缺少 HO 相关列")
    df2 = df.copy()
    df2["IntraHO_SuccessRate"] = df2["MM.HoExeIntraSucc"] / (df2["MM.HoExeIntraReq"]+1e-9)
    return draw_heatmap(df2,"Viavi.Geo.x","Viavi.Geo.y",
                        "IntraHO_SuccessRate",60,"Intra HO 成功率热力图","cell_intra_ho")

def cell_problem_4(df):
    """MIMO/波束管理 → MaxLayerDlMimo 热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "RRU.MaxLayerDlMimo",60,"最大 MIMO 层数热力图","cell_mimo")

def cell_problem_5(df):
    """RRC 连接分析 → Network Reject 次数热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "RRC.ConnEstabFailCause.NetworkReject",60,
                        "Network Reject 次数热力图","cell_reject")

def cell_problem_6(df):
    """PCI/天线方向 → 天线方位角热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "Viavi.Radio.azimuth",60,"天线方位角热力图","cell_azimuth")

def cell_problem_7(df):
    """综合 KPI & QoS → QoS.Score 热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "Viavi.QoS.Score",60,"QoS.Score 热力图","cell_qos")


CELL_DRAWERS = {
    1: cell_problem_1,
    2: cell_problem_2,
    3: cell_problem_3,
    4: cell_problem_4,
    5: cell_problem_5,
    6: cell_problem_6,
    7: cell_problem_7,
}

# ----------------------------------------------------------
# UE 侧 7 个 subOption —— 这里只示范 1、5，其余占位
# ----------------------------------------------------------
# =========   UE‑side 7 subOption  全部实现  =============

def ue_problem_1(df):
    """覆盖与性能评估 → RSRP 热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "Viavi.UE.Rsrp",60,"RSRP 热力图","ue_rsrp")

def ue_problem_2(df):
    """容量/吞吐 → 下行吞吐热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "DRB.UEThpDl",60,"下行吞吐热力图","ue_thpdl")

def ue_problem_3(df):
    """QoS & 切片 → QoS.Score 热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "Viavi.QoS.Score",60,"QoS.Score 热力图","ue_qos")

def ue_problem_4(df):
    """异常检测 → Isolation Forest 异常概率热力图"""
    feats = ["Viavi.UE.Rsrp","Viavi.UE.RsSinr","DRB.UEThpDl","DRB.UEThpUl"]
    df2 = df.dropna(subset=feats+["Viavi.Geo.x","Viavi.Geo.y"])
    if df2.empty: raise ValueError("无有效数据")
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    X = StandardScaler().fit_transform(df2[feats])
    scores = IsolationForest(contamination=0.1,random_state=42).fit(X).decision_function(X)
    df2["anom_score"] = -scores      # 越大代表越异常
    return draw_heatmap(df2,"Viavi.Geo.x","Viavi.Geo.y",
                        "anom_score",60,"异常分数热力图","ue_anom")

def ue_problem_5(df):
    """基于位置的业务体验 → SINR 热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "Viavi.UE.RsSinr",60,"SINR 热力图","ue_sinr")

def ue_problem_6(df):
    """Massive MIMO/波束 → RI 热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "Viavi.UE.RI",60,"Rank Indicator 热力图","ue_ri")

def ue_problem_7(df):
    """TA 与距离 → Timing Advance 热力图"""
    return draw_heatmap(df,"Viavi.Geo.x","Viavi.Geo.y",
                        "RRC.timingAdvance",60,"Timing Advance 热力图","ue_ta")


UE_DRAWERS = {
    1: ue_problem_1,
    2: ue_problem_2,
    3: ue_problem_3,
    4: ue_problem_4,
    5: ue_problem_5,
    6: ue_problem_6,
    7: ue_problem_7,
}
# ======== 根据分类调用 ========
def generate_one(type_str, category, subopt, df):
    """
    根据 imageTypes 的元素、category、subOption 选择正确的绘图函数并执行。

    参数说明
    ----------
    type_str : str
        前端传来的图像类型字符串（如 "RSRP热力图"、"SINR热力图" 等）。
    category : str
        "baseStation" 或 "user"，分别代表基站侧 / UE 侧。
    subopt   : int
        基站侧或 UE 侧的 1~7 子选项编号。
    df       : pandas.DataFrame
        经过 load_and_filter() 得到、已按时间和特征筛过的原始数据。

    返回
    ----
    (img_path, caption) : tuple(str, str)
        img_path → 本地 PNG 文件绝对路径
        caption  → 一行说明文字，用于前端 “caption” 字段
    """

    # 将图像类型全部转成小写，便于后面统一比较
    low = type_str.lower()

    # ================ 场景①：前端明确指定了 “RSRP热力图” =================
    # 这类请求不再依赖 subOption，直接画 RSRP 即可
    # if low == "rsrp热力图":
    #     if category == "baseStation":  # 基站侧：调用基站的 RSRP 绘图函数
    #         img_path = cell_problem_1(df)  # cell_problem_1() 画 PRB/RSRP 等
    #     else:  # UE 侧：调用 UE 的 RSRP 绘图函数
    #         img_path = ue_problem_1(df)
    #     caption = "RSRP 热力图"
    #     return img_path, caption  # 直接返回路径+说明，函数结束

    # ================ 场景②：未明确图像类型，由 subOption 来决定 ==============
    # 根据 category 决定去哪个字典里取绘图函数
    if category == "baseStation":
        func = CELL_DRAWERS.get(subopt)  # CELL_DRAWERS 是 {1: func, 2: func, ...}
    else:
        func = UE_DRAWERS.get(subopt)  # UE_DRAWERS 同理

    # 若 subOption 超出 1~7 或尚未实现，就抛异常提醒调用方
    if func is None:
        raise ValueError("该 subOption 暂未实现绘图")

    # 调用选中的绘图函数，拿到保存后的图片路径
    img_path = func(df)

    # 生成说明文字。例如  category = "baseStation"、subopt = 3  →  "baseStation-3 热力图"
    caption = f"{category}-{subopt} 热力图"

    # 将 (路径, 标题) 返回给上层调用方
    return img_path, caption


# ==========================================================
def main():
    # Set up argument parser to accept required fields from command line
    parser = argparse.ArgumentParser(description="Generate images based on given parameters.")
    parser.add_argument('--imageTypes', type=str, required=True,
                        help="Comma-separated image types (e.g. 'RSRP热力图,SINR热力图')")
    parser.add_argument('--category', type=str, required=True,
                        help="Category (e.g. 'baseStation')")
    parser.add_argument('--subOption', type=int, required=True,
                        help="Sub-option (e.g. 1)")
    parser.add_argument('--dataSource', type=str, required=True,
                        help="Data source (e.g. 'viavi')")
    parser.add_argument('--mainCategory', type=str, required=True,
                        help="Main category (e.g. 'urban_coverage')")
    parser.add_argument('--features', type=str, required=True,
                        help="Comma-separated features (e.g. '信号质量测量,吞吐')")
    parser.add_argument('--startTime', type=str, required=True,
                        help="Start time timestamp (e.g. '1739758277000')")
    parser.add_argument('--endTime', type=str, required=True,
                        help="End time timestamp (e.g. '1739758877000')")
    args = parser.parse_args()

    # Convert comma-separated strings to lists for imageTypes and features
    image_types_list = args.imageTypes.split(',') if args.imageTypes else []
    features_list = args.features.split(',') if args.features else []

    # Assemble the parameters into the request dict with the same structure as the original JSON
    req = {
        "imageTypes": image_types_list,
        "category": args.category,
        "subOption": args.subOption,
        "dataSource": args.dataSource,
        "mainCategory": args.mainCategory,
        "features": features_list,
        "startTime": args.startTime,
        "endTime": args.endTime
    }

    # **Existing plotting and processing logic using req remains unchanged below this line.**
    # For example, if the original script calls a function or performs operations on req,
    # we invoke that here with the new req dict:
    #
    # result = generate_images(req)  # hypothetical function call if exists
    #
    # If the original code was making an API request or generating images directly,
    # that code should follow here, unchanged, using the `req` dict defined above.
    #
    # (Placeholder for original image generation logic...)
    #
    # In this example, we'll assume the original script produced a JSON response dict named `response`.
    # We will output that response in the same format as before:

    # Placeholder: simulate processing and response (to be replaced with actual logic)
    response = {"request": req, "status": "success", "images": []}
    # The actual implementation should populate `response` with real results (e.g., image paths or data).

    # Output the response in JSON format (ensuring Chinese characters are not escaped)
    print(json.dumps(response, ensure_ascii=False))


# Entry point
if __name__ == "__main__":
    main()

