import pandas as pd
import numpy as np
import io                # <‑ 之前就可能已经用到
import contextlib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report

import warnings
warnings.filterwarnings('ignore')


############################
# 1. 加载数据
############################
def load_data(csv_file_path):
    """
    读取CSV文件，并返回pandas的DataFrame。
    这里假设表头已包含你给出的字段名称。
    """
    # df = pd.read_csv(csv_file_path, sep="\t", engine="python")  # 如果实际分隔符是","或其他，需调整sep

    df = pd.read_csv(csv_file_path, sep=",")

    # df = pd.read_csv(csv_file_path, sep="\t")

    # df = pd.read_csv(csv_file_path, sep=None, engine="python")

    return df


############################
# 覆盖与性能评估 (问题1)
# 演示：利用地理位置信息、与基站距离等特征，预测RSRP（Viavi.UE.Rsrp）
############################
def solve_coverage_performance(df):
    """
    使用简单的线性回归来预测 RSRP
    """
    print("=== 覆盖与性能评估 (RSRP回归示例) ===")

    features = ["Viavi.Geo.x", "Viavi.Geo.y", "Viavi.Geo.z", "Viavi.UE.servingDistance"]
    target = "Viavi.UE.Rsrp"

    # 数据清理
    df_sub = df.dropna(subset=features + [target]).copy()
    if len(df_sub) < 2:
        print("数据量太少，无法进行回归演示")
        return

    X = df_sub[features]
    y = df_sub[target]

    # 拆分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("均方误差(MSE):", mse)
    print("预测结果(部分):", y_pred[:5])
    print("-----------------------------------\n")


############################
# 网络容量与吞吐分析 (问题2)
# 演示：根据 CQI、RSRP、SINR等特征，预测下行吞吐 DRB.UEThpDl
############################
def solve_capacity_throughput(df):
    print("=== 网络容量与吞吐分析 (下行吞吐回归示例) ===")

    features = ["Viavi.UE.Rsrp", "Viavi.UE.RsSinr", "DRB.UECqiDl",
                "RRU.PrbUsedDl", "TB.TotNbrDl"]
    target = "DRB.UEThpDl"

    df_sub = df.dropna(subset=features + [target]).copy()
    if len(df_sub) < 2:
        print("数据量太少，无法进行回归演示")
        return

    X = df_sub[features]
    y = df_sub[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("均方误差(MSE):", mse)
    print("预测结果(部分):", y_pred[:5])
    print("------------------------------------------\n")


############################
# QoS与切片管理 (问题3)
# 演示：根据5QI、RSRP、SINR、Priority等特征，分类预测QoS.Score高/低
############################
def solve_qos_slice(df):
    print("=== QoS与切片管理 (QoS.Score分类示例) ===")

    df_sub = df.dropna(subset=["Viavi.QoS.Score"]).copy()
    # 简单定义：QoS.Score >= 90标记为1，否则为0
    df_sub["QoS_label"] = df_sub["Viavi.QoS.Score"].apply(lambda x: 1 if x >= 90 else 0)

    features = ["Viavi.QoS.Priority", "Viavi.QoS.5qi", "Viavi.UE.Rsrp",
                "Viavi.UE.RsSinr", "DRB.UECqiDl"]
    target = "QoS_label"

    df_sub = df_sub.dropna(subset=features + [target])
    if len(df_sub) < 2:
        print("数据量太少，无法进行分类演示")
        return

    X = df_sub[features]
    y = df_sub[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, digits=4))
    print("----------------------------------------\n")


############################
# 异常检测与网络故障排查 (问题4)
# 演示：Isolation Forest 无监督异常检测
############################
def solve_anomaly_detection(df):
    print("=== 异常检测与网络故障排查 (Isolation Forest示例) ===")

    features = ["Viavi.UE.Rsrp", "Viavi.UE.RsSinr", "DRB.UEThpDl", "DRB.UEThpUl"]
    df_sub = df.dropna(subset=features).copy()

    if len(df_sub) < 2:
        print("数据量太少，无法进行异常检测演示")
        return

    X = df_sub[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
    iso_forest.fit(X_scaled)

    anomaly_labels = iso_forest.predict(X_scaled)  # 1表示正常，-1表示异常
    df_sub["anomaly_label"] = anomaly_labels

    anomaly_count = sum(anomaly_labels == -1)
    normal_count = sum(anomaly_labels == 1)

    print(f"数据总量: {len(df_sub)}, 异常数量: {anomaly_count}, 正常数量: {normal_count}")
    # 显示部分异常点
    anomaly_points = df_sub[df_sub["anomaly_label"] == -1]
    print("可能异常的样本(前5条):")
    print(anomaly_points.head())
    print("--------------------------------------------------\n")


############################
# 基于位置的业务体验分析 (问题5)
# 演示：K-Means聚类位置 + RSRP + SINR等
############################
def solve_location_performance(df):
    print("=== 位置与业务体验分析 (K-Means聚类示例) ===")

    features = ["Viavi.Geo.x", "Viavi.Geo.y", "Viavi.UE.Rsrp", "Viavi.UE.RsSinr"]
    df_sub = df.dropna(subset=features).copy()

    if len(df_sub) < 2:
        print("数据量太少，无法进行聚类演示")
        return

    X = df_sub[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42)
    df_sub["cluster"] = kmeans.fit_predict(X_scaled)

    print(df_sub[["Viavi.Geo.x", "Viavi.Geo.y", "Viavi.UE.Rsrp", "Viavi.UE.RsSinr", "cluster"]].head())
    print("-------------------------------------------------\n")


############################
# Massive MIMO与波束管理评估 (问题6)
# 演示：根据RSRP、SINR、RI等特征分类波束ID
############################
def solve_massive_mimo(df):
    print("=== Massive MIMO与波束管理 (波束ID分类示例) ===")

    features = ["Viavi.UE.Rsrp", "Viavi.UE.RsSinr", "Viavi.UE.RI"]
    target = "Viavi.Cell.beam"

    df_sub = df.dropna(subset=features + [target]).copy()
    if len(df_sub) < 2:
        print("数据量太少，无法进行分类演示")
        return

    X = df_sub[features]
    y = df_sub[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("---------------------------------------------\n")


############################
# TA与距离管理 (问题7)
# 演示：利用距离预测Timing Advance
############################
def solve_ta_distance(df):
    print("=== TA与距离管理 (Timing Advance回归示例) ===")

    features = ["Viavi.UE.servingDistance"]
    target = "RRC.timingAdvance"

    df_sub = df.dropna(subset=features + [target]).copy()
    if len(df_sub) < 2:
        print("数据量太少，无法进行回归演示")
        return

    X = df_sub[features]
    y = df_sub[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("均方误差(MSE):", mse)
    print("预测结果(部分):", y_pred[:5])
    print("--------------------------------------------\n")


############################
# 问题调度函数
############################
# def solve_problem(problem_id, df):
#     """
#     problem_id:
#       1 -> 覆盖与性能评估
#       2 -> 网络容量与吞吐分析
#       3 -> QoS与切片管理
#       4 -> 异常检测与网络故障排查
#       5 -> 基于位置的业务体验分析
#       6 -> Massive MIMO与波束管理评估
#       7 -> TA与距离管理
#     """
#     if problem_id == 1:
#         solve_coverage_performance(df)
#     elif problem_id == 2:
#         solve_capacity_throughput(df)
#     elif problem_id == 3:
#         solve_qos_slice(df)
#     elif problem_id == 4:
#         solve_anomaly_detection(df)
#     elif problem_id == 5:
#         solve_location_performance(df)
#     elif problem_id == 6:
#         solve_massive_mimo(df)
#     elif problem_id == 7:
#         solve_ta_distance(df)
#     else:
#         print("无效的 problem_id，请输入 1~7 之间的数字。")

# ─── 文件: wireless_UE_test.py ────────────────────────────────────────────
import io
import contextlib          # ⬅️ 新增，用于捕获 stdout
# ... 其余原有 import 保持不变 …

# -----------------------------------------------------------------------
# 现有的 7 个子函数 (solve_coverage_performance 等) 都不动
# -----------------------------------------------------------------------

# ------------------ 新版调度函数 ---------------------------------------
def solve_problem(problem_id, df, *, to_str: bool = False):
    """
    problem_id:
      1 → 覆盖与性能评估
      2 → 网络容量与吞吐分析
      3 → QoS 与切片管理
      4 → 异常检测与网络故障排查
      5 → 基于位置的业务体验分析
      6 → Massive‑MIMO 与波束管理评估
      7 → TA 与距离管理

    参数
    ----
    df      : 预先加载好的 DataFrame
    to_str  : True  → 捕获所有 print 并返回字符串
              False → 直接 print（旧行为）

    返回
    ----
    * to_str=False → None
    * to_str=True  → str
    """

    # ---------- 若需要捕获，重定向 stdout ----------
    if to_str:
        buf = io.StringIO()
        ctx = contextlib.redirect_stdout(buf)
        ctx.__enter__()

    try:
        # ---------- 原有逻辑 ----------
        if problem_id == 1:
            solve_coverage_performance(df)
        elif problem_id == 2:
            solve_capacity_throughput(df)
        elif problem_id == 3:
            solve_qos_slice(df)
        elif problem_id == 4:
            solve_anomaly_detection(df)
        elif problem_id == 5:
            solve_location_performance(df)
        elif problem_id == 6:
            solve_massive_mimo(df)
        elif problem_id == 7:
            solve_ta_distance(df)
        else:
            print("无效的 problem_id，请输入 1~7 之间的数字。")

    finally:
        # ---------- 恢复 stdout 并返回结果 ----------
        if to_str:
            ctx.__exit__(None, None, None)
            return buf.getvalue()


############################
# 主函数：控制台交互示例
############################
if __name__ == "__main__":
    # 1) 加载数据文件
    #   假设你把示例数据另存为 "sample_data.txt"
    data_file = "data/UEReports.csv"  # 修改为你的实际数据文件名
    df = load_data(data_file)

    # 2) 打印菜单供用户选择
    print("请选择需要解决的问题ID：")
    print(" 1 -> 覆盖与性能评估")
    print(" 2 -> 网络容量与吞吐分析")
    print(" 3 -> QoS与切片管理")
    print(" 4 -> 异常检测与网络故障排查")
    print(" 5 -> 基于位置的业务体验分析")
    print(" 6 -> Massive MIMO与波束管理评估")
    print(" 7 -> TA与距离管理")

    # 3) 接收用户输入
    try:
        problem_id = int(input("请输入 1~7 的数字： "))
        solve_problem(problem_id, df)
    except ValueError:
        print("输入错误，请输入数字 1~7。")
