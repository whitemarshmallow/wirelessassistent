import pandas as pd
import numpy as np
import io                # <‑ 之前就可能已经用到
import contextlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, classification_report

import warnings
warnings.filterwarnings('ignore')


############################
# 1. 加载数据
############################
def load_data(csv_file_path):
    """
    读取基站侧数据文件并返回 DataFrame。
    根据实际情况调整分隔符 sep。
    """
    # 如果你的数据是制表符分隔(\t)，用 sep="\t"
    # 如果是逗号(,)，改为 sep=","
    df = pd.read_csv(csv_file_path, sep=",", engine="python")
    return df


############################
# 问题1：容量与负载管理 (示例：回归预测下行吞吐 DRB.UEThpDl)
############################
def solve_capacity_load(df):
    """
    演示思路：根据 PRB 使用、连接数、可用资源等特征，来预测基站下行吞吐 DRB.UEThpDl。
    使用一个简单的随机森林回归做示范。
    """
    print("=== [1] 容量与负载管理 -> 回归预测下行吞吐 ===")

    features = ["RRU.PrbUsedDl", "RRC.ConnMean", "RRC.ConnMax", "RRU.PrbAvailDl"]
    target = "DRB.UEThpDl"

    df_sub = df.dropna(subset=features + [target]).copy()
    if len(df_sub) < 2:
        print("数据量太少，无法回归演示")
        return

    X = df_sub[features]
    y = df_sub[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    print("预测值(部分):", y_pred[:5])
    print("----------------------------------\n")


############################
# 问题2：能耗与能效分析 (示例：回归预测基站能耗 PEE.Energy)
############################
def solve_energy_efficiency(df):
    """
    根据平均功率 PEE.AvgPower、负载(ThpDl, ThpUl)、连接数等特征来回归预测基站能耗 PEE.Energy。
    """
    print("=== [2] 能耗与能效分析 -> 回归预测基站能耗 ===")

    features = ["PEE.AvgPower", "DRB.UEThpDl", "DRB.UEThpUl", "RRC.ConnMean"]
    target = "PEE.Energy"

    df_sub = df.dropna(subset=features + [target]).copy()
    if len(df_sub) < 2:
        print("数据量太少，无法回归演示")
        return

    X = df_sub[features]
    y = df_sub[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    print("预测值(部分):", y_pred[:5])
    print("----------------------------------\n")


############################
# 问题3：邻区关系和切换优化
# (示例：根据 IntraReq/IntraSucc, InterReq/InterSucc 计算切换成功率是否达标)
############################
def solve_handover_optimization(df):
    """
    演示思路：根据 Intra/Inter HO 的请求与成功数计算成功率，
    将成功率高/低做一个简单的分类标签(如 >80%为1，否则0)，看哪些特征影响切换成功率。
    这里只演示如何做分类模型的流程，真实场景需要更多特征和更大数据。
    """
    print("=== [3] 邻区&切换优化 -> 以 Intra HO 成功率分类为示例 ===")

    # 定义 Intra HO 成功率
    df["IntraHO_SuccessRate"] = df["MM.HoExeIntraSucc"] / (df["MM.HoExeIntraReq"] + 1e-9)
    # 设定一个简单阈值，比如 > 0.8 则 label=1(成功率高)，否则=0(成功率低)
    df["IntraHO_label"] = df["IntraHO_SuccessRate"].apply(lambda x: 1 if x >= 0.8 else 0)

    features = ["RRU.PrbUsedDl", "RRC.ConnMean", "PEE.AvgPower"]  # 举例
    target = "IntraHO_label"

    df_sub = df.dropna(subset=features + [target]).copy()
    if len(df_sub) < 2:
        print("数据量太少，无法分类演示")
        return

    X = df_sub[features]
    y = df_sub[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("分类结果报告：")
    print(classification_report(y_test, y_pred, digits=4))
    print("----------------------------------\n")


############################
# 问题4：基站 MIMO/波束管理评估 (示例：回归预测 RRU.MaxLayerDlMimo)
############################
def solve_mimo_beam(df):
    """
    基于已有指标(吞吐、PRB使用率等)，预测或分析 MaxLayerDlMimo。
    如果该列在示例数据里都是1，分类回归都意义不大，但这里做个演示。
    """
    print("=== [4] MIMO/波束管理 -> 回归预测最大下行MIMO层数 ===")

    features = ["DRB.UEThpDl", "RRU.PrbUsedDl", "RRC.ConnMean"]
    target = "RRU.MaxLayerDlMimo"

    df_sub = df.dropna(subset=features + [target]).copy()
    if len(df_sub) < 2:
        print("数据量太少，无法回归演示")
        return

    X = df_sub[features]
    y = df_sub[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    print("预测值(部分):", y_pred[:5])
    print("----------------------------------\n")


############################
# 问题5：RRC连接与用户体验 (示例：分类判断网络拒绝是否发生)
############################
def solve_rrc_connection(df):
    """
    根据 RRC.ConnEstabAtt/ConnEstabSucc 等字段做简单分析。
    这里演示如何用“NetworkReject > 0”当作是否存在网络拒绝的分类目标。
    """
    print("=== [5] RRC连接 -> 分类判断是否出现网络拒绝 ===")

    target = "RRC.ConnEstabFailCause.NetworkReject"

    features = ["RRC.ConnEstabAtt.mo-Data", "RRC.ConnEstabSucc.mo-Data",
                "RRC.ConnEstabAtt.mo-VoiceCall", "RRC.ConnEstabSucc.mo-VoiceCall"]

    df_sub = df.dropna(subset=features + [target]).copy()
    if len(df_sub) < 2:
        print("数据量太少，无法分类演示")
        return

    # 我们定义: 如果 NetworkReject>0，label=1(有网络拒绝)；否则=0(无)
    df_sub["Reject_label"] = df_sub[target].apply(lambda x: 1 if x > 0 else 0)
    X = df_sub[features]
    y = df_sub["Reject_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("分类结果报告：")
    print(classification_report(y_test, y_pred, digits=4))
    print("----------------------------------\n")


############################
# 问题6：PCI/天线方向 & 邻区配置分析 (示例：KMeans聚类)
############################
def solve_pci_antenna_neighbor(df):
    """
    根据天线方位/PCI/邻区数量(等简单指标)做聚类示例。
    在真实场景下，可能结合地理坐标做更深入的空间分析。
    """
    print("=== [6] PCI/天线&邻区分析 -> KMeans聚类示例 ===")

    # 简单取天线方位、PCI、邻区个数(可能从NeighborX.CellId是否为0来计算)
    # 这里示例用 'Viavi.Radio.azimuth', 'Viavi.NrPci' 两列做聚类
    features = ["Viavi.Radio.azimuth", "Viavi.NrPci"]

    # 如果需要邻区数量，可数一下 _Viavi.NeighborX.CellId 不为0的个数，但此处仅演示
    df_sub = df.dropna(subset=features).copy()
    if len(df_sub) < 2:
        print("数据量太少，无法聚类演示")
        return

    X = df_sub[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42)
    df_sub["cluster"] = kmeans.fit_predict(X_scaled)

    print(df_sub[["Viavi.Radio.azimuth", "Viavi.NrPci", "cluster"]].head())
    print("----------------------------------\n")


############################
# 问题7：整体KPI & QoS预测 (示例：回归预测 Viavi.QoS.Score)
############################
def solve_kpi_qos(df):
    """
    根据多种特征(吞吐, 连接数, 功率等)回归预测 Viavi.QoS.Score。
    """
    print("=== [7] 综合KPI & QoS -> 回归预测QoS.Score ===")

    features = ["DRB.UEThpDl", "DRB.UEThpUl", "RRC.ConnMean", "PEE.AvgPower"]
    target = "Viavi.QoS.Score"

    df_sub = df.dropna(subset=features + [target]).copy()
    if len(df_sub) < 2:
        print("数据量太少，无法回归演示")
        return

    X = df_sub[features]
    y = df_sub[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    print("预测值(部分):", y_pred[:5])
    print("----------------------------------\n")


    ############################
    # 主流程：根据用户输入调用对应函数
    ############################
# def solve_problem(problem_id, df):
#     """
#     问题对应关系：
#       1 -> 容量与负载管理
#       2 -> 能耗与能效分析
#       3 -> 邻区关系和切换优化
#       4 -> 基站MIMO/波束管理
#       5 -> RRC连接分析
#       6 -> PCI/天线方向 & 邻区
#       7 -> 综合KPI & QoS预测
#     """
#     if problem_id == 1:
#         solve_capacity_load(df)
#     elif problem_id == 2:
#         solve_energy_efficiency(df)
#     elif problem_id == 3:
#         solve_handover_optimization(df)
#     elif problem_id == 4:
#         solve_mimo_beam(df)
#     elif problem_id == 5:
#         solve_rrc_connection(df)
#     elif problem_id == 6:
#         solve_pci_antenna_neighbor(df)
#     elif problem_id == 7:
#         solve_kpi_qos(df)
#     else:
#         print("无效的问题 ID，请输入 1~7")

def solve_problem(problem_id, df, *, to_str: bool = False):
    """
    参数
    ----
    problem_id : 1‑7 之间的整数，见下方映射
    df         : 预先加载好的 DataFrame
    to_str     : True  → 捕获函数内部所有 print 并以字符串返回
                 False → 维持原有行为，直接向控制台打印

    返回
    ----
    * to_str=False → None
    * to_str=True  → str  （所有打印内容）
    """

    # ========== 1) 根据 need_capture 判断是否重定向 stdout ==========
    if to_str:
        buf = io.StringIO()
        ctx = contextlib.redirect_stdout(buf)
        ctx.__enter__()          # 开始重定向

    try:
        # ========== 2) 原有业务逻辑保持不变 ==========
        if problem_id == 1:
            solve_capacity_load(df)
        elif problem_id == 2:
            solve_energy_efficiency(df)
        elif problem_id == 3:
            solve_handover_optimization(df)
        elif problem_id == 4:
            solve_mimo_beam(df)
        elif problem_id == 5:
            solve_rrc_connection(df)
        elif problem_id == 6:
            solve_pci_antenna_neighbor(df)
        elif problem_id == 7:
            solve_kpi_qos(df)
        else:
            print("无效的问题 ID，请输入 1~7")

    finally:
        # ========== 3) 如果需要，关闭重定向并返回字符串 ==========
        if to_str:
            ctx.__exit__(None, None, None)   # 恢复 stdout
            return buf.getvalue()            # 返回捕获到的所有文本


############################
# 脚本入口
############################
if __name__ == "__main__":
    data_file = "data/CellReports.csv"  # 修改为你的实际文件路径
    df = load_data(data_file)

    print("请选择要解决的问题 ID：")
    print(" 1 -> 容量与负载管理")
    print(" 2 -> 能耗与能效分析")
    print(" 3 -> 邻区关系和切换优化")
    print(" 4 -> 基站 MIMO/波束管理")
    print(" 5 -> RRC连接分析")
    print(" 6 -> PCI/天线方向 & 邻区分析")
    print(" 7 -> 综合KPI & QoS预测")

    try:
        problem_id = int(input("请输入 1~7："))
        solve_problem(problem_id, df)
    except ValueError:
        print("输入无效，请输入数字 1~7。")