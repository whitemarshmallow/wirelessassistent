# upload_analyse.py
"""
网络优化 CSV 上传 → 提取关键信息 → 调用 DeepSeek(Ark) 生成建议
暴露 1 个函数:  analyse_csv_and_ask_llm(csv_path:str) -> str
"""
# upload_analyse.py  —— 仅展示需要改动的部分 ---------------------------
from openai import OpenAI            # ← 使用官方 openai‑python SDK
import os, sys, io, contextlib, codecs, pandas as pd
from volcenginesdkarkruntime import Ark           # pip install volcengine-sdk-arkruntime

# ========= 1. CSV 特征提取  =========
def _build_prompt(csv_path: str, num_records: int = 100) -> str:
    """
    读取 csv_path，抽取若干字段 → 拼装成 prompt
    （下方字段与之前示例保持一致，可按需求增删）
    """
    df = pd.read_csv(csv_path)

    # 若列缺失直接报错，让上层捕获
    need_cols = [
        'time (ms)', 'event_type', 'ue', 'cell',
        'Viavi.UE.Rsrp', 'Viavi.QoS.Score',
        'reason', 'description', 'Viavi.Geo.x', 'Viavi.Geo.y', 'Viavi.Geo.z'
    ]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 缺少列: {', '.join(miss)}")

    df = df.iloc[:num_records]

    txt = f"""你是无线网络优化专家，请基于下列 {len(df)} 条记录给出降低乒乓切换且保证 QoS 的参数调优建议
(RSRP threshold offset 目前 3 dB, Hysteresis 目前 1 dB)。

- event_type: {df['event_type'].tolist()}
- ue: {df['ue'].tolist()}
- rsrp: {df['Viavi.UE.Rsrp'].tolist()}
- qos_score: {df['Viavi.QoS.Score'].tolist()}
- reason: {df['reason'].tolist()}
- description: {df['description'].tolist()}
- pos(x,y,z): {list(zip(df['Viavi.Geo.x'], df['Viavi.Geo.y'], df['Viavi.Geo.z']))}

请输出：
1) 问题诊断  2) 参数调优建议  3) 预期成效
"""
    return txt

# ========= 2. 调用 DeepSeek / Ark =========
def _call_deepseek(prompt: str) -> str:
    """
    通过 OpenAI SDK 调 DeepSeek‑V3 / DeepSeek‑R1。
    ❶ 建议把 key 写到环境变量  DEEPSEEK_API_KEY  而不是硬编码
    ❷ 选哪种模型：  • deepseek-chat  (即 V3)
                    • deepseek-reasoner (R1 推理模型)
    """
    api_key = "sk-d9c23ab8c0944d38bc9b177748d99f87"
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY 环境变量")

    # 创建客户端（注意 base_url 指向 deepseek）
    client = OpenAI(
        api_key = api_key,
        base_url = "https://api.deepseek.com/v1"
    )

    resp = client.chat.completions.create(
        model = "deepseek-reasoner",          # 如果只需要 V3 可写 "deepseek-chat"
        messages = [
            {"role": "system", "content": "你是一位资深无线网络优化专家。"},
            {"role": "user",   "content": prompt}
        ],
        stream = False                        # 如需流式返回改为 True
    )

    return resp.choices[0].message.content.strip()

# ========= 3. 外部统一调用 =========
def analyse_csv_and_ask_llm(csv_path: str) -> str:
    """
    封装：读 CSV → 组 prompt → 请求 LLM
    出错时抛异常，让 Flask 路由统一处理
    """
    prompt = _build_prompt(csv_path)
    answer = _call_deepseek(prompt)
    return answer


# ============ 本地快速测试 ============
if __name__ == "__main__":
    test_csv = "data/MergedData_NoDuplicates.csv"   # 换成本地 CSV
    print(analyse_csv_and_ask_llm(test_csv))
