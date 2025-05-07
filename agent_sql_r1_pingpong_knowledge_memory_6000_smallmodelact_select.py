import codecs
import sys

import mysql.connector
import pandas as pd
from langchain.utils import openai
from mysql.connector import Error
from openai import OpenAI
from langchain.schema import AIMessage, HumanMessage

# from pingpong_text_information_extraction import analyze_data_and_generate_query
# 先在你的大模型脚本开头导入
from wireless_cell_test import load_data as load_cell_data, solve_problem as solve_cell_problem
from wireless_UE_test import load_data as load_ue_data, solve_problem as solve_ue_problem


# ============ 新增：导入机器学习相关库 ============
from sklearn.linear_model import LinearRegression
import joblib

#和java调用的脚本区别在于脚本返回的是字典，因此这里不能直接挪过去



class Agent:
    def __init__(
        self,
        deepseek_api_key,
        mysql_host="localhost",
        mysql_user="root",
        mysql_password="your_password",
        mysql_database="wireless_optimize_db"
    ):
        """
        1. 用 MySQL 做数据库后端。
        2. chat_history（Python 列表）存储历史；MySQL 存储长期对话。
        3. 使用新的 OpenAI 客户端(DeepSeek 兼容)。
        """

        # 1. 配置 DeepSeek (兼容OpenAI) 客户端
        self.client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1"
        )

        # 2. 连接 MySQL 数据库
        try:
            self.conn = mysql.connector.connect(
                host=mysql_host,
                user=mysql_user,
                password=mysql_password,
                database=mysql_database
            )
            self.cursor = self.conn.cursor()
            print("成功连接到 MySQL 数据库!")
        except Error as e:
            raise Exception(f"连接 MySQL 时出现错误: {e}")

        # 3. 创建对话历史表 & 知识库表（若不存在）
        self._create_tables()

        # 4. 读取已有对话、知识库
        self.chat_history = []
        self._load_long_term_memory()
        self.knowledge_dict = {}
        self._load_knowledge_base()

        # =========== act新增-加载或初始化 RSRP 模型 ============
        try:
            self.rsrp_model = joblib.load('rsrp_model.pkl')
            print("成功加载 RSRP 预测模型 (线性回归)!")
        except:
            print("尚未找到 rsrp_model.pkl，后续如需预测则需调用 train_rsrp_model 训练。")
            self.rsrp_model = None

        # ---------------- 新增训练示例，可选 ----------------
    def train_rsrp_model(self, csv_path='UEReports.csv'):
        df = pd.read_csv(csv_path)
        df['RSRP_future'] = df['Viavi.UE.Rsrp'].shift(-1)
        df.dropna(subset=['RSRP_future'], inplace=True)

        feature_cols = ['Viavi.UE.Rsrp', 'Viavi.UE.Rsrq', 'Viavi.UE.RsSinr']
        X = df[feature_cols]
        y = df['RSRP_future']

        model = LinearRegression()
        model.fit(X, y)

        joblib.dump(model, 'rsrp_model.pkl')
        self.rsrp_model = model
        print("训练并保存线性回归 RSRP 模型完毕！")


    # ========== 历史对话拼接函数 ==========
    def _build_history_messages(self):
        """
        将 self.chat_history 中的多条记录拼接成一个 user→assistant 的序列。
        对每个 assistant，都找它“相邻上一次”的 user(若存在)，一起加入。
        具体逻辑:
          - 遍历 chat_history
          - 遇到 user => 记录到 last_user
          - 遇到 assistant => 如果有 last_user 未用，就先放那条 user，然后放 assistant
            然后清空 last_user
        这样，如果 user 没有后续 assistant，就不加入；assistant 没有前面 user，也直接加入assistant。
        """
        final_msgs = []
        last_user = None

        for entry in self.chat_history:
            if entry["role"] == "user":
                # 暂存这个 user，等待后面若出现assistant就配对
                last_user = entry
            elif entry["role"] == "assistant":
                # 如果有尚未匹配的 user，就先加上
                if last_user:
                    final_msgs.append({
                        "role": "user",
                        "content": last_user["content"]
                    })
                    last_user = None
                # 再加上当前的 assistant
                final_msgs.append({
                    "role": "assistant",
                    "content": entry["content"]
                })
            else:
                # 如果以后你有 system 之类，这里自行处理
                pass

        return final_msgs

    def _load_knowledge_base(self):
        try:
            query = "SELECT `key`, `value` FROM knowledge_base"
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            for k, v in rows:
                self.knowledge_dict[k] = v
            print("知识库加载完毕:", self.knowledge_dict)
        except Error as e:
            print("加载知识库时出错:", e)

    def _load_long_term_memory(self):
        try:
            select_sql = "SELECT role, message FROM conversation_history"
            self.cursor.execute(select_sql)
            rows = self.cursor.fetchall()
            for role, msg in rows:
                self.chat_history.append({"role": role, "content": msg})
        except Error as e:
            print("加载长期对话历史时出错:", e)

    def _create_tables(self):
        try:
            conversation_table = """
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                role VARCHAR(20) NOT NULL,
                message TEXT NOT NULL
            );
            """
            self.cursor.execute(conversation_table)

            knowledge_base_table = """
            CREATE TABLE IF NOT EXISTS knowledge_base (
                `key`   VARCHAR(100) PRIMARY KEY,
                `value` TEXT
            );
            """
            self.cursor.execute(knowledge_base_table)

            # 插入示例知识
            insert_demo_kv = """
            INSERT IGNORE INTO knowledge_base (`key`, `value`)
            VALUES (%s, %s)
            """
            # self.cursor.execute(insert_demo_kv, ("乒乓切换", "含义是用户在小区间的切换"))
            self.conn.commit()

        except Error as e:
            print("创建表时出现错误:", e)

    def _store_message(self, role, message):
        try:
            insert_sql = "INSERT INTO conversation_history (role, message) VALUES (%s, %s)"
            self.cursor.execute(insert_sql, (role, message))
            self.conn.commit()
        except Error as e:
            print("存储对话消息时出错:", e)

    def _combine_with_knowledge(self, base_messages, user_input):
        messages = list(base_messages)
        matched_knowledge = {}

        try:
            select_sql = "SELECT `key`, `value` FROM knowledge_base"
            self.cursor.execute(select_sql)
            rows = self.cursor.fetchall()

            for (db_key, db_value) in rows:
                # 让 user_input 去匹配 db_key
                if user_input.lower() in db_key.lower():
                    matched_knowledge[db_key] = db_value

        except Exception as e:
            print(f"加载 MySQL 知识库时出错: {e}")

        # 同理对本地 self.knowledge_dict 做相同处理
        for k, v in self.knowledge_dict.items():
            if user_input.lower() in k.lower():
                if k not in matched_knowledge:
                    matched_knowledge[k] = v

        print("匹配到的知识 key:", matched_knowledge.keys())

        if matched_knowledge:
            knowledge_text = "【知识库匹配内容】\n"
            for mk, mv in matched_knowledge.items():
                knowledge_text += f" - {mk}: {mv}\n"

            knowledge_message = {
                "role": "system",
                "content": knowledge_text
            }
            messages.insert(0, knowledge_message)
            self.chat_history.append(knowledge_message)

        return messages


    def _keep_system_and_last_two_pairs(self, messages):
        """
        保留所有 system 消息(按出现顺序)；
        然后在 normal_msgs (user/assistant) 中，从后往前找最多2组 (user->assistant) 对。
        如果找不到相邻的 user，就跳过该 assistant，从而避免“assistant”脱离上下文。
        最后拼装并返回：system 消息 + 收集到的对话对（保持原有顺序）。
        """
        # 1. 分离 system 与普通消息
        system_msgs = []
        normal_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system_msgs.append(msg)
            else:
                normal_msgs.append(msg)

        # 2. 从后往前收集最多2组 (user, assistant) 对
        pairs_collected = []
        pair_count = 0
        i = len(normal_msgs) - 1
        while i >= 0 and pair_count < 2:
            if normal_msgs[i]["role"] == "assistant":
                assistant_msg = normal_msgs[i]
                i -= 1
                # 查找与它相邻的 user 消息
                if i >= 0 and normal_msgs[i]["role"] == "user":
                    user_msg = normal_msgs[i]
                    i -= 1
                else:
                    # 如果前一条不是 user，则跳过这个 assistant
                    continue

                # 插入这组对（保持正序：先 user 后 assistant）
                pairs_collected.insert(0, user_msg)
                pairs_collected.insert(1, assistant_msg)
                pair_count += 1
            else:
                i -= 1

        # 3. 拼装：先 system 消息，再加上 normal 的最近对话
        final = system_msgs + pairs_collected
        return final


    def analyze_data_and_generate_query(self,csv_file_path):
        sys.stdout = sys.stdout.detach()
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

        num_records = 200

        # 读取CSV文件
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            return f"Error reading CSV file: {e}"

        # 检查需要的列是否存在
        required_columns = ['time (ms)', 'initiator', 'event_type', 'ue', 'slice', 'cell', 'o1_dn',
                            'reason', 'description', 'x', 'y', 'z']
        for column in required_columns:
            if column not in df.columns:
                return f"Missing required column: {column}"

        # 提取与乒乓切换相关的特征
        time_data = df['time (ms)'].iloc[0:num_records].tolist()  # 提取前num_records条时间数据
        initiator_data = df['initiator'].iloc[0:num_records].tolist()
        event_type_data = df['event_type'].iloc[0:num_records].tolist()
        ue_data = df['ue'].iloc[0:num_records].tolist()
        cell_data = df['cell'].iloc[0:num_records].tolist()
        reason_data = df['reason'].iloc[0:num_records].tolist()
        description_data = df['description'].iloc[0:num_records].tolist()
        x_data = df['x'].iloc[0:num_records].tolist()
        y_data = df['y'].iloc[0:num_records].tolist()
        z_data = df['z'].iloc[0:num_records].tolist()

        # 生成分析文本
        query = f"""
        deepseek，有一个无线通信涉及乒乓切换的网络优化问题需要你帮忙解决看看

        1. 乒乓切换分析:
        - 事件类型（event_type）为: {event_type_data}，显示切换事件的类型。
        - 用户设备（ue）为: {ue_data}，是涉及切换的用户设备标识。
        - 切换原因（reason）为: {reason_data}，展示触发切换的原因。
        - 切换描述（description）为: {description_data}，显示切换过程的详细描述。

        2. 位置信息:
        - 设备坐标（x, y, z）为: {list(zip(x_data, y_data, z_data))}，显示设备的物理位置。

        请根据以上数据进行网络优化建议，特别是在降低乒乓切换频率、优化切换阈值方面。
        """
        return query

    def call_wireless_model(self, model_side: str, problem_id: int, data_file: str) -> str:
        """
        根据 model_side ('cell' or 'ue'), problem_id (1~7), data_file 路径，
        调用对应的小模型脚本 (wireless_cell_test 或 wireless_UE_test)，
        加载数据并运行 solve_problem，然后把结果以字符串形式返回。

        注：小模型本身大多是直接 print 输出结果。若想获取更详细的结构化结果，
        可以考虑在小模型里改造把结果 return 出来；这里先演示简化做法。
        """
        import io
        import sys

        old_stdout = sys.stdout
        # 截获小模型的控制台输出
        mystdout = io.StringIO()
        sys.stdout = mystdout

        try:
            # 加载数据 + 调用小模型
            if model_side == "cell":
                df = load_cell_data(data_file)
                solve_cell_problem(problem_id, df)
            elif model_side == "ue":
                df = load_ue_data(data_file)
                solve_ue_problem(problem_id, df)
            else:
                print(f"未知的 model_side={model_side}, 只能是 'cell' 或 'ue'")
        except Exception as e:
            print(f"调用小模型出错: {e}")

        # 恢复stdout
        sys.stdout = old_stdout
        # 获取小模型产生的全部输出
        result_text = mystdout.getvalue()
        mystdout.close()

        return result_text

    def think(self, user_input):
        """
        改进需求：
        1. 如果用户没有明确请求使用“大模型”，即使识别到“乒乓切换”，也不调用 deepseek；
           只有检测到用户输入中包含类似“请问大模型”/“使用大模型”/“deepseek分析”等关键字时，才调用 deepseek。
        2. 知识库根据多个关键词加载，而不仅是“乒乓切换”。
        其他功能（记忆库、对话上下文）保持不变。
        """

        # ========== 1. 避免连续 user ==========
        if len(self.chat_history) > 0 and self.chat_history[-1]["role"] == "user":
            self.chat_history[-1]["content"] += f"\n{user_input}"
        else:
            self.chat_history.append({"role": "user", "content": user_input})

        # ========== 2. 从对话历史构建 base_messages ==========
        base_messages = self._build_history_messages()

        print("初始的base_messages")
        for i, msg in enumerate(base_messages):
            role_str = msg["role"]
            content_str = msg["content"]
            print(f"  [{i}] role={role_str} => {content_str!r} ...")

        # 记录本轮用户输入
        final_user_msg = user_input
        # 存放分析结果 & 小模型结果 等
        extra_system_info = ""

        # ========== 3. 若输入包含"乒乓切换"，做 CSV 分析 & RSRP预测，但不一定调用大模型 ==========
        if "乒乓切换" in user_input or "pingpong" in user_input:
            print(" 检测到乒乓切换关键词，进行CSV分析 & RSRP预测")
            csv_file_path = "MergedData_NoDuplicates.csv"
            analyze_text = self.analyze_data_and_generate_query(csv_file_path)
            extra_system_info += analyze_text + "\n"


        else:
            print("未检测到乒乓切换关键词，不做相关分析")


        # ========== 4. 根据用户输入匹配“基站侧”或“UE侧”小模型 & 对应问题ID ==========
        model_side = None

        if "基站" in user_input or "cell side" in user_input:
            model_side = "cell"
        elif "UE" in user_input or "用户" in user_input:
            model_side = "ue"

        # 两个映射表
        cell_problem_map = {
            "容量与负载管理": 1,
            "能耗与能效分析": 2,
            "邻区关系和切换优化": 3,
            "基站MIMO": 4,
            "RRC连接分析": 5,
            "PCI/天线方向": 6,
            "综合KPI": 7
        }
        ue_problem_map = {
            "覆盖与性能评估": 1,
            "网络容量与吞吐分析": 2,
            "QoS与切片管理": 3,
            "异常检测与网络故障排查": 4,
            "基于位置的业务体验分析": 5,
            "Massive MIMO与波束管理评估": 6,
            "TA与距离管理": 7
        }

        problem_id = None
        if model_side == "cell":
            for key, val in cell_problem_map.items():
                if key in user_input:
                    problem_id = val
                    break
        elif model_side == "ue":
            for key, val in ue_problem_map.items():
                if key in user_input:
                    problem_id = val
                    break

        if model_side and problem_id:
            # 不一定要调用大模型(Deepseek), 但是可以调用你的小模型
            data_file = "CellReports.csv" if model_side == "cell" else "UEReports.csv"
            result_text = self.call_wireless_model(model_side, problem_id, data_file)
            extra_system_info += f"\n[小模型({model_side} - 问题{problem_id})输出]\n{result_text}\n"
        else:
            print("[提示] 未匹配到小模型调用条件，或未识别出 problem_id")

        # ========== 5. 知识库注入 (通过 _combine_with_knowledge) ==========
        # 这里我们要让不同关键词去触发相应的知识库内容:
        # 你可以在 _combine_with_knowledge 里扩展, 或这里自己处理
        final_messages = self._combine_with_knowledge(base_messages, user_input)

        # ========== 6. 保留 system + 最近2对对话 ==========
        final_messages = self._keep_system_and_last_two_pairs(final_messages)

        # ========== 7. 将本轮 user_input 追加到最后(保证最后一条一定是 user) ==========
        if len(final_messages) == 0:
            final_messages.append({"role": "user", "content": final_user_msg})
        else:
            last_role = final_messages[-1]["role"]
            if last_role == "assistant":
                final_messages.append({"role": "user", "content": final_user_msg})
            elif last_role == "user":
                final_messages[-1]["content"] = final_user_msg
            else:
                final_messages.append({"role": "user", "content": final_user_msg})

        # 如果有额外信息 => system 提示
        if extra_system_info.strip():
            final_messages.insert(0, {
                "role": "system",
                "content": extra_system_info
            })

        print("[调试] extra_system_info:", extra_system_info)

        # ========== 8. 判断是否调用大模型 deepseek ==========
        # 如果用户输入中包含 "使用大模型" 或 "请问deepseek" 或 "deepseek分析" 等，则进行调用
        # 如果没有，就直接返回 extra_system_info 或做其他处理
        call_big_model = False
        if ("使用大模型" in user_input
                or "deepseek" in user_input
                or "大模型" in user_input
                or "analysis by big model" in user_input.lower()):
            call_big_model = True

        if not call_big_model:
            print(" 用户未请求大模型分析，故不调用 deepseek，直接返回小模型/乒乓切换结果")
            # 你可以直接把 extra_system_info 返回给前端，或者包装一下
            return "【不调用大模型】\n" + extra_system_info.strip()

        # ========== 如果需要大模型 => 调用 deepseek reasoner ==========
        print("调用 deepseek-reasoner 的 final_messages:")
        for i, msg in enumerate(final_messages):
            role_str = msg["role"]
            content_str = msg["content"]
            print(f"  [{i}] role={role_str} => {content_str!r} ...")

        #和脚本的区别在于脚本返回的是字典，因此这里不能直接挪过去



        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=final_messages
        )
        return response.choices[0].message.content.strip()

    def respond(self, user_input):
        """
        respond() 只需要:
          - 存储用户输入
          - 调用 think()
          - 存储大模型回答
          - 返回回答
        """
        self._store_message("user", user_input)
        self.chat_history.append({"role": "user", "content": user_input})

        # 大模型思考 => 得到答案
        answer = self.think(user_input)

        self._store_message("assistant", answer)
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer


# ===== 测试用例 =====
if __name__ == "__main__":
    agent = Agent(
        deepseek_api_key="sk-d9c23ab8c0944d38bc9b177748d99f87",
        mysql_host="localhost",
        mysql_user="root",
        mysql_password="1234",
        # mysql_database="wireless_optimize_db2"
        mysql_database = "wireless_optimize_db"
    )

    question1 = input("请输入您的需求: ")
    reply1 = agent.respond(question1)
    print("助手:", reply1)
