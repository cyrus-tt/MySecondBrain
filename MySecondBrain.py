__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ------------ 分割线 ------------
# 上面这三行必须在最前面！
# 下面才是其他的 import

import streamlit as st
import os
import chromadb
# ... 后面的代码import streamlit as st
import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import json
# 引入之前写好的 ingest 逻辑 (假设你把 universal_ingest.py 里的函数封装好了)
# 为了演示方便，这里我们直接简化写在里面
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# 2. 页面设置
st.set_page_config(page_title="My Second Brain", layout="wide")
st.title("🧠 我的第二大脑 (Agent版)")

# 3. 初始化 (缓存资源)
@st.cache_resource
def init_system():
    # A. 连 DeepSeek
    api_key = st.secrets["DEEPSEEK_API_KEY"]
    client = OpenAI(
        api_key=api_key, 
        base_url="https://api.deepseek.com"
    )
    # B. 连数据库
    chroma_client = chromadb.PersistentClient(path="./my_company_data")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    collection = chroma_client.get_or_create_collection(name="second_brain", embedding_function=ef)
    return client, collection

client, collection = init_system()

# 5. 侧边栏：知识投喂站 (修改版)
with st.sidebar:
    st.header("📂 知识投喂")
    uploaded_file = st.file_uploader("上传 TXT 资料", type=["txt"])
    
    if uploaded_file and st.button("吃掉它！"):
        # 1. 读取并存库
        text = uploaded_file.read().decode("utf-8")
        collection.add(
            documents=[text],
            ids=[uploaded_file.name]
        )
        st.success(f"已吞噬: {uploaded_file.name}")
        
        # 2. 【关键修改】存完之后，往聊天记录里塞一条“系统通知”
        # 这样 AI 就知道：“哦，原来刚才存了个文件，那用户问的时候我要去查库。”
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"✅ 我已经学习了文件 **{uploaded_file.name}** 的内容。现在你可以问我关于它的问题了！"
        })
        
        # 强制刷新页面，让这句话立刻显示出来
        st.rerun()

# 5. 定义 Agent 工具函数
def search_knowledge(query):
    results = collection.query(query_texts=[query], n_results=1)
    if not results['documents'][0]:
        return "数据库里没有相关信息。"
    return results['documents'][0][0]

def save_file(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"文件已保存: {filename}"

# 工具 Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "查数据库",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": "保存文件",
            "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "content": {"type": "string"}}, "required": ["filename", "content"]}
        }
    }
]

# 6. 聊天主逻辑
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for msg in st.session_state.messages:
    if msg["role"] != "tool": # 不显示工具调用的中间杂音
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 处理输入
if prompt := st.chat_input("问我任何事，或者让我帮你总结保存..."):
    # 显示用户输入
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 构造请求 (带上历史记录，防止失忆)
    # 注意：真实项目中要处理 messages 格式，这里简化只发最近几条
    api_messages = [
        {"role": "system", "content": "你是一个智能助手。请按需使用工具。"}
    ] + [m for m in st.session_state.messages if m["role"] != "tool"] # 简化处理

    # 第一轮调用
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=api_messages,
        tools=tools
    )
    msg = response.choices[0].message
    
    # 如果 AI 要调工具
    if msg.tool_calls:
        # 显示 AI 正在思考的动效
        with st.chat_message("assistant"):
            st.markdown("⚙️ 正在调用工具处理...")
            
        for tool_call in msg.tool_calls:
            fname = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            tool_result = ""
            if fname == "search_knowledge":
                tool_result = search_knowledge(args["query"])
            elif fname == "save_file":
                tool_result = save_file(args["filename"], args["content"])
            
            # 把工具结果塞回给 AI (这里简化处理，不存入 session_state 显示给用户看)
            api_messages.append(msg)
            api_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })
            
        # 第二轮调用：生成最终回答
        final_resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=api_messages
        )
        ai_reply = final_resp.choices[0].message.content
        
        st.chat_message("assistant").markdown(ai_reply)
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        
    else:
        # 不需要工具，直接回
        st.chat_message("assistant").markdown(msg.content)
        st.session_state.messages.append({"role": "assistant", "content": msg.content})
