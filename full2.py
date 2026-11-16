# 加载基础库
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # Windows 本地环境没有 pysqlite3，继续使用系统自带的 sqlite3
    pass
import streamlit as st
# 导入 `streamlit` 库，并起别名 `st`
# Streamlit，Python 快速 Web 可视化框架，不需要写 HTML/JS，只用几行 Python 就能生成网页界面、输入框、聊天框、按钮，用来做这个 RAG 应用的前端页面。
# 安装命令：`pip install streamlit`
import re
import os
# 导入 Python 内置`os`模块
# 作用：操作系统接口，一般用来读取环境变量（存放 OpenAI Key）、读写本地文件、路径处理。属于内置库，不用 pip 安装。
import fitz 
# `fitz` 是PyMuPDF库的导入名
# 读取、解析 PDF 文档，提取 PDF 里的文字，是 RAG 流程里【文档加载器】
import chromadb
# 导入 `chromadb` 向量数据库
# ChromaDB 是轻量开源向量数据库，RAG 核心组件：把文档文本转成向量、持久化存储、做相似度检索。用户提问时，从库里检索相关片段，再交给大模型回答。
# 安装命令：pip install chromadb
import uuid
# 导入 Python 内置`uuid`模块
# 用来生成全局唯一 ID。往 ChromaDB 新增文档切片的时候，每个切片都需要一个不重复的`id`，一般用 `uuid.uuid4()` 生成随机唯一标识。
# 内置库，无需安装。
from openai import OpenAI
# 从 `openai` 包里面导入 `OpenAI` 这个客户端类
# 用来创建客户端，调用 GPT 对话接口、Embedding 向量化接口。

# ================= 1. 配置你的 API =================
client = OpenAI(
    # 只要你想用 OpenAI 的 SDK（openai 这个库）去调用它的接口（embedding、chat.completions、图像等）
    api_key = st.secrets.get("DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY")),
    base_url="https://ws-guf27zl5m8zvagh9.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

# ================= 2. 计算向量的工具函数 =================
# 批量调用 OpenAI Embedding 接口
# 把一批文本转成向量（embedding），是 RAG 里【文本向量化】的核心工具，配合前面 chromadb 使用。
def get_embeddings(texts, batch_size=10):
# `batch_size=10`：批次大小，默认一次最多发给 OpenAI 10 条文本；
# OpenAI 接口支持批量，分批调用可以减少请求次数、提升效率，同时避免单次请求超长超限。
    all_embeddings = []
    # 初始化空列表，用来**收集全部文本对应的向量**，最后把完整向量列表返回。
    for i in range(0, len(texts), batch_size):
        # 把长文本列表切分成一小批一小批，分批请求接口。
        batch = texts[i:i+batch_size]
        # 切片，取出当前这一批文本，`batch` 就是本轮要向量化的文本数组。
        try:
            # `try-except` 异常捕获块：防止接口报错（网络、密钥错误、限流、文本过长）直接导致整个程序崩溃；出错时进入 except 分支。
            response = client.embeddings.create(
# 服务器处理完，会返回一大包结构化数据（包含每条文本对应的 embedding 向量、使用 token 数等）
# 把这包结果赋值给 `response`，后面我们从里面把向量取出来。
                model="tongyi-embedding-vision-plus-2026-03-06",
                input=batch,
                # `input` 是接口规定的参数，接收你要转向量的文本；这里传的就是上一句切块得到的 `batch`（一小批文本列表），刚好对应我们分批循环的逻辑。
                encoding_format="float"
                # `encoding_format="float"`：指定返回向量格式为浮点数数组，chromadb 默认接收这种格式
            )
            all_embeddings.extend([data.embedding for data in response.data])
            # data.embedding for data in response.dreata：列表推导式（list comprehension），是 Python 简写循环、快速生成列表的写法。
            # 即从response.data之中取出data，然后将data的值交给前面的data.embedding表达式，执行赋予向量操作
            # `extend()`：把本轮向量追加到总列表
        except Exception as e:
            # 只要 try 里面任意代码抛出异常，就执行这里
            st.error(f"计算向量失败: {e}")
            # `st.error()`：streamlit 组件，在网页上弹出红色报错提示，展示错误信息 e
            return None
        # 写了 `return None` = 明确传递语义：只要这里报错，立刻终止整个向量化流程，函数直接返回失败标记，不再继续处理剩下的批次
    return all_embeddings
# 所有批次全部请求完成后，返回完整的向量列表

# ================= 3. 初始化 ChromaDB =================
if "collection" not in st.session_state:
    # 不加这个判断，每次交互都会重复创建数据库集合，引发报错、数据丢失。
    chroma_client = chromadb.Client()
    # 如果想要持久化保存，要写成 `chromadb.PersistentClient(path="./chroma_db")`，数据会存在本地文件夹
    st.session_state.collection = chroma_client.get_or_create_collection(name="pdf_notes")
    # collection 是 ChromaDB 的核心单元：用来存放文本、文本向量（embedding）、元数据 metadata；
    # RAG 里一般一个业务文档库对应一个 collection
collection = st.session_state.collection
# 从会话状态取出之前保存好的向量集合，赋值给局部变量 `collection`
# 方便后面代码直接写 `collection.add()`、`collection.query()` 来存入文本、做向量检索。

# ================= 4. 优化的切块函数（按段落合并） =================
def split_text_by_paragraph(text, max_chunk_size=600):
    # RAG 里的文本分块（chunking）逻辑，属于「按段落优先合并」策略
    # 目标：尽量不把一个完整段落拦腰切断，同时控制每一块文本长度上限
    """按段落切分，保证一句话不会被生硬切断"""
    paragraphs = text.split('\n')
    # 原理：原始文本里换行的地方，就认为是段落分隔。
    chunks = []
    # 最终结果列表，用来存放一个个切好的文本块（chunk）
    current_chunk = ""
    # 正在组装、还没存进结果的临时块，用来不断累加段落，直到快要触达长度上限
    
    for para in paragraphs:
        # 循环遍历上面拆分出来的每一个段落 `para`
        para = para.strip()
        # `.strip()`：去掉段落首尾的空格、`\t`、换行；清理脏空白
        if not para:
            # `if not para:`：判断清理之后这个段落是空字符串（原来就是空行）
            continue
        # `continue`：直接跳过，不把空行加入块里，避免很多无用空白文本
        # 如果加上这一段没超过限制，就合并
        if len(current_chunk) + len(para) <= max_chunk_size:
            # 临时块现有长度 + 当前这个段落的长度 ≤ 最大上限
            current_chunk += para + "\n"
            # 把这个段落拼进 `current_chunk`，末尾加换行符保留段落分隔感
        else:
            # 超过限制，保存当前的，重新开一段
            if current_chunk:
                # 先判断 `current_chunk` 不为空
                chunks.append(current_chunk)
                # 把已经组装好的这块，存入结果列表 `chunks`
            current_chunk = para + "\n"
            # 重置 `current_chunk`：把当前这个超长段落，作为新块的起点
    
    if current_chunk:
        # 循环全部段落跑完之后，最后还有一块没放进列表的 current_chunk，收尾存入 chunks
# for 循环里只有触发 else 的时候才 append；循环结束最后一块不会自动保存，必须手动收尾
        chunks.append(current_chunk)
    return chunks
# 返回切分完成的所有文本块列表，后续就可以循环这个列表，向 ChromaDB add 文本 + 向量

# ================= 5. 页面布局 =================
st.title("ScholarRAG:学术级文献解析系统")
# `st.title()`：Streamlit 的页面大标题，渲染在网页主内容区最上方。

with st.sidebar:
# Streamlit 封装好了 sidebar 的上下文，进入 `with`，切换渲染区域；退出，切回主区域。
    st.header("文档上传区")
    # `st.header()`：侧边栏内二级标题。
    uploaded_files = st.file_uploader("请上传PDF文件（支持多选）", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
    # 第三步（上）：生成当前文件的专属签名
        current_file_names = [f.name for f in uploaded_files]
        signature = "".join(current_file_names)
    
        if st.button("解析并存入知识库"):
        
        # 第三步（下）：判断这次上传的文件是否和上次一样
            if st.session_state.get("last_uploaded_files") != signature:
            
            # 第二步：遍历所有文件
                for file in uploaded_files:
                
                    with st.spinner(f"正在解析 {file.name}..."):
                        pdf_bytes = file.read()
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    
                        all_chunks = []
                        all_metadatas = []
                        all_ids = []
                    
                        for page_num, page in enumerate(doc):
                            page_text = page.get_text()
                            page_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。；：“”《》、（）\n]', ' ', page_text)
                            if not page_text.strip():
                                continue
                            
                            chunks = split_text_by_paragraph(page_text)
                        
                            for chunk in chunks:
                                all_chunks.append(chunk)
                                all_metadatas.append({"source": file.name, "page": page_num + 1})
                                all_ids.append(str(uuid.uuid4()))
                    
                        if all_chunks:
                            with st.spinner(f"正在计算 {file.name} 的向量..."):
                                embeddings = get_embeddings(all_chunks)
                            
                                if embeddings:
                                    collection.add(
                                        documents=all_chunks,
                                        embeddings=embeddings,
                                        metadatas=all_metadatas,
                                        ids=all_ids
                                    )
                                    st.success(f"✅ {file.name} 成功存入 {len(all_chunks)} 个文本块！")
                                else:
                                    st.error(f"❌ {file.name} 向量化失败。")
                        else:
                            st.error(f"❌ {file.name} 未提取到文字。")
            
            # 所有文件循环解析完后，记录下这次的签名
                st.session_state["last_uploaded_files"] = signature
                st.success("🎉 所有文档解析完成！您可以开始提问了。")
            
            else:
            # 如果签名一样，说明已经点过了，直接提示
                st.info("这批文件已经解析过啦，无需重复解析。请直接在下方提问。")

# ================= 6. 聊天界面 =================
# **问答交互入口**：保存对话历史、用户提问、向量检索，检索到上下文之后，下一步就是拼 prompt 喂给 LLM 生成答案。
if "messages" not in st.session_state:
    # `st.session_state`：会话状态，只要浏览器标签不关，里面的数据跨页面 rerun 不会丢失
    # 如果会话里还没有 `messages` 这个键，就新建一个空列表；`messages` 用来存一轮一轮的聊天记录
    st.session_state.messages = []

# 页面渲染历史对话：循环读取已经保存的聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # `st.chat_message(role)`：Streamlit 的气泡组件
        st.markdown(message["content"])

if prompt_text := st.chat_input("请输入关于该 PDF 的问题："):
# - 用户没输入：`prompt_text` 是空，不进这个分支
# - 用户输入并回车：拿到用户的问题字符串，进入 if 内部逻辑

    with st.chat_message("user"):
        st.markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    # # 2. 把这条用户消息追加进 `messages` 对话历史列表，永久保存到会话里

    with st.chat_message("assistant"):
        # - 提前打开助手的气泡框
        with st.spinner("正在翻阅论文..."):
            # 加载动画：提示用户正在检索 PDF 知识库（向量检索阶段，还没调用 LLM）
            
            # 1. 向量检索：捞取最相关的 4 段
            #前面的要用PDF chunk 文字转向量；现在**用户问题也用同一个 embedding 模型，转到同一个向量空间**，才能做相似度比对
            query_embedding = get_embeddings([prompt_text])[0]
# 所以要把单个问题包成列表 `[prompt_text]`
# 返回是 `[向量1]`，取 `[0]` 拿到这一条问题对应的一维向量
            results = collection.query(
                # 返回值 `results` 是一个嵌套字典，里面包含：匹配到的文本、metadata（文件名、页码）、距离分数等
                # `collection.query()` 是 ChromaDB 的相似度检索 API
                query_embeddings=[query_embedding],
                # `query_embeddings`：传入向量列表（所以外面套一层`[]`）
                n_results=4
                # `n_results=4`：检索返回相似度最高的**4 个 chunk**
            )
            
            retrieved_context = ""
            # 准备拼接检索出来的原文片段，作为给 LLM 的参考上下文。
            if results['documents'] and results['documents'][0]:
                # Chroma 的 `collection.query()` 返回的 results 里，`results["documents"]` 是一个双层嵌套列表
# 判断：向量检索**确实拿到了至少一条匹配到的文本片段**。防止检索为空的时候，后面循环、取内容出现异常或者无效拼接。
# 先看有没有这一层容器，再看容器里面有没有真正的片段。

# **语义相似检索**：哪怕用词不完全一样、同义改写，也能匹配出来；缺点是有时候会语义漂移，漏掉专业术语。
                for i in range(len(results['documents'][0])):
                    text = results['documents'][0][i]
                    meta = results['metadatas'][0][i]
                    retrieved_context += f"【内容片段 {i+1} (来源：{meta['source']} 第 {meta['page']} 页)】:\n{text}\n\n"
                    # 循环把向量检索 Top4 的 chunk，连同文件名、页码，格式化字符串追加到 `retrieved_context`。

            # 2. 引入关键词兜底（混合检索）：如果问题里有“问题一”，强行捞取所有带“问题一”的段落
            # 预设一组领域专业关键词。**先看用户提问里有没有包含这个关键词**，有就触发精确关键词检索。
            # 向量检索看 “意思像不像”；关键词检索看 “字有没有出现”，二者互补，形成混合检索。
            keywords = ["问题一", "MILP", "混合整数线性规划"]
            for kw in keywords:
                if kw in prompt_text:
                    # 去数据库里强行搜包含这个词的段落
                    kw_results = collection.get(where_document={"$contains": kw})
# 真正工业级混合检索一般用 BM25，而不是这种简单 `$contains`。
                    if kw_results and kw_results['documents']:
                        for idx, doc_text in enumerate(kw_results['documents'][:2]): # 最多补2段
# 目的：如果数据库里几十条都包含 MILP，只拿前两条，避免检索文本太多，token 超限、LLM 变慢、开销暴涨
                            meta = kw_results['metadatas'][idx]
                            retrieved_context += f"【🔥 关键词强制命中 (来自第 {meta['page']} 页)】:\n{doc_text}\n\n"
                            # `retrieved_context`：字符串变量，用来汇总所有检索回来的上下文（向量检索结果 + 这里关键词兜底结果）

            # 3. 组装 Prompt
            # `system_prompt`（系统提示词）：用来给 LLM 定角色、定规则。
            # 作用：在对话最开头，告诉大模型它是谁、要遵守什么规则，优先级高于用户问题。
            # 这里约束它是学术助手，必须基于检索内容回答，减少幻觉。
            system_prompt = f"""你是一个严谨的学术论文助手。请根据以下【检索到的内容】回答用户的问题。
【回答规则】：
1. 如果【检索到的内容】中完全找不到相关信息，请回答：“根据上传的文档，无法找到相关描述”。
2. 如果【检索到的内容】中有间接线索，请进行合理的上下文推理，但不要凭空捏造数据。
3. 回答时，必须引用出处（如：第X页）。
4. 如果内容中包含乱码（如菱形符号），那是 PDF 公式提取错误，请忽略乱码，尽可能根据文字上下文作答。

【检索到的内容】：
{retrieved_context}
"""
            messages_to_send = [{"role": "system", "content": system_prompt}] + st.session_state.messages
            # `[{"role": "system", "content": system_prompt}]`：系统消息，放我们写好的角色规则 + 检索出来的参考文档（`retrieved_context`）
            # 发给大模型的消息队列 = 系统指令 + 历史问答，实现**带记忆的对话**，大模型能记得前面聊了什么。
            
            # 4. 生成回答
            try:
                response = client.chat.completions.create(
                    model="qwen-vl-ocr-2025-11-20",
                    messages=messages_to_send,
                    temperature=0.1
                    # 温度参数，**越低越严谨、随机性越小、越忠于原文**；
                )
                answer = response.choices[0].message.content
                # `choices`：模型返回的候选回答列表（默认只返回 1 条）
            except Exception as e:
                answer = f"调用大模型出错：{e}"

            st.markdown(answer)
            with st.expander(" 查看检索到的底层依据"):
                # `st.expander`：Streamlit 的折叠面板组件，网页上会出现一个可以点开 / 收起的框，标题是 `查看检索到的底层依据`
                st.info(retrieved_context)
                # 把 `retrieved_context`（也就是前面检索拿到的所有文档片段、带页码标记的内容）放在面板里面展示

    st.session_state.messages.append({"role": "assistant", "content": answer})
    # 把刚刚大模型输出的回答，追加存入会话历史：
    # `role="assistant"` 代表这条是助手消息
# 作用：下一轮用户提问的时候，`st.session_state.messages` 里就包含这条回答，模型就能看到本次对话，实现连续聊天记忆。
