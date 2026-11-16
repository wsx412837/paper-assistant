# - Embedding 模型：只把文字变成向量，不会回答问题
# - LLM 大模型：看懂参考片段、组织语言输出答案
# - Chroma：向量数据库，用来存向量 + 原文 + 页码元数据，做相似度搜索
# - SessionState：保存对话记录，页面刷新 /rerun 不丢数据

import streamlit as st
# 导入 `streamlit` 库，并起别名 `st`
# Streamlit，Python 快速 Web 可视化框架，不需要写 HTML/JS，只用几行 Python 就能生成网页界面、输入框、聊天框、按钮，用来做这个 RAG 应用的前端页面。
# 安装命令：`pip install streamlit`
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
    # 就必须先创建 client，然后用 client.xxx 去调用。
    api_key="sk-ws-H.PIPPMIL.7Vwl.MEUCIQDUK9KVnHlry6YhKfYrMzNrSW_7qrhMrmTtdVi9lKweRwIgd-7rRGQ5gsFLKsiccCFSG2vL8n2rhRkYK4Vj8f7g0WA", # ⚠️ 替换成你的
    base_url="https://ws-guf27zl5m8zvagh9.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

# ================= 2. 计算向量的工具函数 =================
# 批量调用 OpenAI Embedding 接口
# 把一批文本转成向量（embedding），是 RAG 里【文本向量化】的核心工具，配合前面 chromadb 使用。
def get_embeddings(texts, batch_size=10):
    # `def` 是英文 define（定义） 的缩写，在 Python 里用来定义函数。这里就是定义了get_embeddings这个函数
    # `texts`：文本列表，比如 `["片段1","片段2","片段3"]`，就是 PDF 切分好的文本块
# `batch_size=10`：批次大小，默认一次最多发给 OpenAI 10 条文本；
# OpenAI 接口支持批量，分批调用可以减少请求次数、提升效率，同时避免单次请求超长超限。
    all_embeddings = []
    # 初始化空列表，用来**收集全部文本对应的向量**，最后把完整向量列表返回。
    for i in range(0, len(texts), batch_size):
        # `range(start, stop, step)`：从下标 0 开始，每次步长等于`batch_size`，循环切分文本列表
        # 把长文本列表切分成一小批一小批，分批请求接口。
        batch = texts[i:i+batch_size]
        # 切片，取出当前这一批文本，`batch` 就是本轮要向量化的文本数组。
        # `i:i+batch_size`：切片写法
        try:
            # `try-except` 异常捕获块：防止接口报错（网络、密钥错误、限流、文本过长）直接导致整个程序崩溃；出错时进入 except 分支。
            response = client.embeddings.create(
                # client：就是你前面提前创建好的 OpenAI 客户端对象，里面存了你的 API Key、接口地址，相当于 “和 OpenAI 服务器的通讯器”。
                # `embeddings`：client 下面的向量嵌入功能模块，专门管 embedding 相关请求，区分聊天、画图等别的功能。
                # `create()`：这是发起请求的方法：创建一次 embedding 任务，向服务器发送数据。
                # `response =`：
# 服务器处理完，会返回一大包结构化数据（包含每条文本对应的 embedding 向量、使用 token 数等）
# 把这包结果赋值给 `response`，后面我们从里面把向量取出来。
                model="text-embedding-v1",
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
        # `return None`：函数直接返回空，告诉上层调用：向量生成失败
        # 写了 `return None` = 明确传递语义：只要这里报错，立刻终止整个向量化流程，函数直接返回失败标记，不再继续处理剩下的批次！
        # 这个代码里最核心的必要性：报错后要不要中断后续批次；
    return all_embeddings
# 所有批次全部请求完成后，返回完整的向量列表

# ================= 3. 初始化 ChromaDB =================
if "collection" not in st.session_state:
    # `st.session_state`：Streamlit 的会话状态容器。
    # Streamlit 有一个特点：用户每点一个按钮、输入文字，整个脚本会从头到尾重新跑一遍（rerun）。
    # 不加这个判断，每次交互都会重复创建数据库集合，引发报错、数据丢失。
    chroma_client = chromadb.Client()
    # `chromadb.Client()`：创建 ChromaDB 的内存版客户端（默认是内存模式！程序一关，所有向量数据全部消失，不会持久化到硬盘）
    # `chroma_client`：这个变量就是你和向量数据库交互的客户端句柄，用来新建、获取集合。
    # 如果想要持久化保存，要写成 `chromadb.PersistentClient(path="./chroma_db")`，数据会存在本地文件夹
    st.session_state.collection = chroma_client.create_collection(name="pdf_notes")
    # 在 ChromaDB 里新建一个名叫 `pdf_notes` 的集合（collection）
    # collection 是 ChromaDB 的核心单元：用来存放文本、文本向量（embedding）、元数据 metadata；
    # RAG 里一般一个业务文档库对应一个 collection
collection = st.session_state.collection
# 从会话状态取出之前保存好的向量集合，赋值给局部变量 `collection`
# 方便后面代码直接写 `collection.add()`、`collection.query()` 来存入文本、做向量检索。

# ================= 4. 优化的切块函数（按段落合并） =================
def split_text_by_paragraph(text, max_chunk_size=600):
    # RAG 里的文本分块（chunking）逻辑，属于「按段落优先合并」策略
    # 目标：尽量不把一个完整段落拦腰切断，同时控制每一块文本长度上限
    # `max_chunk_size=600`：单块最大字符长度，默认 600 字符；这里是字符数，不是 token！这点很关键
    """按段落切分，保证一句话不会被生硬切断"""
    paragraphs = text.split('\n')
    # `text.split('\n')`：用换行符 `\n` 把整篇文本切割成一个段落列表。
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
st.title("智能 PDF 论文助手 (混合检索)")
# `st.title()`：Streamlit 的页面大标题，渲染在网页主内容区最上方。

with st.sidebar:
    # `with` 是 Python 的**上下文管理器语法**。
    # 核心：自动帮你做「进入前准备」和「退出后清理 / 收尾」，不用手动写 try-finally
    # `with st.sidebar:`：上下文管理器，把里面所有组件放到左侧侧边栏，和主页面分开。
    # 含义：下面所有组件，都放到侧边栏这个容器里面。离开这个 `with` 块之后，组件渲染位置回到主页面。
# Streamlit 封装好了 sidebar 的上下文，进入 `with`，切换渲染区域；退出，切回主区域。
    st.header("文档上传区")
    # `st.header()`：侧边栏内二级标题。
    uploaded_file = st.file_uploader("请上传 PDF 文件", type=["pdf"])
    # `st.file_uploader`：文件上传组件
# `type=["pdf"]`：限制只能上传 pdf 后缀文件
# 返回值 `uploaded_file`：
# 用户还没上传：`None`
# 用户上传成功：得到一个 `UploadedFile` 对象（流对象，不是本地文件路径）
# `UploadedFile` 是 Streamlit 自定义的一个类的实例对象，是对浏览器上传文件的一层包装。
# 它不等于文件本身，可以理解成：一个指向上传文件数据的「管道 / 句柄」。
# 它本质*内存里的临时文件流，不在你电脑硬盘上生成真实文件
# Streamlit 每次页面 rerun（输入框打字、点按钮），这个 `UploadedFile` 对象会被重置！
# 并且：**这个流是一次性的！默认只能 `.read()` 一次**，读完之后流指针到末尾，再次 `.read()` 就拿到空字节。
    
    if uploaded_file is not None:
        # 判断：用户已经上传了 PDF 文件，才渲染下面的按钮和解析逻辑；没上传就直接跳过这块。
        if st.button("解析并存入知识库"):
            # 在侧边栏区域生成一个按钮。
# Streamlit 核心特性：只有点击按钮，这个 if 内部代码才执行一次；页面其他交互 rerun 时，不会自动跑解析逻辑
# 防止反复解析 PDF。
# 如果你连续多次点击【解析并存入知识库】按钮：
# 会把同一篇 PDF 的 chunk，重复多次 add 进 ChromaDB。
#   同一个文本块，重复存入多条向量，id 会冲突（如果你 id 生成规则不变，直接报错！）
#   就算 id 不一样，检索的时候，同一个内容多次命中，检索结果冗余、问答变慢。

            with st.spinner("正在解析 PDF..."):
                # `st.spinner("文字")`：加载动画。进入这个上下文的时候，页面显示转圈 + 提示文字；代码块执行完毕，动画自动消失。
                # 用来给用户反馈 “程序正在干活，不是卡住了”。
                pdf_bytes = uploaded_file.read()
                # `.read()`：把上传文件读取成二进制字节流 (bytes)。
                # 计算机底层所有文件（PDF、图片、视频）全部都是由一串 0 和 1 组成的二进制数据。
# 在 Python 里，`bytes` 类型就是**二进制字节流**：以字节为单位的原始数据。
# PDF 不是纯文本！它是二进制格式文件，不能当成字符串读取，**PyMuPDF (fitz) 打开 PDF，接收的必须是二进制原始数据**。
# `UploadedFile` 对象**只是一个 “流通道”，不是已经读好的二进制数据本身**
# 它不会自动把全部文件加载成字节给你，你必须调用 `.read()` 主动拉取数据。
# Streamlit 收到浏览器上传文件，在内存创建这个流对象 `uploaded_file`，但**数据还在流缓冲区，没有全部取出来**；
#  fitz 没有办法直接接收 `UploadedFile` 这个 Streamlit 包装对象！它不认这个类型，它只认 bytes 或者本地路径。
#  所以必须手动 `.read()`，把 UploadedFile 管道里的数据读出来，拿到纯 bytes，传给 `stream=` 参数。
                # 上传的文件是在内存里，没有存到磁盘，直接读字节交给 fitz。
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                # `fitz` 就是 PyMuPDF，PDF 解析库。
                # `stream=pdf_bytes`：从字节流打开 PDF（而不是从本地路径打开文件）
                # `filetype="pdf"`：声明流类型是 pdf
                
                # 初始化 3 个列表，**为后面批量添加到 ChromaDB 做准备**：
                all_chunks = []
                # `all_chunks`：存放所有切好的文本块（`documents`）
                all_metadatas = []
                # `all_metadatas`：元数据，一般存页码、文件名等信息，检索的时候可以溯源 “这段文字来自 pdf 第几页”
                all_ids = []
                # `all_ids`：ChromaDB 要求每个 chunk 必须有唯一 id，用来标识每一条向量数据
                
                for page_num, page in enumerate(doc):
                    # `enumerate(doc)`：循环 PDF 的每一页
                    # `page_num`：页码，从 0 开始（后面展示给用户的时候记得 + 1，因为习惯从第 1 页开始）
                    # `page`：单页 PDF 对象
                    page_text = page.get_text()
                    # `page.get_text()`：提取当前页面里的全部文字，返回字符串。
                    # 只能提取文本；扫描版图片 PDF 拿不到文字，需要 OCR。
                    if not page_text.strip():
                        # 判断：这一页清理空白之后是空文本（空白页、只有图片没有文字）
                        continue
                    #  `continue`：跳过这一页，不做切块、不入库，避免无效空白 chunk。
                    
                    # 使用段落级切块
                    chunks = split_text_by_paragraph(page_text)
                    # `chunks = split_text_by_paragraph(page_text)`
# 调用之前写好的按段落切块函数，把当前这一页提取出来的长文本 `page_text`，切分成多个短文本块 chunk。
                    for chunk in chunks:
                        # 遍历当前页面切出来的每一个文本块
                        all_chunks.append(chunk)
                        # 把文本内容放进文档列表，后面给 Chroma 存原始文本
                        all_metadatas.append({"source": uploaded_file.name, "page": page_num + 1})
                        # **metadata 元数据**：记录这条 chunk 来自哪个 PDF 文件、第几页。
                        # 后续检索到这条片段的时候，你就可以告诉用户：该内容来自《xxx.pdf》第 x 页，方便溯源。
                        # `uploaded_file.name`：上传 PDF 的原始文件名
                        # `page_num+1`：页码（因为 enumerate 从 0 开始计数，+1 符合人类阅读习惯)
                        all_ids.append(str(uuid.uuid4()))
                        # str(uuid.uuid4()) 生成一个**全局唯一随机 ID**，转字符串。
# ChromaDB 要求每一条向量数据必须有唯一`id`；用 uuid 的好处：**多次上传同一个 PDF，也不会出现 id 重复冲突报错**
                
                if all_chunks:
                    # 判断：`all_chunks`列表不为空（成功拿到了至少一个文本块）。
                    with st.spinner(f"正在计算向量..."):
                        embeddings = get_embeddings(all_chunks)
# `get_embeddings(all_chunks)`：这是你自己写的 embedding 函数，输入一批文本块，调用 Embedding 模型，输出一组向量（浮点数数组）。
                    
                    if embeddings:
                        # `if embeddings:` 判断向量结果正常返回，不是空
                        collection.add(
                            # ChromaDB 集合的批量新增接口，一次性把文本、向量、元数据、id 全部存入向量库
                            documents=all_chunks,
                            # `documents`：原始文本片段（检索命中后展示给用户、喂给 LLM）
                            # **LLM = Large Language Model，大语言模型**
# 简单一句话：**经过海量文本训练出来、能理解和生成人类语言的巨型神经网络模型**
# 像 ChatGPT、通义千问、文心一言、Llama、Qwen（通义千问开源版）都属于 LLM。
                            embeddings=embeddings,
                            # `embeddings`：文本对应的向量数组，用于相似度检索
                            metadatas=all_metadatas,
                            # `metadatas`：溯源信息（文件名、页码）
                            ids=all_ids
                            # `ids`：每条记录唯一标识
                        )
                        st.success(f"✅ 成功存入 {len(all_chunks)} 个文本块！可以提问了。")
                        # 成功：弹出绿色成功提示，告诉用户一共入库多少块
                    else:
                        st.error("❌ 向量化失败。")
                        # 失败（embedding 返回空）：红色报错：向量化失败（常见原因：embedding 接口超时、key 错误、超长文本报错）
                else:
                    st.error("❌ 未提取到文字。")
                    # 外层`if all_chunks`的 else：PDF 里面没有提取到任何可解析文字（扫描版图片 PDF、纯图片、空白 PDF），提示未提取文字。

# ================= 6. 聊天界面 =================
# **问答交互入口**：保存对话历史、用户提问、向量检索，检索到上下文之后，下一步就是拼 prompt 喂给 LLM 生成答案。
if "messages" not in st.session_state:
    # `st.session_state`：会话状态，只要浏览器标签不关，里面的数据跨页面 rerun 不会丢失
    # 如果会话里还没有 `messages` 这个键，就新建一个空列表；`messages` 用来存一轮一轮的聊天记录
    # 每条消息是字典格式：`{"role":"user/assistant", "content":"消息文本"}`
    st.session_state.messages = []

# 页面渲染历史对话：循环读取已经保存的聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # `st.chat_message(role)`：Streamlit 的气泡组件
# `role="user"` → 用户气泡（一般在右边）
# `role="assistant"` → AI 助手气泡（左边）
        st.markdown(message["content"])
        # `st.markdown()`：支持 markdown 渲染（公式、换行、标题，适合论文问答）
# 只要页面 rerun（比如发消息），这段循环就会重新执行一遍，把历史对话全部画出来。

if prompt_text := st.chat_input("请输入关于该 PDF 的问题："):
    # `:=` 海象运算符：一边接收输入，一边赋值给 `prompt_text`
    # `st.chat_input()`：Streamlit 专用底部聊天输入框，回车提交
# - 用户没输入：`prompt_text` 是空，不进这个分支
# - 用户输入并回车：拿到用户的问题字符串，进入 if 内部逻辑

    with st.chat_message("user"):
        st.markdown(prompt_text)
        # 1. 先在页面上把用户刚刚发的问题，渲染成用户聊天气泡
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    # # 2. 把这条用户消息追加进 `messages` 对话历史列表，永久保存到会话里

    with st.chat_message("assistant"):
        # - 提前打开助手的气泡框
        with st.spinner("正在翻阅论文..."):
            # 加载动画：提示用户正在检索 PDF 知识库（向量检索阶段，还没调用 LLM）
            
            # 1. 向量检索：捞取最相关的 4 段
            #前面的要用PDF chunk 文字转向量；现在**用户问题也用同一个 embedding 模型，转到同一个向量空间**，才能做相似度比对
            query_embedding = get_embeddings([prompt_text])[0]
            # `get_embeddings` 函数接收**文本列表**，返回向量列表
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
                # 外层对应一次查询（你只查了 1 个 query），内层是返回的 n_results 条文本。
                # 它等价于两步判断，用 `and` 连起来，**两个条件必须同时成立才会进入 if 里面的循环**：
# `results['documents']`：判断：`results` 里面的 `documents` 是不是不是空列表、不是 None
#`results['documents'][0]`：取外层列表的第 0 项（也就是你这次查询对应的那一组片段列表），再判断这个内层列表是不是非空。
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
            # 向量检索看 “意思像不像”；关键词检索看 “字有没有出现”，二者互补，这就是混合检索。
            keywords = ["问题一", "MILP", "混合整数线性规划"]
            for kw in keywords:
                if kw in prompt_text:
                    # 去数据库里强行搜包含这个词的段落
                    kw_results = collection.get(where_document={"$contains": kw})
                    # `collection.get()` 是 Chroma 的过滤查询
                    # `where_document={"$contains": kw}`：Chroma 内置过滤器：筛选条件 **文档 chunk 的原文文本中，包含 kw 这个连续子串**
                    # 新增专业名词需要手动改代码。
# 真正工业级混合检索一般用 BM25，而不是这种简单 `$contains`。
                    if kw_results and kw_results['documents']:
                        for idx, doc_text in enumerate(kw_results['documents'][:2]): # 最多补2段
                            # `kw_results['documents']`：所有匹配到的文本块列表
                            # `[:2]`：列表切片，只取前 2 条，**截断上限为 2**
# 目的：如果数据库里几十条都包含 MILP，只拿前两条，避免检索文本太多，token 超限、LLM 变慢、开销暴涨
# `enumerate()`：同时拿到下标 idx和 文本内容`doc_text`
# `idx`：当前这条是第几个，用来同步取对应的元数据
                            meta = kw_results['metadatas'][idx]
# `metadatas` 和 `documents` 是一一对应的列表：第 0 个 metadata 对应第 0 段文本，第 1 个 metadata 对应第 1 段文本。
# `meta` 拿到当前这条 chunk 的元数据，代码里存了`page`页码（PDF 切分的时候记录的页码）。
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
                    model="qwen-plus",
                    messages=messages_to_send,
                    temperature=0.1
                    # 温度参数，**越低越严谨、随机性越小、越忠于原文**；RAG 知识库场景一般用 0~0.2，减少胡说（幻觉）。
                )
                answer = response.choices[0].message.content
                # `choices`：模型返回的候选回答列表（默认只返回 1 条）
# `choices[0]`：取第一条结果
# `.message.content`：取出模型生成的文本答案，赋值给 `answer`
            except Exception as e:
                answer = f"调用大模型出错：{e}"

            st.markdown(answer)
            # 把刚才拿到的回答，以 Markdown 格式渲染到网页上（支持换行、标题、公式）。
            with st.expander(" 查看检索到的底层依据"):
                # `st.expander`：Streamlit 的折叠面板组件，网页上会出现一个可以点开 / 收起的框，标题是 `查看检索到的底层依据`
                st.info(retrieved_context)
                # 把 `retrieved_context`（也就是前面检索拿到的所有文档片段、带页码标记的内容）放在面板里面展示

    st.session_state.messages.append({"role": "assistant", "content": answer})
    # 把刚刚大模型输出的回答，追加存入会话历史：
    # `role="assistant"` 代表这条是助手消息
# 作用：下一轮用户提问的时候，`st.session_state.messages` 里就包含这条回答，模型就能看到本次对话，实现连续聊天记忆。


    # `collection.query()` → 向量相似度检索（语义检索）
# 输入文本，转成向量，计算向量距离，返回语义最像的 TopK 片段。
# > 核心：按相似度打分排序，找 “意思相近” 的内容。

# `collection.get()` → 过滤查询（filter query）
# 它不做向量计算、不比较语义、不打分、不排序。
# 它的逻辑是：遍历库里的数据，只把满足你设定条件的数据挑出来返回，本质是一个「筛选器」。