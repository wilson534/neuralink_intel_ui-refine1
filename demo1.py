import os
import socket
import whisper
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import pyttsx3
import streamlit as st
from datetime import datetime
import torch
from openvino import Core
from sympy import false
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from optimum.intel.openvino import OVModelForSequenceClassification
#自定义 模块
from whisper_ov_runner import transcribe_with_openvino
import json
#引入扣子库
from cozepy import COZE_CN_BASE_URL
from cozepy import Coze, TokenAuth, Message, ChatStatus, MessageContentType
#这里是加入了本地的情绪分析
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from db_logger import init_db,insert_log
import numpy as np
import librosa

# ========== 模块1：录音模块 ==========
def record_audio(filename="input.wav", duration=5, samplerate=16000):
    print("\n🎙️ 正在录音中...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    wav.write(filename, samplerate, recording)
    print("✅ 录音完成：", filename)
    return filename

# 加上 NPU，GPU调度
core = Core()


# ========== 模块2：Whisper STT ==========
# whisper_model = whisper.load_model("base")
#
def transcribe_audio(file_path):
    text = transcribe_with_openvino("input.wav")
    return text

# ========== 模块3：联网判断 ==========
def is_online():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False

# ========== 模块4：本地 LLM（Ollama） ==========
# def local_llm_query(prompt):
#     try:
#         res = requests.post("http://localhost:11434/api/generate", json={
#             "model": "qwen:7b",
#             "prompt": prompt,
#             "stream": False
#         })
#         return res.json()["response"]
#     except:
#         return "[本地模型响应失败]"

def local_llm_query(prompt):
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "qwen:7b",
            "prompt": prompt,
            "stream": True
        }, stream=True)

        output = ""
        for line in res.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                output += data.get("response", "")
        return output if output else "[无流式响应]"
    except Exception as e:
        return f"[流式响应失败: {str(e)}]"


# ========== 模块5：增强情绪识别（使用情感分类模型） ==========
emo_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")
#保存起来
# emo_model = OVModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese",export=True)
# emo_model.save_pretrained("models/emotion_openvino")
#已经在项目中进行保存了 所以直接调用
emo_model = OVModelForSequenceClassification.from_pretrained("models/emotion_openvino",device="GPU")


def enhanced_emotion_analysis(text):
    inputs = emo_tokenizer(text, return_tensors="pt", truncation=True, padding=True,max_length=2048)
    outputs = emo_model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    emo_label = torch.argmax(probs, dim=-1).item()
    emo_map = {0: "负面", 1: "正面"}
    return emo_map.get(emo_label, "中性")


def analyze_voice_emotion(filename="input.wav"):
    try:
        y, sr = librosa.load(filename, sr=16000)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

        # 简单阈值模拟分类逻辑（可替换为模型）
        energy = np.mean(librosa.feature.rms(y=y))
        pitch = np.mean(librosa.yin(y, fmin=80, fmax=400, sr=sr))

        if energy < 0.02:
            return "低落"
        elif pitch > 200:
            return "激动"
        else:
            return "平静"
    except Exception as e:
        return f"语音情绪识别失败:{str(e)}"

# ========== 模块6：意图识别与建议生成 ==========
def analyze_intent_and_suggestion(text):
    intent_keywords = {
        "请求帮助": ["不会", "怎么办", "帮我", "能不能", "你知道吗"],
        "表达情绪": ["我好难过", "我生气", "我不想", "我害怕"],
        "寻求陪伴": ["陪我", "你在吗", "和我玩", "我一个人"]
    }
    for intent, keys in intent_keywords.items():
        if any(k in text for k in keys):
            suggestion = f"检测到孩子可能在'{intent}'，建议父母给予关注与引导。"
            return intent, suggestion
    return "普通交流", "目前无需特殊干预。"

# ========== 模块7：TTS ==========
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    engine.say(text)
    engine.runAndWait()

# ========== 模块8：Coze Agent 云端调用 ==========

def clean_coze_reply(raw):
    parts = raw.strip().split("\n", 1)
    if len(parts) == 2 and parts[1].lstrip().startswith("{"):
        return parts[0].strip()
    return raw.strip()


def coze_agent_call(query):
    try:
        api_url = "https://api.coze.cn/open_api/v2/chat"
        payload = {
            "bot_id": "7516852953635455039",  # 请替换成你的 bot_id
            "user": "child_user",
            "query": query,
            "stream": False
        }
        headers = {
            "Authorization": "Bearer pat_t0eI6BgSXzCZZKw9d8FQTA4rfaKAEaTRZO4jt9r6T2euoz6lsN3N3aMNcL1ONKOc",
            "Content-Type": "application/json"
        }
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        tem = response.json()["messages"][1]["content"]
        tem = clean_coze_reply(tem)
        return tem
    except Exception as e:
        return f"[Coze 云端响应失败：{str(e)}]"


#======================新模块: 历史情绪分析 ====================
def plot_emotion_trends():
    st.sidebar.empty() #新加入的
    conn = sqlite3.connect("dialogue_log.db")
    # df = pd.read_sql_query("SELECT timestamp, text_emotion FROM logs", conn)
    df = pd.read_sql_query("SELECT * FROM logs", conn)  # 查询所有数据
    conn.close()

    if df.empty:
        st.warning("暂无情绪数据记录")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['week'] = df['timestamp'].dt.to_period("W").astype(str)
    df['month'] = df['timestamp'].dt.to_period("M").astype(str)

    st.subheader("📈 情绪趋势分析")

    emotion_counts_week = df.groupby(['week', 'text_emotion']).size().reset_index(name='count')
    emotion_counts_month = df.groupby(['month', 'text_emotion']).size().reset_index(name='count')

    tab1, tab2 = st.tabs(["📅 周视图", "🗓 月视图"])
    # 设置Matplotlib的字体参数
    plt.rcParams['font.family'] = 'SimHei'  # 选择一个支持中文的字体
    with tab1:
        fig1 = plt.figure(figsize=(8,4))
        sns.lineplot(data=emotion_counts_week, x='week', y='count', hue='text_emotion', marker='o')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)

    with tab2:
        fig2 = plt.figure(figsize=(6,4))
        sns.barplot(data=emotion_counts_month, x='month', y='count', hue='text_emotion')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)


#==========================新加入 存储情绪=====================
def show_logs():
    conn = sqlite3.connect("dialogue_log.db")
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC", conn)
    st.dataframe(df)


# ========== 模块9：Streamlit 界面 ==========
st.title("👨‍👩‍👧 NeruaLink离线/在线亲子对话分析系统")
st.caption("Whisper + Ollama + Coze + 情绪识别 + 意图分析 + TTS")
init_db() #程序开始的时候 初始化一遍
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

col1, col2 = st.columns(2)
mode = col1.radio("模式选择：", ["自动切换（推荐）", "强制离线", "强制在线"])
duration = col2.slider("录音时长（秒）", 2, 10, 5)

if st.button("🎙️ 开始对话"):
    filename = record_audio(duration=duration)
    text = transcribe_audio(filename)

    if mode == "强制离线":
        online = False
    elif mode == "强制在线":
        online = True
    else:
        online = is_online()

    if online:
        reply = coze_agent_call(text)
    else:
        reply = local_llm_query(text)

    emotion = enhanced_emotion_analysis(text)
    intent, suggestion = analyze_intent_and_suggestion(text)
    speak_text(reply)
    voice_emotion = analyze_voice_emotion(filename)
    # 修改这一行
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 添加年月日
    st.session_state.chat_log.append((timestamp, text, reply, emotion,voice_emotion,intent, suggestion))
    insert_log(
        timestamp,
        text,
        reply,
        emotion,
        voice_emotion,
        intent,
        suggestion
    )

# 在 StreamLit 主页面上中添加了入口按钮
if st.sidebar.button("📊 查看情绪趋势图"):
    plot_emotion_trends()

if st.sidebar.button("📜 查看历史记录"):
    show_logs()

st.subheader("🧾 对话记录")
for t, user, bot, emo, voice_emo, intent, sugg in reversed(st.session_state.chat_log):
    st.markdown(f"**🕒 {t}**")
    st.markdown(f"👧 孩子说：`{user}`")
    st.markdown(f"🤖 回复：{bot}")
    st.markdown(f"🧠 文本情绪：{emo} ｜ 🔊 语音情绪：{voice_emo}")
    st.markdown(f"🧩 意图：{intent}")
    st.markdown(f"📌 建议：*{sugg}*")
    st.markdown("---")

