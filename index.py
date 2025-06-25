import streamlit as st
from openai import OpenAI
import requests


# page_icon="<UNK>" 后续可以加入图片
st.set_page_config(page_title="NeuraLink", page_icon="<UNK>", layout="centered")

st.title("NeuraLink 心灵纽带")

api_key = "sk-68ec4223332248a4bf9e74e9e582247b"
api_url = "https://api.deepseek.com"

#文本输入
prompt = st.text_input("请输入你的问题或点击下方语音: ","")

#上传音频文件
audio_file = st.file_uploader("上传音频文件(wav/mp3/m4a)",type=["wav","mp3","m4a"])

if st.button("🎤 开始识别（Whisper）") and audio_file is not None:
    import whisper
    model = whisper.load_model("base")
    st.info("识别中,请稍等...")
    result = model.transcribe(audio_file.read())
    prompt = result["text"]
    st.success(f"识别结果: {prompt}")

# 与AI交互
# if prompt:
#     # 你可以替换为你自己的LLM模型调用方式 "sk-68ec4223332248a4bf9e74e9e582247b"
#     client = openai.OpenAI(api_key="sk-68ec4223332248a4bf9e74e9e582247b")
#     with st.spinner("AI 正在思考中..."):
#         response = client.chat.completions.create(
#             model = "deepseek-R1",
#             messages=[
#                 {"role":"user","content":prompt}
#             ]
#         )
#         answer = response.choices[0].message.content
#         st.markdown(f"🤖 **AI回复：**\n\n{answer}")


if prompt:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=True
    )

    print(response.choices[0].message.content)