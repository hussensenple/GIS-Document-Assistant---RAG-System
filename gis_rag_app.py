import streamlit as st
import os
import tempfile
import json
from collections import Counter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 1. إعدادات الصفحة
st.set_page_config(page_title="GIS Document Assistant", page_icon="🗺️", layout="wide")
st.title("🗺️ GIS Document Assistant - RAG System")

# 2. إعدادات الـ Sidebar (المفاتيح ورفع الملفات والإحصائيات)
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    
    # 🌟 Bonus 1: دعم أكثر من ملف PDF (accept_multiple_files=True)
    uploaded_files = st.file_uploader("📄 Upload GIS PDFs", type="pdf", accept_multiple_files=True)
    
    # 🌟 Bonus 3: اختيار لغة الرد
    language = st.selectbox("🌐 لغة الرد (Response Language)", ["English", "Arabic"])
    
    st.divider()
    st.header("📊 إحصائيات الجلسة")
    
    # مكان لعرض الإحصائيات لاحقاً
    stats_placeholder = st.empty()

if not api_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar to start.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# 3. معالجة الملفات (Processing)
if uploaded_files and "vectorstore" not in st.session_state:
    with st.spinner("📚 Processing PDFs..."):
        all_chunks = []
        
        # المرور على كل الملفات المرفوعة
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # تقسيم النص
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
        
        # إنشاء Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        st.session_state.vectorstore = Chroma.from_documents(all_chunks, embeddings)
        
    st.success(f"✅ Processed {len(uploaded_files)} files into {len(all_chunks)} chunks!")

# المتغيرات المحفوظة في الجلسة (Session State) للإحصائيات والمحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []
if "used_pages" not in st.session_state:
    st.session_state.used_pages = []

# 4. واجهة المحادثة (Chat Interface)
if "vectorstore" in st.session_state:
    
    # عرض المحادثات السابقة
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # استقبال سؤال جديد
    if question := st.chat_input("Ask about your GIS documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # البحث في الـ Vector Store
                docs = st.session_state.vectorstore.similarity_search(question, k=3)
                context = "\n\n".join([d.page_content for d in docs])
                
                # تجميع أرقام الصفحات للإحصائيات
                for d in docs:
                    page_num = d.metadata.get("page", "Unknown")
                    st.session_state.used_pages.append(page_num)
                
                # 🌟 Bonus 3: تطبيق اللغة في الـ Prompt
                lang_instruction = "Answer in Arabic." if language == "Arabic" else "Answer in English."
                
                prompt = f"""Use this GIS document context to answer the question.
                {lang_instruction}
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:"""
                
                # إرسال للـ LLM
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
                response = llm.invoke(prompt)
                answer = response.content
                
                st.write(answer)
                
                # 🌟 المتطلب الرابع: عرض المصادر
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(docs, 1):
                        page = doc.metadata.get("page", "?")
                        st.write(f"**Source {i}** (Page {page}):")
                        st.write(doc.page_content[:200] + "...")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # تحديث الإحصائيات في الـ Sidebar (🌟 Bonus 4)
    with stats_placeholder.container():
        st.write(f"💬 عدد الأسئلة: {len([m for m in st.session_state.messages if m['role'] == 'user'])}")
        if st.session_state.used_pages:
            most_common_page = Counter(st.session_state.used_pages).most_common(1)[0][0]
            st.write(f"📄 أكثر صفحة استُخدمت كمصدر: {most_common_page}")

    # 🌟 Bonus 2: تصدير المحادثة لـ JSON
    if st.session_state.messages:
        st.divider()
        chat_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=4)
        st.download_button(
            label="💾 تحميل المحادثة (JSON)",
            data=chat_json,
            file_name="gis_chat_history.json",
            mime="application/json"
        )