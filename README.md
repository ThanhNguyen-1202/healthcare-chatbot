# Bản nâng cấp RAG++ Lite

> Bản zip này đã được làm nhẹ để tải được trong ChatGPT. Hai file model `.pkl` dung lượng lớn và file raw XML MedlinePlus đã được loại khỏi gói nén. Backend vẫn chạy được nhờ fallback rule-based + RAG++ và database in-memory khi `MONGO_URI` trống. Xem thêm `docs/UPGRADE_REPORT.md` và `docs/API_EXAMPLES.md`.

# 🏥 Healthcare Chatbot with LLM + RAG

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb" alt="MongoDB">
  <img src="https://img.shields.io/badge/Gemini-LLM-8E75FF?logo=google" alt="Gemini">
  <img src="https://img.shields.io/badge/RAG-Knowledge%20Retrieval-orange" alt="RAG">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker" alt="Docker">
</p>

Healthcare Chatbot là hệ thống hỗ trợ **thu thập triệu chứng, sàng lọc ban đầu, gợi ý khoa khám và giải thích tham khảo** bằng cách kết hợp **LLM + RAG + rule-based logic**.

Hệ thống được xây dựng theo hướng thực tế, ưu tiên:
- 💬 hội thoại nhiều lượt
- 🧠 lưu nhớ ngữ cảnh
- 🚨 phát hiện dấu hiệu nguy hiểm
- 📋 trả về kết quả dễ hiểu cho người dùng phổ thông
- 🛡️ vẫn hoạt động được ngay cả khi dịch vụ LLM bị lỗi hoặc hết quota

> ⚠️ **Lưu ý quan trọng:** Đây là hệ thống hỗ trợ sàng lọc ban đầu và cung cấp thông tin tham khảo. Kết quả từ chatbot **không thay thế chẩn đoán, chỉ định điều trị hoặc tư vấn trực tiếp từ bác sĩ**.

> ✅ **Bản nâng cấp theo đề cương khóa luận:** Project hiện đã bổ sung intake mở rộng, sàng lọc 5 mức, RAG++ hybrid retrieval, giải thích có dẫn nguồn kiểu IEEE, nhận diện nhóm nguy cơ và fallback an toàn. Xem chi tiết tại `UPGRADE_NOTES.md`.


---

## 🎯 1. Mục tiêu dự án

Dự án được xây dựng nhằm hỗ trợ người dùng:
- 📝 nhập triệu chứng bằng ngôn ngữ tự nhiên
- ❓ được chatbot hỏi thêm thông tin còn thiếu theo ngữ cảnh
- 🩺 nhận kết quả gồm **top 3 bệnh liên quan**
- ⏱️ biết được **mức độ ưu tiên thăm khám**
- 🏥 được gợi ý **khoa điều trị phù hợp**
- 📚 đọc **giải thích tham khảo** từ kho tri thức nội bộ + MedlinePlus

Ngoài ra, dự án còn là một mô hình thực hành để kết hợp:
- ⚡ **FastAPI**
- 🍃 **MongoDB**
- 🤖 **Gemini API**
- 🔎 **RAG**
- 🐳 **Docker**
- 💻 giao diện chat gần với trải nghiệm thực tế

---

## ✨ 2. Tính năng chính

### 💬 2.1. Chat nhiều lượt có nhớ ngữ cảnh
- Người dùng không cần nhập hết thông tin trong một câu.
- Bot có thể hỏi tiếp những trường còn thiếu.
- Session chat được lưu vào MongoDB.

### 📥 2.2. Thu thập dữ liệu intake theo hội thoại
Hệ thống gom thông tin từ nhiều lượt chat thành một intake snapshot gồm:
- triệu chứng chính
- thời gian
- mức độ
- nhiệt độ
- vị trí triệu chứng
- bệnh nền
- tuổi
- giới tính
- triệu chứng phụ
- dấu hiệu nguy hiểm

### 🧩 2.3. Kết hợp rule-based extraction và Gemini extraction
Hệ thống dùng hai lớp bóc tách dữ liệu:
- **Rule-based extraction** để xử lý nhanh, ổn định
- **Gemini extraction** để hiểu các câu tự nhiên phức tạp hơn

Nếu Gemini lỗi hoặc hết quota, hệ thống vẫn có thể tiếp tục nhờ fallback rule-based.

### 🧪 2.4. Dự đoán top 3 bệnh liên quan
Sau khi intake đủ dữ liệu, hệ thống sẽ dự đoán:
- top 3 bệnh liên quan
- phần trăm phù hợp
- và trả về theo định dạng dễ đọc

### 🚦 2.5. Phân loại mức độ sàng lọc
Hệ thống hỗ trợ 5 mức sàng lọc:
- 🚑 **Mức 1 - Cấp cứu ngay**
- 🔴 **Mức 2 - Rất khẩn**
- 🟠 **Mức 3 - Khẩn mức vừa**
- 🟡 **Mức 4 - Thông thường**
- 🟢 **Mức 5 - Tự theo dõi**

### 🚨 2.6. Phát hiện red flags
Bot có thể nhận diện các dấu hiệu nguy hiểm như:
- đau ngực
- khó thở
- ngất
- co giật
- tím tái
- chảy máu nhiều
- sốt cao kéo dài
- hoặc các tổ hợp triệu chứng cần ưu tiên đánh giá sớm

### 🏥 2.7. Gợi ý khoa điều trị
Sau khi có kết quả bệnh liên quan, hệ thống sẽ gợi ý khoa phù hợp, ví dụ:
- Hô hấp
- Da liễu
- Tiêu hóa
- Tim mạch
- Nội tiết
- Thần kinh
- Thận - Tiết niệu
- Cơ xương khớp
- Nội tổng quát

### 📖 2.8. Giải thích kết quả bằng RAG
Phần giải thích được xây dựng từ:
- kho tri thức nội bộ
- dữ liệu MedlinePlus đã parse
- retriever ưu tiên theo triệu chứng, dấu hiệu nguy hiểm và tên bệnh

### 🖥️ 2.9. Giao diện chat có lịch sử trò chuyện
Frontend hỗ trợ:
- chat nhiều lượt
- xem lại lịch sử
- đổi tên cuộc trò chuyện
- xóa từng cuộc trò chuyện
- xóa toàn bộ lịch sử phía giao diện
- thu gọn / mở thanh sidebar

### 🐳 2.10. Hỗ trợ chạy bằng Docker
Dự án có thể chạy:
- theo kiểu local truyền thống
- hoặc bằng Docker / Docker Compose

---

## 🛠️ 3. Công nghệ sử dụng

### ⚙️ Backend
- Python
- FastAPI
- Uvicorn
- Pydantic
- Motor / PyMongo
- python-dotenv

### 🎨 Frontend
- HTML
- CSS
- JavaScript

### 🗄️ Cơ sở dữ liệu
- MongoDB
- MongoDB Atlas

### 🤖 AI / NLP / Logic
- Gemini API
- Rule-based extraction
- Rule-based / ML disease prediction
- Triage logic
- Department routing
- RAG retrieval + explanation

### 🚀 DevOps / Quản lý mã nguồn
- Docker
- Docker Compose
- Git
- GitHub

---

## 🏗️ 4. Kiến trúc hệ thống

```text
Frontend (HTML/CSS/JS)
        |
        v
FastAPI Backend
        |
        +--> Chat Service
        |       +--> Rule-based extraction
        |       +--> Gemini extraction
        |       +--> Intake snapshot
        |
        +--> Prediction Service
        |       +--> Disease prediction
        |       +--> Triage classification
        |       +--> Department routing
        |
        +--> RAG
        |       +--> Retriever
        |       +--> Explainer
        |       +--> Translation fallback
        |
        +--> MongoDB / MongoDB Atlas
                +--> chat_sessions
                +--> predictions
```

---

## 📁 5. Cấu trúc thư mục chính

```text
healthcare-chatbot/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes_chat.py
│   │   │   ├── routes_llm.py
│   │   │   ├── routes_predict.py
│   │   │   └── routes_rag.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── db/
│   │   │   ├── mongo.py
│   │   │   └── repositories/
│   │   ├── mappings/
│   │   ├── rag/
│   │   │   ├── retriever.py
│   │   │   └── explainer.py
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   ├── .env
│   ├── .env.docker
│   ├── requirements.txt
│   └── requirements-lock.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── frontend/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── api.js
│   │   ├── app.js
│   │   └── chat.js
│   └── index.html
│
├── docker/
│   ├── api.Dockerfile
│   └── frontend.Dockerfile
│
├── scripts/
├── docker-compose.yml
├── .dockerignore
├── .env.example
└── README.md
```

---

## 🔄 6. Luồng hoạt động của chatbot

1. 👤 Người dùng nhập triệu chứng qua giao diện chat  
2. 💾 Backend lưu tin nhắn vào session MongoDB  
3. 🧠 Hệ thống bóc tách dữ liệu bằng:
   - rule-based extraction
   - Gemini extraction khi cần
4. 📦 Dữ liệu được gom vào intake snapshot  
5. ❗ Nếu thiếu trường quan trọng, bot sẽ hỏi tiếp theo ngữ cảnh  
6. ✅ Khi đủ dữ liệu, hệ thống sẽ:
   - chạy prediction
   - phân loại mức độ sàng lọc
   - gợi ý khoa điều trị
   - truy xuất tài liệu phù hợp từ RAG
7. 📤 Bot trả về:
   - top 3 bệnh liên quan
   - phần trăm phù hợp
   - trạng thái sàng lọc
   - khoa gợi ý
   - giải thích tham khảo
   - cảnh báo nếu có red flag

---

## 🚦 7. 5 trạng thái sàng lọc

- 🚑 **Cấp cứu ngay:** cần đến cơ sở y tế hoặc cấp cứu ngay lập tức
- 🔴 **Rất khẩn:** cần được đánh giá y tế rất sớm
- 🟠 **Khẩn:** nên đi khám sớm trong ngày hoặc sớm nhất có thể
- 🟡 **Ưu tiên khám sớm:** nên đi khám trong thời gian gần
- 🟢 **Theo dõi / khám thường:** có thể tiếp tục theo dõi hoặc khám thông thường nếu chưa có dấu hiệu nặng

---

## 🧾 8. Bộ dữ liệu sử dụng

### 🦠 8.1. Dữ liệu bệnh
Dự án sử dụng bộ dữ liệu bệnh để:
- suy ra nhóm bệnh liên quan
- gom nhóm bệnh
- ánh xạ sang khoa điều trị

Các file processed thường dùng:
- `ViMedical_Disease_grouped.csv`
- `ViMedical_Disease_department_mapped.csv`

### 🗂️ 8.2. Mapping và alias
Hệ thống sử dụng các file mapping để hỗ trợ:
- tên bệnh → khoa
- tên bệnh → alias tiếng Anh / đồng nghĩa
- rule triage
- rule rerank bệnh

Ví dụ:
- `disease_to_department.json`
- `disease_aliases.json`
- `triage_rules.json`

### 🌍 8.3. Dữ liệu MedlinePlus
Dữ liệu MedlinePlus được:
1. parse từ XML
2. làm sạch nội dung
3. thêm disease names / aliases
4. merge với kho tri thức nội bộ
5. dùng làm nguồn tham khảo cho RAG

Các file liên quan:
- `medlineplus_health_topics.xml`
- `medlineplus_rag_knowledge.jsonl`
- `rag_knowledge_base_merged.jsonl`

---

## 🔎 9. RAG trong dự án

RAG được dùng để giải thích tại sao hệ thống gợi ý nhóm bệnh hoặc khoa như vậy.

### 📚 Nguồn tri thức
- Kho tri thức nội bộ
- MedlinePlus

### 🎯 Retriever hiện ưu tiên theo
- triệu chứng
- dấu hiệu nguy hiểm
- khoa khám
- tên bệnh dự đoán
- alias bệnh

### ✅ Mục tiêu
- giải thích theo **bệnh** tốt hơn
- không chỉ dừng ở giải thích theo **triệu chứng**

---

## 💻 10. Chạy project local

### 📥 10.1. Clone project
```bash
git clone <repo-url>
cd healthcare-chatbot
```

### 🧪 10.2. Tạo môi trường ảo

#### Windows
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
```

### 📦 10.3. Cài thư viện
```bash
pip install -r requirements.txt
```

### ⚙️ 10.4. Tạo file môi trường
Tạo file `backend/.env` với nội dung tối thiểu:

```env
APP_NAME=healthcare-chatbot
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

MONGO_URI=
MONGO_DB=healthcare_chatbot

GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_EXTRACTOR_MODEL=gemini-3.1-flash-lite
GEMINI_TRANSLATION_MODEL=gemini-3.1-flash-lite
```

### ▶️ 10.5. Chạy backend
```bash
uvicorn app.main:app --reload
```

### 🌐 10.6. Chạy frontend
Mở file `frontend/index.html` bằng:
- Live Server
- hoặc một static server đơn giản

Ví dụ:
```text
http://127.0.0.1:5500/frontend/index.html
```

---

## 🐳 11. Chạy project bằng Docker

Dự án hỗ trợ chạy bằng Docker với:
- backend container
- frontend container
- MongoDB Atlas ở ngoài cloud

### 📄 11.1. File `.env.docker`
Tạo file `backend/.env.docker`:

```env
APP_NAME=healthcare-chatbot
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

MONGO_URI=mongodb+srv://USERNAME:PASSWORD@cluster0.wqbnc3j.mongodb.net/healthcare_chatbot?retryWrites=true&w=majority&appName=Cluster0
MONGO_DB=healthcare_chatbot

GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_EXTRACTOR_MODEL=gemini-3.1-flash-lite
GEMINI_TRANSLATION_MODEL=gemini-3.1-flash-lite
```

> ⚠️ Nếu password MongoDB Atlas có ký tự đặc biệt như `@`, `#`, `%`, `/`, cần URL encode trước khi đưa vào `MONGO_URI`.

### 🧱 11.2. Build và chạy
Từ thư mục gốc project:

```bash
docker compose up --build
```

### 🔌 11.3. Các cổng mặc định
- **Frontend:** http://localhost:8080
- **Backend:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs

---

## 🔗 12. Các API chính

- `POST /chat/` — Gửi tin nhắn chat
- `GET /chat/session/{session_id}` — Xem session chat
- `GET /chat/session/{session_id}/intake` — Xem intake snapshot
- `POST /predict/` — Test prediction trực tiếp
- `GET /predict/session/{session_id}` — Xem prediction history
- `POST /llm/extract` — Test Gemini intake extraction
- `POST /rag/retrieve` — Test RAG retrieval

---

## 🛡️ 13. Xử lý lỗi và fallback

Hệ thống đã được bổ sung nhiều cơ chế để chạy ổn định hơn:

### 🤖 13.1. Khi Gemini hết quota
- chatbot không dừng hẳn
- hệ thống quay về rule-based extraction

### 🧾 13.2. Khi thiếu dữ liệu đầu vào
- bot hỏi tiếp theo ngữ cảnh
- không yêu cầu người dùng nhập toàn bộ ngay từ đầu

### 💬 13.3. Khi người dùng trả lời ngắn
Các câu như:
- “không”
- “không có”
- “không ạ”
- “ho”
- “sốt”

vẫn được hệ thống cố gắng hiểu theo ngữ cảnh câu hỏi hiện tại.

### 🌐 13.4. Khi dịch tài liệu RAG thất bại
- hệ thống có thể fallback sang bản tóm tắt tiếng Việt đơn giản
- tránh trả nguyên văn tiếng Anh cho người dùng cuối

---

## 🖥️ 14. Giao diện người dùng

Frontend hiện hỗ trợ:
- chat dạng hội thoại
- lịch sử chat bên trái
- đổi tên cuộc trò chuyện
- xóa từng cuộc trò chuyện
- xóa toàn bộ lịch sử giao diện
- thu gọn / mở sidebar
- ô nhập cố định dưới cùng

Lịch sử giao diện hiện lưu ở:
- `localStorage`

---

## 📌 15. Trạng thái hiện tại của dự án

Đây là phiên bản **MVP / prototype có thể demo**.

Hệ thống hiện đang kết hợp:
- FastAPI backend
- MongoDB lưu session
- Gemini extraction
- Prediction logic
- Triage logic
- Department routing
- RAG giải thích
- Frontend chat cơ bản
- Docker deployment local

---

## ⚠️ 16. Hạn chế hiện tại

- Chưa thay thế được chẩn đoán y khoa thực tế
- Chất lượng dự đoán còn phụ thuộc nhiều vào dữ liệu và luật hiện tại
- Một số phần vẫn còn thiên về rule-based
- Chưa có xác thực người dùng
- Lịch sử chat hiện chủ yếu ở frontend
- RAG hiện chủ yếu là keyword retrieval, chưa phải vector search hoàn chỉnh

---

## 🚀 17. Hướng phát triển

Trong tương lai có thể mở rộng theo các hướng:
- cải thiện mô hình dự đoán bệnh
- thêm dataset y tế thực tế và sạch hơn
- bổ sung vector search / embeddings cho RAG
- tối ưu giao diện người dùng
- thêm dashboard quản lý lịch sử ca bệnh
- thêm xác thực người dùng
- triển khai production trên cloud

---

## 🔐 18. Lưu ý an toàn

Hệ thống này chỉ hỗ trợ sàng lọc ban đầu và cung cấp thông tin tham khảo.

Người dùng vẫn cần đến:
- cơ sở y tế
- bác sĩ chuyên môn
- hoặc cấp cứu

nếu có triệu chứng nặng, kéo dài hoặc có dấu hiệu nguy hiểm.

---

## 👨‍💻 19. Tác giả

- **Họ tên:** Nguyễn Chí Thanh
- **Dự án:** Healthcare Chatbot with LLM + RAG
- **Mục đích:** Học tập, thực tập, nghiên cứu và demo kỹ thuật

---

## 20. Mô hình DDXPlus mới

Dự án hiện sử dụng **DDXPlus Structured Disease Model** thay cho mô hình TF-IDF ViMedical trong luồng dự đoán chính.

- 49 bệnh DDXPlus
- 515 token bằng chứng
- đầu vào gồm evidence, tuổi và giới tính
- đầu ra Top-3 bệnh có tên Việt/Anh, ICD-10 và khoa gợi ý
- model có sẵn tại `backend/app/ml/artifacts/ddxplus_disease_model.pkl`

Chạy trực tiếp:

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Train lại:

```powershell
python scripts/prepare_ddxplus_dataset.py
python scripts/train_ddxplus_structured_model.py --epochs 1 --chunksize 100000
python scripts/test_ddxplus_integration.py --rows 1000
```

Chi tiết: `docs/DDXPLUS_MIGRATION.md`. Các script ViMedical cũ chỉ được giữ lại để đối chiếu và không còn được backend tải khi khởi động.
