const SESSION_STORAGE_KEY = "chat_session_id";
const CHAT_HISTORY_STORAGE_KEY = "chat_session_history";
const CHAT_FEEDBACK_STORAGE_KEY = "chat_response_feedback";
const SIDEBAR_COLLAPSED_KEY = "chat_sidebar_collapsed";

let currentSessionId = localStorage.getItem(SESSION_STORAGE_KEY) || null;
let loadingMessageDiv = null;
let historySearchTerm = "";
let activeRecognition = null;
let voiceIsRecording = false;
let voiceManuallyStopped = true;
let voiceBaseText = "";
let voiceFinalTranscript = "";
let voiceInterimTranscript = "";
let voiceRestartTimer = null;
let voiceSessionStartedAt = 0;

function getAppShell() {
  return document.getElementById("app-shell");
}

function getChatBox() {
  return document.getElementById("chat-box");
}

function getHistoryList() {
  return document.getElementById("history-list");
}

function getCurrentChatTitleElement() {
  return document.getElementById("current-chat-title");
}

function getCurrentChatSubtitleElement() {
  return document.getElementById("current-chat-subtitle");
}

function getToggleSidebarButton() {
  return document.getElementById("toggle-sidebar-btn");
}

function getHistorySearchInput() {
  return document.getElementById("history-search-input");
}

function getDisclaimerModal() {
  return document.getElementById("disclaimer-modal");
}

function getMessageInput() {
  return document.getElementById("message-input");
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function truncateText(text, maxLength = 80) {
  if (!text) return "";
  const normalized = String(text).replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) return normalized;
  return normalized.slice(0, maxLength - 3).trim() + "...";
}

function formatDateTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";

  return date.toLocaleString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "2-digit",
  });
}

function getStoredChatHistory() {
  try {
    const raw = localStorage.getItem(CHAT_HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error("Không đọc được lịch sử chat:", error);
    return [];
  }
}

function saveStoredChatHistory(items) {
  localStorage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(items));
}

function getHistoryItem(sessionId) {
  return getStoredChatHistory().find((item) => item.session_id === sessionId) || null;
}

function normalizeSearchText(text) {
  return String(text || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
    .trim();
}

function sortHistoryItems(items) {
  return [...items].sort((a, b) => {
    if (Boolean(a.is_pinned) !== Boolean(b.is_pinned)) {
      return Boolean(b.is_pinned) - Boolean(a.is_pinned);
    }

    return new Date(b.updated_at || 0) - new Date(a.updated_at || 0);
  });
}

function filterHistoryItems(items) {
  const q = normalizeSearchText(historySearchTerm);
  if (!q) return items;

  return items.filter((item) => {
    const haystack = normalizeSearchText(`${item.title || ""} ${item.preview || ""}`);
    return haystack.includes(q);
  });
}

function isSidebarCollapsed() {
  return localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "true";
}

function applySidebarState() {
  const appShell = getAppShell();
  const toggleBtn = getToggleSidebarButton();
  if (!appShell) return;

  if (isSidebarCollapsed()) {
    appShell.classList.add("sidebar-collapsed");
  } else {
    appShell.classList.remove("sidebar-collapsed");
  }

  if (toggleBtn) {
    toggleBtn.title = isSidebarCollapsed()
      ? "Mở lịch sử chat"
      : "Thu gọn lịch sử chat";
  }
}

function toggleSidebar() {
  localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(!isSidebarCollapsed()));
  applySidebarState();
}

function setCurrentSessionId(sessionId) {
  currentSessionId = sessionId;

  if (sessionId) {
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  } else {
    localStorage.removeItem(SESSION_STORAGE_KEY);
  }

  renderHistoryList();
}

function setCurrentChatTitle(title, subtitle) {
  const titleEl = getCurrentChatTitleElement();
  const subtitleEl = getCurrentChatSubtitleElement();

  if (titleEl) {
    titleEl.textContent = title || "Healthcare Chatbot";
  }

  if (subtitleEl) {
    subtitleEl.textContent =
      subtitle || "Hỗ trợ tư vấn và sàng lọc ban đầu";
  }
}

function upsertHistoryItem(item) {
  const items = getStoredChatHistory();
  const existingIndex = items.findIndex((x) => x.session_id === item.session_id);
  const existing = existingIndex >= 0 ? items[existingIndex] : null;

  const cleanItem = {
    session_id: item.session_id,
    title: item.title || existing?.title || "Cuộc trò chuyện mới",
    preview: item.preview ?? existing?.preview ?? "",
    updated_at: item.updated_at || new Date().toISOString(),
    is_custom_title: item.is_custom_title ?? existing?.is_custom_title ?? false,
    is_pinned: item.is_pinned ?? existing?.is_pinned ?? false,
  };

  if (existing && existing.is_custom_title && !item.force_title_update) {
    cleanItem.title = existing.title;
    cleanItem.is_custom_title = true;
  }

  if (existingIndex >= 0) {
    items[existingIndex] = cleanItem;
  } else {
    items.unshift(cleanItem);
  }

  saveStoredChatHistory(sortHistoryItems(items));
  renderHistoryList();
}

function deleteHistoryItem(sessionId) {
  const items = getStoredChatHistory().filter((item) => item.session_id !== sessionId);
  saveStoredChatHistory(sortHistoryItems(items));

  if (currentSessionId === sessionId) {
    setCurrentSessionId(null);
    buildDefaultWelcome();
  } else {
    renderHistoryList();
  }
}

function togglePinHistoryItem(sessionId) {
  const items = getStoredChatHistory();
  const idx = items.findIndex((item) => item.session_id === sessionId);
  if (idx < 0) return;

  items[idx] = {
    ...items[idx],
    is_pinned: !items[idx].is_pinned,
    updated_at: items[idx].updated_at || new Date().toISOString(),
  };

  saveStoredChatHistory(sortHistoryItems(items));
  renderHistoryList();
}

function clearAllHistory() {
  const confirmed = window.confirm("Em có muốn xóa toàn bộ lịch sử chat trong giao diện không?");
  if (!confirmed) return;

  localStorage.removeItem(CHAT_HISTORY_STORAGE_KEY);
  setCurrentSessionId(null);
  buildDefaultWelcome();
}

function extractSessionMeta(session) {
  const messages = Array.isArray(session?.messages) ? session.messages : [];
  const firstUserMessage = messages.find((msg) => msg.role === "user");
  const lastMessage = messages[messages.length - 1];

  return {
    title: truncateText(firstUserMessage?.content || "Cuộc trò chuyện mới", 50),
    preview: truncateText(lastMessage?.content || "", 72),
    updated_at: session?.updated_at || new Date().toISOString(),
  };
}


function simpleHash(text) {
  let hash = 0;
  const value = String(text || "");

  for (let i = 0; i < value.length; i++) {
    hash = (hash << 5) - hash + value.charCodeAt(i);
    hash |= 0;
  }

  return Math.abs(hash).toString(36);
}

function getStoredFeedback() {
  try {
    const raw = localStorage.getItem(CHAT_FEEDBACK_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch (error) {
    console.error("Không đọc được đánh giá phản hồi:", error);
    return {};
  }
}

function saveStoredFeedback(feedbackMap) {
  localStorage.setItem(CHAT_FEEDBACK_STORAGE_KEY, JSON.stringify(feedbackMap));
}

function getFeedbackKey(content) {
  return `${currentSessionId || "welcome"}:${simpleHash(content)}`;
}

function getFeedbackValue(content) {
  const feedbackMap = getStoredFeedback();
  return feedbackMap[getFeedbackKey(content)]?.value || "";
}

function saveFeedbackValue(content, value) {
  const feedbackMap = getStoredFeedback();
  feedbackMap[getFeedbackKey(content)] = {
    value,
    session_id: currentSessionId || null,
    content_preview: truncateText(content, 160),
    created_at: new Date().toISOString(),
  };
  saveStoredFeedback(feedbackMap);
}

function attachFeedbackControls(container, content) {
  if (!container || !content) return;

  const feedbackWrap = document.createElement("div");
  feedbackWrap.className = "feedback-actions";

  const currentValue = getFeedbackValue(content);

  feedbackWrap.innerHTML = `
    <span class="feedback-label">Đánh giá phản hồi:</span>
    <button class="feedback-btn feedback-up ${currentValue === "up" ? "active" : ""}" type="button">👍 Hữu ích</button>
    <button class="feedback-btn feedback-down ${currentValue === "down" ? "active" : ""}" type="button">👎 Chưa hữu ích</button>
    <span class="feedback-saved">${currentValue ? "Đã lưu đánh giá" : ""}</span>
  `;

  const upBtn = feedbackWrap.querySelector(".feedback-up");
  const downBtn = feedbackWrap.querySelector(".feedback-down");
  const savedLabel = feedbackWrap.querySelector(".feedback-saved");

  function updateFeedback(value) {
    saveFeedbackValue(content, value);
    upBtn.classList.toggle("active", value === "up");
    downBtn.classList.toggle("active", value === "down");
    savedLabel.textContent = "Đã lưu đánh giá";
  }

  upBtn.addEventListener("click", function () {
    updateFeedback("up");
  });

  downBtn.addEventListener("click", function () {
    updateFeedback("down");
  });

  container.appendChild(feedbackWrap);
}

function formatBotMessage(content) {
  let formatted = escapeHtml(content).replace(/\n/g, "<br>");

  formatted = formatted.replace(
    /(Cảnh báo:.*?)(<br><br>|$)/g,
    '<div class="alert-box">$1</div>'
  );

  formatted = formatted.replace(
    /(Trạng thái sàng lọc: Cấp cứu ngay)/g,
    '<strong class="triage-emergency">$1</strong>'
  );

  formatted = formatted.replace(
    /(Trạng thái sàng lọc: Rất khẩn)/g,
    '<strong class="triage-very-urgent">$1</strong>'
  );

  formatted = formatted.replace(
    /(Trạng thái sàng lọc: Khẩn mức vừa)/g,
    '<strong class="triage-urgent">$1</strong>'
  );

  formatted = formatted.replace(
    /(Trạng thái sàng lọc: Khẩn cấp)/g,
    '<strong class="triage-urgent">$1</strong>'
  );

  formatted = formatted.replace(
    /(Trạng thái sàng lọc: Khẩn)/g,
    '<strong class="triage-urgent">$1</strong>'
  );

  formatted = formatted.replace(
    /(Trạng thái sàng lọc: Ưu tiên khám sớm)/g,
    '<strong class="triage-soon">$1</strong>'
  );

  formatted = formatted.replace(
    /(Trạng thái sàng lọc: Thông thường)/g,
    '<strong class="triage-soon">$1</strong>'
  );

  formatted = formatted.replace(
    /(Trạng thái sàng lọc: Tự theo dõi)/g,
    '<strong class="triage-normal">$1</strong>'
  );

  formatted = formatted.replace(
    /(Trạng thái sàng lọc: Theo dõi \/ khám thường)/g,
    '<strong class="triage-normal">$1</strong>'
  );

  formatted = formatted.replace(
    /(Khuyến nghị hành động:.*?)(<br>|$)/g,
    '<div class="action-box">$1</div>'
  );

  formatted = formatted.replace(
    /(Khoa gợi ý:.*?)(<br>|$)/g,
    '<div class="department-box">$1</div>'
  );


  // Ẩn phần trăm bệnh nếu backend cũ vẫn còn trả về dạng "- Tên bệnh: 80% phù hợp tương đối"
  formatted = formatted.replace(
    /- ([^:<br]+):\s*\d+(?:\.\d+)?%\s*(phù hợp tương đối)?/g,
    '- $1'
  );

  // Ẩn dạng "- Tên bệnh (80%)"
  formatted = formatted.replace(
    /- ([^(<br]+)\s*\(\s*\d+(?:\.\d+)?%\s*\)/g,
    '- $1'
  );

  formatted = formatted.replace(
    /(Top 3 bệnh liên quan:<br>(?:- .*?<br>?)+)/g,
    '<div class="disease-box">$1</div>'
  );

  formatted = formatted.replace(
    /(Vì sao hệ thống gợi ý như vậy:<br>(?:- .*?<br>?)+)/g,
    '<div class="reason-box">$1</div>'
  );

  formatted = formatted.replace(
    /(Giải thích (?:RAG theo bệnh nghi ngờ|tham khảo từ kho tri thức):.*)/gs,
    '<div class="rag-box">$1</div>'
  );

  return formatted;
}

function addMessage(content, sender, isLoading = false) {
  const chatBox = getChatBox();
  if (!chatBox) return null;

  const wrapper = document.createElement("div");
  wrapper.classList.add("message", sender);

  const avatar = document.createElement("div");
  avatar.classList.add("message-avatar", sender === "bot" ? "bot-avatar" : "user-avatar");
  avatar.textContent = sender === "bot" ? "🤖" : "👤";

  const contentWrap = document.createElement("div");
  contentWrap.classList.add("message-content");

  const bubble = document.createElement("div");
  bubble.classList.add("message-bubble", sender === "bot" ? "bot-bubble" : "user-bubble");

  if (isLoading) {
    bubble.classList.add("loading-bubble");
  }

  if (sender === "bot") {
    bubble.innerHTML = formatBotMessage(content);
  } else {
    bubble.textContent = content;
  }

  contentWrap.appendChild(bubble);

  if (sender === "bot" && !isLoading) {
    attachFeedbackControls(contentWrap, content);
  }

  wrapper.appendChild(avatar);
  wrapper.appendChild(contentWrap);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;

  return wrapper;
}

function clearChatBox() {
  const chatBox = getChatBox();
  if (chatBox) {
    chatBox.innerHTML = "";
  }
}

function showLoadingMessage() {
  loadingMessageDiv = addMessage("Đang phân tích thông tin...", "bot", true);
}

function removeLoadingMessage() {
  if (loadingMessageDiv) {
    loadingMessageDiv.remove();
    loadingMessageDiv = null;
  }
}

function buildDefaultWelcome() {
  clearChatBox();
  addMessage(
    "Dạ em chào anh/chị. Em là trợ lý y khoa ảo, ở đây để lắng nghe và hỗ trợ ghi nhận tình trạng sức khỏe của mình. Anh/chị đang cảm thấy khó chịu ở đâu nhất và tình trạng này xuất hiện từ lúc nào ạ? Anh/chị cũng có thể cho em biết giới tính và tuổi nếu có thể ạ?", // [CHANGED] Bỏ hỏi mức độ chữ, giữ hỏi điểm đau 0-10.
    "bot"
  );
  setCurrentChatTitle(
    "Healthcare Chatbot",
    "Hỗ trợ tư vấn và sàng lọc ban đầu"
  );
}

function renderHistoryList() {
  const historyList = getHistoryList();
  if (!historyList) return;

  const allItems = sortHistoryItems(getStoredChatHistory());
  const items = filterHistoryItems(allItems);

  if (allItems.length === 0) {
    historyList.innerHTML = `<div class="history-empty">Chưa có cuộc trò chuyện nào.</div>`;
    return;
  }

  if (items.length === 0) {
    historyList.innerHTML = `<div class="history-empty">Không tìm thấy cuộc trò chuyện phù hợp.</div>`;
    return;
  }

  historyList.innerHTML = "";

  items.forEach((item) => {
    const wrapper = document.createElement("div");
    wrapper.className =
      "history-item" +
      (item.session_id === currentSessionId ? " active" : "") +
      (item.is_pinned ? " pinned" : "");

    wrapper.innerHTML = `
      <button class="history-pin-btn ${item.is_pinned ? "active" : ""}" type="button" title="${item.is_pinned ? "Bỏ ghim" : "Ghim cuộc trò chuyện"}">${item.is_pinned ? "📌" : "📍"}</button>
      <button class="history-delete-btn" type="button" title="Xóa cuộc trò chuyện">×</button>
      <div class="history-item-title">${escapeHtml(item.title || "Cuộc trò chuyện")}</div>
      <div class="history-item-preview">${escapeHtml(item.preview || "")}</div>
      <div class="history-item-time">${escapeHtml(formatDateTime(item.updated_at))}</div>
    `;

    wrapper.addEventListener("click", async function () {
      await openHistorySession(item.session_id);
    });

    const pinBtn = wrapper.querySelector(".history-pin-btn");
    if (pinBtn) {
      pinBtn.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        togglePinHistoryItem(item.session_id);
      });
    }

    const deleteBtn = wrapper.querySelector(".history-delete-btn");
    if (deleteBtn) {
      deleteBtn.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();

        const confirmed = window.confirm("Em có muốn xóa cuộc trò chuyện này khỏi lịch sử giao diện không?");
        if (confirmed) {
          deleteHistoryItem(item.session_id);
        }
      });
    }

    historyList.appendChild(wrapper);
  });
}

async function openHistorySession(sessionId) {
  try {
    const sessionData = await getChatSession(sessionId);

    setCurrentSessionId(sessionId);
    clearChatBox();

    if (!sessionData.messages || !Array.isArray(sessionData.messages) || sessionData.messages.length === 0) {
      buildDefaultWelcome();
      return;
    }

    sessionData.messages.forEach((msg) => {
      addMessage(msg.content || "", msg.role === "user" ? "user" : "bot");
    });

    const meta = extractSessionMeta(sessionData);
    const stored = getHistoryItem(sessionId);
    const displayTitle = stored?.title || meta.title;

    setCurrentChatTitle(displayTitle, "Đang xem lại cuộc trò chuyện cũ");

    upsertHistoryItem({
      session_id: sessionId,
      title: displayTitle,
      preview: meta.preview,
      updated_at: meta.updated_at,
      is_custom_title: stored?.is_custom_title ?? false,
      is_pinned: stored?.is_pinned ?? false,
    });
  } catch (error) {
    console.error("Không mở được lịch sử chat:", error);
    addMessage("Không tải được cuộc trò chuyện cũ.", "bot");
  }
}

async function loadChatHistory() {
  renderHistoryList();
  applySidebarState();

  if (!currentSessionId) {
    buildDefaultWelcome();
    return;
  }

  await openHistorySession(currentSessionId);
}

async function handleSendMessage() {
  const input = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");

  if (!input) return;

  const message = input.value.trim();
  if (!message) return;

  addMessage(message, "user");
  input.value = "";
  input.disabled = true;
  if (sendBtn) sendBtn.disabled = true;

  showLoadingMessage();

  const pendingTitle = truncateText(message, 50);

  try {
    const data = await sendMessageToBackend(message, currentSessionId);

    if (data.session_restarted) { // [NEW] Backend báo session cũ đã kết thúc, frontend chuyển sang phiên mới.
      clearChatBox(); // [NEW] Không hiển thị tin nhắn mới nối tiếp phiên đã chẩn đoán xong.
      addMessage(message, "user"); // [NEW] Tin nhắn vừa gửi được đặt vào phiên mới.
    }

    if (data.session_id) {
      setCurrentSessionId(data.session_id);
    }

    removeLoadingMessage();
    addMessage(data.reply || "Hệ thống chưa trả lời được.", "bot");

    const sessionData = await getChatSession(currentSessionId);
    const meta = sessionData
      ? extractSessionMeta(sessionData)
      : {
          title: pendingTitle,
          preview: truncateText(data.reply || "", 72),
          updated_at: new Date().toISOString(),
        };

    const stored = getHistoryItem(currentSessionId);
    const displayTitle = stored?.title || meta.title || pendingTitle;

    setCurrentChatTitle(displayTitle, "Đang trò chuyện");

    upsertHistoryItem({
      session_id: currentSessionId,
      title: displayTitle,
      preview: meta.preview,
      updated_at: meta.updated_at,
      is_custom_title: stored?.is_custom_title ?? false,
      is_pinned: stored?.is_pinned ?? false,
    });
  } catch (error) {
    console.error("Có lỗi khi gửi tin nhắn:", error);
    removeLoadingMessage();
    addMessage("Có lỗi khi kết nối tới backend.", "bot");
  } finally {
    input.disabled = false;
    if (sendBtn) sendBtn.disabled = false;
    input.focus();
  }
}

function startNewChat() {
  setCurrentSessionId(null);
  buildDefaultWelcome();

  const input = document.getElementById("message-input");
  if (input) {
    input.value = "";
    input.focus();
  }
}

function handleHistorySearchInput(event) {
  historySearchTerm = event?.target?.value || "";
  renderHistoryList();
}

function renameCurrentConversation() {
  if (!currentSessionId) {
    window.alert("Em hãy chọn hoặc tạo một cuộc trò chuyện trước.");
    return;
  }

  const existing = getHistoryItem(currentSessionId);
  const currentTitle =
    existing?.title ||
    document.getElementById("current-chat-title")?.textContent ||
    "Cuộc trò chuyện mới";

  const newTitle = window.prompt("Nhập tên mới cho cuộc trò chuyện:", currentTitle);

  if (newTitle === null) return;

  const cleanTitle = newTitle.trim();
  if (!cleanTitle) return;

  upsertHistoryItem({
    session_id: currentSessionId,
    title: cleanTitle,
    preview: existing?.preview || "",
    updated_at: new Date().toISOString(),
    is_custom_title: true,
    is_pinned: existing?.is_pinned ?? false,
    force_title_update: true,
  });

  setCurrentChatTitle(cleanTitle, "Đang trò chuyện");
}

function setVoiceButtonState(isRecording) {
  const voiceBtn = document.getElementById("voice-btn");
  if (!voiceBtn) return;

  if (isRecording) {
    voiceBtn.classList.add("recording");
    voiceBtn.textContent = "⏺";
    voiceBtn.title = "Đang nghe liên tục, bấm lại để dừng";
  } else {
    voiceBtn.classList.remove("recording");
    voiceBtn.textContent = "🎤";
    voiceBtn.title = "Nhập triệu chứng bằng giọng nói";
  }
}

function normalizeVoiceText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[.,!?;:]+$/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function appendUniqueVoiceChunk(chunk) {
  const cleanChunk = String(chunk || "").replace(/\s+/g, " ").trim();
  if (!cleanChunk) return;

  const current = normalizeVoiceText(voiceFinalTranscript);
  const incoming = normalizeVoiceText(cleanChunk);

  if (!incoming) return;

  // Tránh lặp khi Chrome trả lại cùng một đoạn final nhiều lần.
  if (current === incoming || current.endsWith(incoming)) return;

  // Nếu đoạn mới đã bao gồm toàn bộ transcript cũ, thay bằng đoạn đầy đủ hơn.
  if (incoming.startsWith(current) && current.length > 0) {
    voiceFinalTranscript = cleanChunk;
    return;
  }

  voiceFinalTranscript = `${voiceFinalTranscript} ${cleanChunk}`
    .replace(/\s+/g, " ")
    .trim();
}

function buildVoiceInputValue() {
  const input = document.getElementById("message-input");
  if (!input) return;

  input.value = `${voiceBaseText} ${voiceFinalTranscript} ${voiceInterimTranscript}`
    .replace(/\s+/g, " ")
    .trim();

  input.focus();
}

function startVoiceRecognition() {
  if (!activeRecognition) return;

  const input = document.getElementById("message-input");
  voiceIsRecording = true;
  voiceManuallyStopped = false;

  if (!voiceBaseText && input?.value?.trim()) {
    voiceBaseText = input.value.trim();
  }

  setVoiceButtonState(true);

  try {
    activeRecognition.start();
  } catch (error) {
    // Chrome sẽ báo lỗi nếu start() được gọi khi recognition vẫn đang chạy.
    // Trường hợp này không cần hiển thị lỗi cho người dùng.
    console.warn("Voice recognition chưa sẵn sàng để start lại:", error);
  }
}

function stopVoiceRecognition() {
  voiceIsRecording = false;
  voiceManuallyStopped = true;

  if (voiceRestartTimer) {
    clearTimeout(voiceRestartTimer);
    voiceRestartTimer = null;
  }

  // Lưu cả đoạn interim cuối cùng để tránh mất 1-2 từ cuối câu.
  appendUniqueVoiceChunk(voiceInterimTranscript);
  voiceInterimTranscript = "";
  buildVoiceInputValue();

  setVoiceButtonState(false);

  try {
    activeRecognition?.stop();
  } catch (error) {
    console.warn("Không thể dừng nhập giọng nói:", error);
  }
}


function applyQuickIntakeToInput() {
  const ageInput = document.getElementById("quick-age-input");
  const genderSelect = document.getElementById("quick-gender-select");
  const symptomInput = document.getElementById("quick-symptom-input");
  const input = getMessageInput();

  if (!input) return;

  const age = ageInput?.value?.trim() || "";
  const gender = genderSelect?.value?.trim() || "";
  const symptom = symptomInput?.value?.trim() || "";

  if (!age && !gender && !symptom) {
    window.alert("Em hãy nhập ít nhất tuổi, giới tính hoặc triệu chứng chính.");
    return;
  }

  const parts = [];

  if (gender || age) {
    const personInfo = [
      gender ? `giới tính ${gender}` : "",
      age ? `${age} tuổi` : "",
    ].filter(Boolean).join(", ");

    parts.push(`Tôi là người bệnh ${personInfo}.`);
  }

  if (symptom) {
    parts.push(`Triệu chứng chính của tôi là ${symptom}.`);
  }

  const quickText = parts.join(" ");
  input.value = input.value.trim()
    ? `${input.value.trim()} ${quickText}`
    : quickText;

  input.focus();
}

function openDisclaimerModal() {
  const modal = getDisclaimerModal();
  if (modal) {
    modal.classList.remove("hidden");
  }
}

function closeDisclaimerModal() {
  const modal = getDisclaimerModal();
  if (modal) {
    modal.classList.add("hidden");
  }
}

function getConversationExportText() {
  const title =
    document.getElementById("current-chat-title")?.textContent?.trim() ||
    "Healthcare Chatbot";

  const subtitle =
    document.getElementById("current-chat-subtitle")?.textContent?.trim() ||
    "";

  const lines = [
    "HEALTHCARE CHATBOT - NỘI DUNG CUỘC TRÒ CHUYỆN",
    `Tiêu đề: ${title}`,
    subtitle ? `Ghi chú: ${subtitle}` : "",
    `Thời gian xuất: ${new Date().toLocaleString("vi-VN")}`,
    "",
    "Lưu ý: Nội dung này chỉ dùng để tham khảo, không thay thế chẩn đoán hoặc tư vấn của bác sĩ.",
    "",
    "----- NỘI DUNG CHAT -----",
  ].filter(Boolean);

  const messages = Array.from(document.querySelectorAll("#chat-box .message"));

  messages.forEach((message, index) => {
    const role = message.classList.contains("user") ? "Người dùng" : "Chatbot";
    const bubble = message.querySelector(".message-bubble");
    const content = bubble?.innerText?.trim() || "";

    if (content) {
      lines.push("");
      lines.push(`[${index + 1}] ${role}:`);
      lines.push(content);
    }
  });

  const feedbackMap = getStoredFeedback();
  const currentFeedback = Object.values(feedbackMap).filter(
    (item) => !currentSessionId || item.session_id === currentSessionId
  );

  if (currentFeedback.length > 0) {
    lines.push("");
    lines.push("----- ĐÁNH GIÁ PHẢN HỒI -----");

    currentFeedback.forEach((item, index) => {
      const label = item.value === "up" ? "Hữu ích" : "Chưa hữu ích";
      lines.push(`${index + 1}. ${label} - ${item.content_preview || ""}`);
    });
  }

  return lines.join("\n");
}

function exportCurrentConversation() {
  const messages = document.querySelectorAll("#chat-box .message-bubble");

  if (!messages.length) {
    window.alert("Chưa có nội dung để xuất.");
    return;
  }

  const text = getConversationExportText();
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");

  link.href = url;
  link.download = `healthcare-chatbot-${timestamp}.txt`;
  document.body.appendChild(link);
  link.click();
  link.remove();

  URL.revokeObjectURL(url);
}

function initVoiceInput() {
  const voiceBtn = document.getElementById("voice-btn");
  const input = document.getElementById("message-input");

  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!voiceBtn || !input) return;

  if (!SpeechRecognition) {
    voiceBtn.disabled = true;
    voiceBtn.title = "Trình duyệt này chưa hỗ trợ nhập giọng nói";
    return;
  }

  activeRecognition = new SpeechRecognition();
  activeRecognition.lang = "vi-VN";
  activeRecognition.continuous = true;
  activeRecognition.interimResults = true;
  activeRecognition.maxAlternatives = 1;

  activeRecognition.onstart = function () {
    voiceSessionStartedAt = Date.now();
    setVoiceButtonState(true);
  };

  activeRecognition.onresult = function (event) {
    voiceInterimTranscript = "";

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      const transcript = (result[0]?.transcript || "").replace(/\s+/g, " ").trim();
      if (!transcript) continue;

      if (result.isFinal) {
        appendUniqueVoiceChunk(transcript);
      } else {
        // Giữ đoạn đang nghe tạm thời, không ghi đè transcript đã final.
        voiceInterimTranscript = `${voiceInterimTranscript} ${transcript}`
          .replace(/\s+/g, " ")
          .trim();
      }
    }

    buildVoiceInputValue();
  };

  activeRecognition.onerror = function (event) {
    console.warn("Lỗi nhập giọng nói:", event.error);

    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      voiceIsRecording = false;
      voiceManuallyStopped = true;
      setVoiceButtonState(false);
      voiceBtn.disabled = true;
      voiceBtn.title = "Trình duyệt chưa được cấp quyền micro";
      return;
    }

    // Các lỗi như no-speech/audio-capture/network thường làm recognition tự dừng.
    // onend sẽ khởi động lại nếu người dùng chưa chủ động dừng.
  };

  activeRecognition.onend = function () {
    // Nhiều trình duyệt không final toàn bộ câu trước khi onend.
    // Gộp interim cuối để tránh tình trạng nghe được chữ này mất chữ kia.
    appendUniqueVoiceChunk(voiceInterimTranscript);
    voiceInterimTranscript = "";
    buildVoiceInputValue();

    if (voiceIsRecording && !voiceManuallyStopped) {
      setVoiceButtonState(true);

      if (voiceRestartTimer) clearTimeout(voiceRestartTimer);

      // Tự mở lại sau khi Chrome/Edge tự ngắt phiên nghe.
      // Delay ngắn giúp tránh lỗi "recognition already started" nhưng vẫn giảm mất chữ.
      voiceRestartTimer = setTimeout(function () {
        try {
          activeRecognition.start();
        } catch (error) {
          console.warn("Không thể tự khởi động lại nhập giọng nói:", error);
        }
      }, 120);
      return;
    }

    setVoiceButtonState(false);
  };

  voiceBtn.onclick = function () {
    if (voiceIsRecording) {
      stopVoiceRecognition();
      return;
    }

    voiceBaseText = input.value.trim();
    voiceFinalTranscript = "";
    voiceInterimTranscript = "";
    voiceSessionStartedAt = Date.now();
    startVoiceRecognition();
  };
}

function bindUIEvents() {
  const sendBtn = document.getElementById("send-btn");
  const messageInput = document.getElementById("message-input");
  const newChatBtn = document.getElementById("new-chat-btn");
  const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
  const renameChatBtn = document.getElementById("rename-chat-btn");
  const clearHistoryBtn = document.getElementById("clear-history-btn");
  const historySearchInput = getHistorySearchInput();
  const disclaimerBtn = document.getElementById("disclaimer-btn");
  const closeDisclaimerBtn = document.getElementById("close-disclaimer-btn");
  const acceptDisclaimerBtn = document.getElementById("accept-disclaimer-btn");
  const disclaimerModal = getDisclaimerModal();
  const exportChatBtn = document.getElementById("export-chat-btn");
  const applyIntakeBtn = document.getElementById("apply-intake-btn");

  if (sendBtn) {
    sendBtn.onclick = handleSendMessage;
  }

  if (messageInput) {
    messageInput.onkeypress = function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleSendMessage();
      }
    };
  }

  if (newChatBtn) {
    newChatBtn.onclick = startNewChat;
  }

  if (toggleSidebarBtn) {
    toggleSidebarBtn.onclick = toggleSidebar;
  }

  if (renameChatBtn) {
    renameChatBtn.onclick = renameCurrentConversation;
  }

  if (clearHistoryBtn) {
    clearHistoryBtn.onclick = clearAllHistory;
  }

  if (historySearchInput) {
    historySearchInput.oninput = handleHistorySearchInput;
  }

  if (disclaimerBtn) {
    disclaimerBtn.onclick = openDisclaimerModal;
  }

  if (closeDisclaimerBtn) {
    closeDisclaimerBtn.onclick = closeDisclaimerModal;
  }

  if (acceptDisclaimerBtn) {
    acceptDisclaimerBtn.onclick = closeDisclaimerModal;
  }

  if (disclaimerModal) {
    disclaimerModal.addEventListener("click", function (event) {
      if (event.target === disclaimerModal) {
        closeDisclaimerModal();
      }
    });
  }

  if (exportChatBtn) {
    exportChatBtn.onclick = exportCurrentConversation;
  }

  if (applyIntakeBtn) {
    applyIntakeBtn.onclick = applyQuickIntakeToInput;
  }

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      closeDisclaimerModal();
    }
  });
}

function initChatUI() {
  bindUIEvents();
  initVoiceInput();
  loadChatHistory();
}

window.toggleSidebar = toggleSidebar;
window.renameCurrentConversation = renameCurrentConversation;
window.clearAllHistory = clearAllHistory;
window.togglePinHistoryItem = togglePinHistoryItem;
window.startNewChat = startNewChat;
window.handleSendMessage = handleSendMessage;
window.applyQuickIntakeToInput = applyQuickIntakeToInput;
window.openDisclaimerModal = openDisclaimerModal;
window.exportCurrentConversation = exportCurrentConversation;
window.initChatUI = initChatUI;