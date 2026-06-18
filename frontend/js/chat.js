const SESSION_STORAGE_KEY = "chat_session_id";
const CHAT_HISTORY_STORAGE_KEY = "chat_session_history";
const SIDEBAR_COLLAPSED_KEY = "chat_sidebar_collapsed";

let currentSessionId = localStorage.getItem(SESSION_STORAGE_KEY) || null;
let loadingMessageDiv = null;

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
      subtitle || "Hỗ trợ thu thập triệu chứng và sàng lọc ban đầu";
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

  items.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
  saveStoredChatHistory(items);
  renderHistoryList();
}

function deleteHistoryItem(sessionId) {
  const items = getStoredChatHistory().filter((item) => item.session_id !== sessionId);
  saveStoredChatHistory(items);

  if (currentSessionId === sessionId) {
    setCurrentSessionId(null);
    buildDefaultWelcome();
  } else {
    renderHistoryList();
  }
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

  wrapper.appendChild(bubble);
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
    "Hỗ trợ thu thập triệu chứng và sàng lọc ban đầu"
  );
}

function renderHistoryList() {
  const historyList = getHistoryList();
  if (!historyList) return;

  const items = getStoredChatHistory();

  if (items.length === 0) {
    historyList.innerHTML = `<div class="history-empty">Chưa có cuộc trò chuyện nào.</div>`;
    return;
  }

  historyList.innerHTML = "";

  items.forEach((item) => {
    const wrapper = document.createElement("div");
    wrapper.className = "history-item" + (item.session_id === currentSessionId ? " active" : "");

    wrapper.innerHTML = `
      <button class="history-delete-btn" type="button" title="Xóa cuộc trò chuyện">×</button>
      <div class="history-item-title">${escapeHtml(item.title || "Cuộc trò chuyện")}</div>
      <div class="history-item-preview">${escapeHtml(item.preview || "")}</div>
      <div class="history-item-time">${escapeHtml(formatDateTime(item.updated_at))}</div>
    `;

    wrapper.addEventListener("click", async function () {
      await openHistorySession(item.session_id);
    });

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
    force_title_update: true,
  });

  setCurrentChatTitle(cleanTitle, "Đang trò chuyện");
}

function bindUIEvents() {
  const sendBtn = document.getElementById("send-btn");
  const messageInput = document.getElementById("message-input");
  const newChatBtn = document.getElementById("new-chat-btn");
  const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
  const renameChatBtn = document.getElementById("rename-chat-btn");
  const clearHistoryBtn = document.getElementById("clear-history-btn");

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
}

function initChatUI() {
  bindUIEvents();
  loadChatHistory();
}

window.toggleSidebar = toggleSidebar;
window.renameCurrentConversation = renameCurrentConversation;
window.clearAllHistory = clearAllHistory;
window.startNewChat = startNewChat;
window.handleSendMessage = handleSendMessage;
window.initChatUI = initChatUI;