const API_BASE_URL = "http://127.0.0.1:8000";
const DEVICE_ID_STORAGE_KEY = "healthcare_device_id"; // [NEW] Khóa lưu Device ID độc lập cho từng thiết bị/trình duyệt.

function getOrCreateDeviceId() { // [NEW] Tạo UUID một lần và lưu LocalStorage để backend map session theo thiết bị.
  let deviceId = localStorage.getItem(DEVICE_ID_STORAGE_KEY);

  if (!deviceId) {
    deviceId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(DEVICE_ID_STORAGE_KEY, deviceId);
  }

  return deviceId;
}

async function sendMessageToBackend(message, sessionId = null) {
  const response = await fetch(`${API_BASE_URL}/chat/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      device_id: getOrCreateDeviceId(), // [NEW] Gửi Device ID xuống backend trong mọi request chat.
      message: message,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Không gửi được tin nhắn tới backend: ${response.status} - ${errorText}`);
  }

  return await response.json();
}

async function getChatSession(sessionId) {
  const response = await fetch(`${API_BASE_URL}/chat/session/${sessionId}`);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Không lấy được lịch sử session: ${response.status} - ${errorText}`);
  }

  return await response.json();
}