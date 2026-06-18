document.addEventListener("DOMContentLoaded", function () {
  if (typeof window.initChatUI === "function") {
    window.initChatUI();
  } else {
    console.error("initChatUI chưa được nạp. Hãy kiểm tra thứ tự script: api.js -> chat.js -> app.js");
  }
});