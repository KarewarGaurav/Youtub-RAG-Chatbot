const BACKEND_URL = "https://karewargaurav-youtube-rag-chatbot.hf.space";

let currentVideoId = null;
let isProcessing = false;

// DOM Elements
const serverStatusEl = document.getElementById("serverStatus");
const activeVideoIdEl = document.getElementById("activeVideoId");
const refreshVideoBtn = document.getElementById("refreshVideoBtn");
const chatMessagesEl = document.getElementById("chatMessages");
const userInputEl = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const clearChatBtn = document.getElementById("clearChatBtn");
const errorAlertEl = document.getElementById("errorAlert");
const openSidePanelBtn = document.getElementById("openSidePanelBtn");

// Initialize Extension
document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  detectActiveYouTubeVideo();
  setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
  if (refreshVideoBtn) {
    refreshVideoBtn.addEventListener("click", () => {
      detectActiveYouTubeVideo();
      checkBackendHealth();
    });
  }

  if (sendBtn) {
    sendBtn.addEventListener("click", handleSendMessage);
  }

  if (userInputEl) {
    userInputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    userInputEl.addEventListener("input", () => {
      userInputEl.style.height = "auto";
      userInputEl.style.height = `${Math.min(userInputEl.scrollHeight, 80)}px`;
      updateSendButtonState();
    });
  }

  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", clearChatMessages);
  }

  // Quick Chips Click Event
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const promptText = chip.getAttribute("data-prompt");
      if (promptText && userInputEl) {
        userInputEl.value = promptText;
        updateSendButtonState();
        handleSendMessage();
      }
    });
  });

  // Open Side Panel from popup
  if (openSidePanelBtn) {
    openSidePanelBtn.addEventListener("click", async () => {
      if (chrome?.sidePanel?.open) {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab) {
          chrome.sidePanel.open({ tabId: tab.id });
        }
      } else {
        alert("Side Panel API is supported in Chrome 114+.");
      }
    });
  }
}

// 1. Detect Active YouTube Video from Browser Tab
function detectActiveYouTubeVideo() {
  if (typeof chrome !== "undefined" && chrome.tabs) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && tabs[0].url) {
        const videoId = parseYouTubeVideoId(tabs[0].url);
        if (videoId) {
          currentVideoId = videoId;
          updateVideoIdDisplay(videoId);
          showError(null);
        } else {
          currentVideoId = null;
          updateVideoIdDisplay("Not on a YouTube video tab");
        }
      } else {
        updateVideoIdDisplay("No active tab found");
      }
      updateSendButtonState();
    });
  } else {
    // Fallback for testing outside chrome extension context
    currentVideoId = "JeJ4UOUoxZc";
    updateVideoIdDisplay("JeJ4UOUoxZc (Demo Mode)");
    updateSendButtonState();
  }
}

// Parse YouTube Video ID from URL string
function parseYouTubeVideoId(url) {
  if (!url) return null;
  const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
  const match = url.match(regExp);
  return (match && match[2].length === 11) ? match[2] : null;
}

// Update Active Video UI Display
function updateVideoIdDisplay(idText) {
  if (activeVideoIdEl) {
    activeVideoIdEl.textContent = idText;
  }
}

// Helper: Update API Credits Progress Bar UI
function updateApiCreditsUI(apiUsage) {
  const creditsTextEl = document.getElementById("creditsText");
  const creditProgressBarEl = document.getElementById("creditProgressBar");
  if (!apiUsage) return;

  const used = (apiUsage.used || 0).toLocaleString();
  const limit = (apiUsage.limit || 14400).toLocaleString();
  const percentage = Math.min(100, Math.max(0, apiUsage.percentage_used || 0));

  if (creditsTextEl) {
    creditsTextEl.textContent = `${used} / ${limit} Requests Used`;
  }
  if (creditProgressBarEl) {
    creditProgressBarEl.style.width = `${percentage}%`;
  }
}

// 2. Check Backend Server Status (FastAPI http://localhost:8000)
async function checkBackendHealth() {
  if (!serverStatusEl) return;
  
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      serverStatusEl.className = "status-badge online";
      serverStatusEl.querySelector(".status-text").textContent = "API Connected";
      if (data.api_usage) {
        updateApiCreditsUI(data.api_usage);
      }
    } else {
      throw new Error("Server returned non-200");
    }
  } catch (err) {
    serverStatusEl.className = "status-badge offline";
    serverStatusEl.querySelector(".status-text").textContent = "API Offline (Start FastAPI)";
  }
}

// 3. Handle Send Question
async function handleSendMessage() {
  const questionText = userInputEl.value.trim();
  if (!questionText || !currentVideoId || isProcessing) return;

  // Add User Message to Chat UI
  appendMessage("user", questionText);
  userInputEl.value = "";
  userInputEl.style.height = "auto";
  showError(null);
  
  // Show Typing Indicator
  const typingIndicator = appendTypingIndicator();
  isProcessing = true;
  updateSendButtonState();

  try {
    const response = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        video_id: currentVideoId,
        question: questionText
      })
    });

    const data = await response.json();
    removeTypingIndicator(typingIndicator);

    if (response.ok && data.answer) {
      appendMessage("assistant", data.answer);
      if (data.api_usage) {
        updateApiCreditsUI(data.api_usage);
      }
    } else {
      const errorMsg = data.detail || "Failed to generate answer from transcript.";
      appendMessage("assistant", `⚠️ Error: ${errorMsg}`);
      showError(errorMsg);
    }
  } catch (err) {
    removeTypingIndicator(typingIndicator);
    const errorMsg = "Could not connect to FastAPI server. Please run `uvicorn app:app --reload` on port 8000.";
    appendMessage("assistant", `⚠️ Connection Error: Cannot reach backend server on http://localhost:8000.`);
    showError(errorMsg);
    checkBackendHealth();
  } finally {
    isProcessing = false;
    updateSendButtonState();
  }
}

// Helper: Render Chat Bubbles
function appendMessage(sender, text) {
  if (!chatMessagesEl) return;

  const bubbleContainer = document.createElement("div");
  bubbleContainer.className = `message-bubble ${sender}`;

  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;

  const time = document.createElement("span");
  time.className = "message-time";
  const now = new Date();
  time.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  bubbleContainer.appendChild(content);
  bubbleContainer.appendChild(time);

  chatMessagesEl.appendChild(bubbleContainer);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

// Helper: Show Typing Indicator
function appendTypingIndicator() {
  if (!chatMessagesEl) return null;

  const indicator = document.createElement("div");
  indicator.className = "message-bubble assistant typing-bubble";
  indicator.innerHTML = `
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  chatMessagesEl.appendChild(indicator);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  return indicator;
}

function removeTypingIndicator(indicatorEl) {
  if (indicatorEl && indicatorEl.parentNode) {
    indicatorEl.parentNode.removeChild(indicatorEl);
  }
}

// Helper: Update Send Button State
function updateSendButtonState() {
  if (!sendBtn || !userInputEl) return;
  const hasText = userInputEl.value.trim().length > 0;
  const hasVideo = currentVideoId !== null;
  sendBtn.disabled = !hasText || !hasVideo || isProcessing;
}

// Helper: Display Error Box
function showError(msg) {
  if (!errorAlertEl) return;
  if (msg) {
    errorAlertEl.textContent = msg;
    errorAlertEl.classList.remove("hidden");
  } else {
    errorAlertEl.classList.add("hidden");
  }
}

// Helper: Clear Chat History (Frontend & Backend Checkpoint Memory)
async function clearChatMessages() {
  if (!chatMessagesEl) return;

  if (currentVideoId) {
    try {
      await fetch(`${BACKEND_URL}/clear_history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: currentVideoId })
      });
    } catch (e) {
      console.warn("Could not clear backend memory:", e);
    }
  }

  chatMessagesEl.innerHTML = `
    <div class="system-message">
      <div class="welcome-box">
        <h3>👋 Chat Memory Cleared</h3>
        <p>Ask any new question about the current active YouTube video!</p>
      </div>
    </div>
  `;
}
