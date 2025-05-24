const container = document.querySelector(".container");
const chatsContainer = document.querySelector(".chats-container");
const promptForm = document.querySelector(".prompt-form");
const promptInput = promptForm.querySelector(".prompt-input");
const deleteChatsBtn = document.querySelector("#delete-chats-btn");
const priorityContainer = document.querySelector(".priority-container");

const socket = io("http://localhost:5005");

let hasAskedPriorities = false;
let botBuffer = "";
let botBufferTimer = null;
let typingIntervalRef = null;

const scrollToBottom = () =>
  container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });

const createMessageElement = (content, ...classes) => {
  const div = document.createElement("div");
  div.classList.add("message", ...classes);
  div.innerHTML = content;
  return div;
};

const typingEffect = (text, textElement, callback) => {
  textElement.innerHTML = "";
  let index = 0;

  typingIntervalRef = setInterval(() => {
    if (index < text.length) {
      const char = text.charAt(index++);
      textElement.innerHTML += char === "\n" ? "<br>" : char;
      scrollToBottom();
    } else {
      clearInterval(typingIntervalRef);
      document.body.classList.remove("bot-responding");
      if (typeof callback === "function") callback();
    }
  }, 20);
};

const handleFormSubmit = (e) => {
  e.preventDefault();
  const userMessage = promptInput.value.trim();
  if (!userMessage) return;

  promptInput.value = "";
  document.body.classList.add("chats-active", "bot-responding");

  const userMsgDiv = createMessageElement(`<p class="message-text">${userMessage}</p>`, "user-message");
  chatsContainer.appendChild(userMsgDiv);

  const botMsgDiv = createMessageElement(`<img class="avatar" src="images/ss.png" /> <p class="message-text">Just a sec...</p>`, "bot-message", "loading");
  chatsContainer.appendChild(botMsgDiv);

  socket.emit("user_uttered", { message: userMessage });
  scrollToBottom();
};

socket.on("bot_uttered", (response) => {
  document.querySelectorAll(".bot-message.loading").forEach((el) => el.remove());

  const isKeepPrefIntent = response.intent?.name === "keep_preferences";

  if (isKeepPrefIntent) {
    hasAskedPriorities = false; 
  }

  if (
    (response.trigger_scores_ui === true) &&
    !hasAskedPriorities
  ) {
    generatePriorityRequest();
    return;
  }

  if (response.text) {
    botBuffer += response.text + "\n";
    clearTimeout(botBufferTimer);
    botBufferTimer = setTimeout(() => {
      const botMsgDiv = createMessageElement(`
        <img class="avatar" src="images/ss.png" alt="Chatbot Avatar" />
        <p class="message-text">${botBuffer.trim().replace(/\n/g, "<br>")}</p>
      `, "bot-message");

      chatsContainer.appendChild(botMsgDiv);
      typingEffect(botBuffer.trim(), botMsgDiv.querySelector(".message-text"));
      botBuffer = "";
      scrollToBottom();
    }, 500);
  }
});

promptForm.addEventListener("submit", handleFormSubmit);

document.querySelector("#stop-response-btn").addEventListener("click", () => {
  if (typingIntervalRef) clearInterval(typingIntervalRef);
  document.body.classList.remove("bot-responding");
  const loading = chatsContainer.querySelector(".bot-message.loading");
  if (loading) loading.remove();
});

deleteChatsBtn.addEventListener("click", () => {
  chatsContainer.innerHTML = "";
  priorityContainer.innerHTML = "";
  priorityContainer.style.display = "none";
  document.body.classList.remove("chats-active", "bot-responding");
  hasAskedPriorities = false;

  const greetingHeader = document.querySelector(".app-header");
  if (greetingHeader) {
    greetingHeader.remove(); 
    return; 
  }

  if (typeof socket !== "undefined") {
    socket.emit("user_uttered", { message: "/restart" });
  }

  const chatIcon = document.querySelector(".chatbot-icon");
  if (chatIcon) {
    chatIcon.style.display = "none";
  }
});

const showPriorityButtons = () => {
  priorityContainer.innerHTML = "";
  const priorities = [
    "Gaming", "Display", "Battery life", "Build quality", "Performance",
    "Price", "Weight", "Cooling system", "Fans", "Sound", "Graphics"
  ];

  priorities.forEach(priority => {
    const button = document.createElement("button");
    button.classList.add("priority-btn");
    button.textContent = priority;
    button.dataset.priority = priority;
    button.addEventListener("click", () => button.classList.toggle("selected"));
    priorityContainer.appendChild(button);
  });

  const nextBtn = document.createElement("button");
  nextBtn.textContent = "Next";
  nextBtn.classList.add("mt-4", "px-6", "py-2", "bg-[#642CDC]", "text-white", "rounded-full", "font-medium", "hover:bg-[#4f22ad]", "transition");
  priorityContainer.appendChild(nextBtn);
  priorityContainer.style.display = "flex";

  nextBtn.addEventListener("click", () => {
    const selected = Array.from(document.querySelectorAll(".priority-btn.selected"));
    if (selected.length === 0) {
      alert("Please select at least one priority.");
      return;
    }

    priorityContainer.innerHTML = "";

    selected.forEach(btn => {
      const label = btn.dataset.priority;
      const wrapper = document.createElement("div");
      wrapper.classList.add("flex", "flex-col", "items-start", "mb-4", "w-full", "max-w-md");

      const title = document.createElement("label");
      title.textContent = `${label} Priority`;
      title.classList.add("mb-1", "text-sm", "font-medium", "text-gray-800");

      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = "1";
      slider.max = "5";
      slider.value = "3";
      slider.classList.add("w-full", "accent-[#642CDC]", "cursor-pointer");

      const valueDisplay = document.createElement("span");
      valueDisplay.textContent = "3";
      valueDisplay.classList.add("text-sm", "ml-2", "text-[#642CDC]");

      slider.addEventListener("input", () => {
        valueDisplay.textContent = slider.value;
      });

      const sliderRow = document.createElement("div");
      sliderRow.classList.add("flex", "items-center", "gap-3", "w-full");
      sliderRow.appendChild(slider);
      sliderRow.appendChild(valueDisplay);

      wrapper.appendChild(title);
      wrapper.appendChild(sliderRow);
      priorityContainer.appendChild(wrapper);
    });

    const submitBtn = document.createElement("button");
    submitBtn.textContent = "Submit Ratings";
    submitBtn.classList.add("mt-4", "px-6", "py-2", "bg-[#642CDC]", "text-white", "rounded-full", "font-medium", "hover:bg-[#4f22ad]", "transition", "block");
    priorityContainer.appendChild(submitBtn);

    submitBtn.addEventListener("click", () => {
      const ratings = [];
      const allWrappers = priorityContainer.querySelectorAll("div.flex-col");

      allWrappers.forEach(wrapper => {
        const labelEl = wrapper.querySelector("label");
        const slider = wrapper.querySelector("input[type='range']");
        if (labelEl && slider) {
          const label = labelEl.textContent.replace(" Priority", "");
          ratings.push({ priority: label, value: slider.value });
        }
      });

      if (ratings.length === 0) {
        alert("No valid slider data found.");
        return;
      }

      priorityContainer.innerHTML = "";
      const summaryHTML = `
        <img class="avatar" src="images/ss.png" alt="Chatbot Avatar" />
        <p class="message-text">
          Thanks! I've recorded your priorities. Here's what you selected:<br><br>
          ${ratings.map(r => `• <strong>${r.priority}</strong>: ${r.value}/5`).join("<br>")}
        </p>
      `;
      const summaryDiv = createMessageElement(summaryHTML, "bot-message");
      chatsContainer.appendChild(summaryDiv);

      socket.emit("user_uttered", {
        message: "/submit_priorities",
        metadata: { priorities: ratings }
      });

      scrollToBottom();
    });

    scrollToBottom();
  });

  scrollToBottom();
};

const generatePriorityRequest = () => {
  if (hasAskedPriorities) return;

  const botMsgWrapper = document.createElement("div");
  botMsgWrapper.classList.add("bot-message", "message");

  const avatar = document.createElement("img");
  avatar.src = "images/ss.png";
  avatar.alt = "Chatbot Avatar";
  avatar.classList.add("avatar");

  const textWrapper = document.createElement("p");
  textWrapper.classList.add("message-text");

  botMsgWrapper.appendChild(avatar);
  botMsgWrapper.appendChild(textWrapper);
  chatsContainer.appendChild(botMsgWrapper);

  typingEffect("Please select and rate the features that matter most to you in a laptop.", textWrapper, () => {
    showPriorityButtons();
    hasAskedPriorities = true;
    scrollToBottom();
  });
};
