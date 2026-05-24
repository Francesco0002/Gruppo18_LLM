(function () {
  const faviconHref = "/favicon?v=unisa";
  const rels = ["icon", "shortcut icon", "apple-touch-icon"];

  document
    .querySelectorAll('link[rel="icon"], link[rel="shortcut icon"], link[rel="apple-touch-icon"]')
    .forEach((link) => link.remove());

  for (const rel of rels) {
    const link = document.createElement("link");
    link.rel = rel;
    link.type = "image/png";
    link.href = faviconHref;
    document.head.appendChild(link);
  }

  const resizeChatInput = () => {
    const input = document.querySelector("#chat-input");
    if (!input || input.tagName !== "TEXTAREA") {
      return;
    }

    const maxHeight = Math.max(154, Math.round(window.innerHeight * 0.35));
    input.style.setProperty("height", "auto", "important");
    input.style.setProperty("max-height", `${maxHeight}px`, "important");
    input.style.setProperty(
      "height",
      `${Math.min(input.scrollHeight, maxHeight)}px`,
      "important",
    );
    input.style.setProperty(
      "overflow-y",
      input.scrollHeight > maxHeight ? "auto" : "hidden",
      "important",
    );
  };

  const installTextareaAutosize = () => {
    const input = document.querySelector("#chat-input");
    if (!input || input.tagName !== "TEXTAREA" || input.dataset.diemAutosize === "true") {
      return;
    }

    input.dataset.diemAutosize = "true";
    input.addEventListener("input", () => requestAnimationFrame(resizeChatInput));
    input.addEventListener("change", () => requestAnimationFrame(resizeChatInput));
    input.addEventListener("focus", () => requestAnimationFrame(resizeChatInput));
    resizeChatInput();
  };

  const observer = new MutationObserver(() => {
    installTextareaAutosize();
    resizeChatInput();
  });

  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
  });

  window.addEventListener("resize", resizeChatInput);
  installTextareaAutosize();
  requestAnimationFrame(resizeChatInput);
  setInterval(resizeChatInput, 300);
})();
