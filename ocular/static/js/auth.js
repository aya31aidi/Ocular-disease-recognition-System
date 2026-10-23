(function () {
  "use strict";

  document.querySelectorAll("[data-reveal]").forEach((btn) => {
    const field = document.getElementById(btn.dataset.reveal);
    btn.addEventListener("click", () => {
      const hidden = field.type === "password";
      field.type = hidden ? "text" : "password";
      btn.textContent = hidden ? "Hide" : "Show";
    });
  });

  // Rough strength hint: length plus character variety.
  const pwd = document.querySelector("[data-strength]");
  const meter = document.querySelector(".strength");
  if (pwd && meter) {
    pwd.addEventListener("input", () => {
      const v = pwd.value;
      let score = 0;
      if (v.length >= 8) score++;
      if (v.length >= 12) score++;
      if (/[a-z]/.test(v) && /[A-Z]/.test(v)) score++;
      if (/\d/.test(v) && /[^A-Za-z0-9]/.test(v)) score++;
      meter.dataset.score = v ? String(Math.max(score, 1)) : "0";
    });
  }
})();
