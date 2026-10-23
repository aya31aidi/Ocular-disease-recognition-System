(function () {
  "use strict";

  const MAX_BYTES = 10 * 1024 * 1024;
  const ACCEPTED = ["image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"];
  const STAGES = [
    "Cropping to the retinal disc…",
    "Running VGG19 feature extraction…",
    "Scoring eight classes…",
    "Computing Grad-CAM heatmap…",
  ];

  const form = document.getElementById("analyze-form");
  const zone = document.getElementById("dropzone");
  const input = document.getElementById("file-input");
  const img = document.getElementById("preview-img");
  const nameEl = document.getElementById("preview-name");
  const details = document.getElementById("preview-details");
  const clearBtn = document.getElementById("clear-file");
  const submit = document.getElementById("submit-btn");
  const progress = document.getElementById("progress-text");
  if (!form || !input) return;

  const idleText = progress.innerHTML;
  let objectUrl = null;

  const formatSize = (bytes) =>
    bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;

  function showError(message) {
    progress.textContent = message;
    progress.style.color = "var(--bad)";
  }

  function reset() {
    input.value = "";
    zone.classList.remove("has-file");
    submit.disabled = true;
    progress.innerHTML = idleText;
    progress.style.color = "";
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }

  function preview(file) {
    if (!file) return reset();
    // TIFF often has no browser preview, so only block on size and obvious non-images.
    if (file.type && !ACCEPTED.includes(file.type)) {
      reset();
      return showError(`${file.name} is not a supported image type.`);
    }
    if (file.size > MAX_BYTES) {
      reset();
      return showError(`${file.name} is ${formatSize(file.size)}; the limit is 10 MB.`);
    }

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    img.onload = () => {
      details.textContent = `${img.naturalWidth} × ${img.naturalHeight}px · ${formatSize(file.size)}`;
    };
    img.onerror = () => {
      details.textContent = `${formatSize(file.size)} · no browser preview for this format`;
    };
    img.src = objectUrl;
    nameEl.textContent = file.name;
    details.textContent = formatSize(file.size);
    zone.classList.add("has-file");
    submit.disabled = false;
    progress.innerHTML = idleText;
    progress.style.color = "";
  }

  submit.disabled = !input.files.length;
  input.addEventListener("change", () => preview(input.files[0]));
  clearBtn.addEventListener("click", (e) => {
    e.preventDefault();
    reset();
    input.click();
  });

  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (e) => {
      e.preventDefault();
      zone.classList.add("dragging");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, (e) => {
      e.preventDefault();
      zone.classList.remove("dragging");
    })
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    preview(file);
  });

  // Paste an image straight from the clipboard
  document.addEventListener("paste", (e) => {
    const file = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    preview(file);
  });

  form.addEventListener("submit", (e) => {
    if (!input.files.length) {
      e.preventDefault();
      return showError("Choose an image first.");
    }
    form.classList.add("is-analyzing");
    zone.classList.add("is-analyzing");
    submit.disabled = true;
    let stage = 0;
    progress.textContent = STAGES[0];
    setInterval(() => {
      stage = Math.min(stage + 1, STAGES.length - 1);
      progress.textContent = STAGES[stage];
    }, 650);
  });

  // Coming back via the browser's back button restores the page from cache.
  window.addEventListener("pageshow", (e) => {
    if (e.persisted) {
      form.classList.remove("is-analyzing");
      zone.classList.remove("is-analyzing");
      submit.disabled = !input.files.length;
    }
  });
})();
