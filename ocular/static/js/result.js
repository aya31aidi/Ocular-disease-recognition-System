(function () {
  "use strict";

  const stage = document.getElementById("viewer");
  if (!stage) return;

  const slider = stage.querySelector(".split-input");
  const setSplit = (value) => stage.style.setProperty("--split", `${value}%`);

  document.querySelectorAll('input[name="view"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.checked) stage.dataset.mode = radio.value;
    });
  });

  if (slider) {
    setSplit(slider.value);
    slider.addEventListener("input", () => setSplit(slider.value));

    // Small sweep on load so the comparison slider is discoverable.
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const duration = 1800;
      let start = null;
      let frame = 0;
      const step = (now) => {
        start ??= now;
        const t = Math.min((now - start) / duration, 1);
        const value = 50 + 24 * Math.sin(t * Math.PI * 2) * (1 - t);
        setSplit(value);
        slider.value = value;
        if (t < 1) frame = requestAnimationFrame(step);
      };
      setTimeout(() => { frame = requestAnimationFrame(step); }, 500);
      slider.addEventListener("pointerdown", () => cancelAnimationFrame(frame), { once: true });
    }
  }

  // Animate the confidence ring from zero.
  const ring = document.querySelector(".ring-meter");
  if (ring) {
    const target = ring.style.getPropertyValue("--p");
    ring.style.setProperty("--p", "0");
    requestAnimationFrame(() => requestAnimationFrame(() => ring.style.setProperty("--p", target)));
  }
})();
