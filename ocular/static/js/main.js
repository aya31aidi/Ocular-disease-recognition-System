(function () {
  "use strict";

  const root = document.documentElement;

  // Theme: follows the OS until toggled, then remembers the choice.
  const toggle = document.querySelector("[data-theme-toggle]");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const current = root.dataset.theme ||
        (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = current === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked */ }
    });
  }

  // Mobile menu
  const navToggle = document.querySelector(".nav-toggle");
  const navLinks = document.getElementById("nav-links");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      const open = navLinks.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", String(open));
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && navLinks.classList.contains("open")) {
        navLinks.classList.remove("open");
        navToggle.setAttribute("aria-expanded", "false");
        navToggle.focus();
      }
    });
  }

  // Header border once the page scrolls
  const header = document.querySelector(".site-header");
  if (header) {
    const onScroll = () => header.classList.toggle("scrolled", window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
  }

  // Dismissible flash messages, auto-hiding success notes
  document.querySelectorAll(".flash").forEach((flash) => {
    const close = () => {
      flash.style.transition = "opacity .25s";
      flash.style.opacity = "0";
      setTimeout(() => flash.remove(), 250);
    };
    flash.querySelector(".flash-close")?.addEventListener("click", close);
    if (flash.classList.contains("flash-success")) setTimeout(close, 6000);
  });

  // Confirmation for destructive forms
  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (e) => {
      if (!confirm(form.dataset.confirm)) e.preventDefault();
    });
  });

  document.querySelectorAll("[data-print]").forEach((btn) => btn.addEventListener("click", () => window.print()));

  // Scroll reveal
  const reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && reveals.length) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px" });
    reveals.forEach((el) => io.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add("in"));
  }

  // Highlight the current section in the conditions index
  const spyLinks = document.querySelectorAll("[data-spy]");
  if (spyLinks.length && "IntersectionObserver" in window) {
    const byKey = new Map([...spyLinks].map((a) => [a.dataset.spy, a]));
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        spyLinks.forEach((a) => a.classList.remove("active"));
        byKey.get(entry.target.id)?.classList.add("active");
      });
    }, { rootMargin: "-30% 0px -60% 0px" });
    byKey.forEach((_, key) => {
      const section = document.getElementById(key);
      if (section) spy.observe(section);
    });
  }
})();
