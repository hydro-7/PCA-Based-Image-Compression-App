document.addEventListener("DOMContentLoaded", function () {
  const toggleBtn = document.getElementById("modeToggle");

  function setTheme(mode) {
    document.body.classList.toggle("dark-mode", mode === "dark");
    localStorage.setItem("theme", mode);
  }

  const saved = localStorage.getItem("theme") || "light";
  setTheme(saved);

  toggleBtn.addEventListener("click", () => {
    const isDark = document.body.classList.contains("dark-mode");
    setTheme(isDark ? "light" : "dark");
  });
});
