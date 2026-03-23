const classifyBtn = document.getElementById("classifyBtn");
const textInput = document.getElementById("textInput");
const fileInput = document.getElementById("fileInput");
const loader = document.getElementById("loader");
const errorBox = document.getElementById("errorBox");

const API_BASE = "http://127.0.0.1:5000";

function showLoader() {
  loader.classList.remove("hidden");
}

function hideLoader() {
  loader.classList.add("hidden");
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

classifyBtn.addEventListener("click", async () => {
  hideError();
  showLoader();

  try {
    const text = textInput.value.trim();
    const file = fileInput.files[0];

    let response;
    let result;

    if (text) {
      response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text })
      });
    } else if (file) {
      const formData = new FormData();
      formData.append("file", file);

      response = await fetch(`${API_BASE}/predict-file`, {
        method: "POST",
        body: formData
      });
    } else {
      throw new Error("Please enter text or upload a file.");
    }

    result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Prediction failed.");
    }

    localStorage.setItem("classidocs_result", JSON.stringify(result));
    window.location.href = "results.html";

  } catch (error) {
    showError(error.message);
  } finally {
    hideLoader();
  }
});