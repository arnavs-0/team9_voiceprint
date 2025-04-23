export function enroll() {
  const startBtn = document.getElementById("startRecord");
  const timer = document.getElementById("timer");
  const status = document.getElementById("recordingStatus");
  const results = document.getElementById("results");
  const resultMsg = document.getElementById("resultMessage");
  const userName = document.getElementById("userName");
  const nameError = document.getElementById("nameError");

  let mediaRecorder;
  let audioChunks = [];
  let audioContext;

  startBtn.addEventListener("click", async () => {
    if (!userName.value.trim()) {
      nameError.textContent = "Please enter your name";
      return;
    }
    nameError.textContent = "";

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      startBtn.disabled = true;
      userName.disabled = true;
      status.textContent = "Recording...";

      mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
        const reader = new FileReader();

        reader.onload = async (e) => {
          const base64data = e.target.result;
          const name = userName.value.trim();

          try {
            status.textContent = "Sending to server...";
            const response = await fetch("/enroll", {
              method: "POST",
              headers: {
                "Content-Type": "application/x-www-form-urlencoded",
              },
              body: `audio=${encodeURIComponent(
                base64data
              )}&name=${encodeURIComponent(name)}`,
            });

            const data = await response.json();
            resultMsg.textContent = data.message;
            results.classList.remove("hidden");
            status.textContent = "";
          } catch (err) {
            status.textContent = "Error during enrollment: " + err.message;
            console.error("Error during enrollment:", err);
          }
        };

        reader.readAsDataURL(audioBlob);
      };

      mediaRecorder.start();

      let seconds = 3;
      timer.textContent = seconds;
      const countInterval = setInterval(() => {
        seconds--;
        timer.textContent = seconds;

        if (seconds <= 0) {
          clearInterval(countInterval);
          mediaRecorder.stop();
          status.textContent = "Processing...";

          stream.getTracks().forEach((track) => track.stop());
        }
      }, 1000);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      status.textContent = "Error: Could not access microphone";
    }
  });
}

export function users() {
  let userToDelete = null;
  let wavFileToDelete = null;
  const modal = document.getElementById("deleteModal");
  const statusMsg = document.getElementById("status-message");
  const deleteButtons = document.querySelectorAll(".delete-user-btn");

  deleteButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const userId = this.getAttribute("data-user-id");
      const userName = this.getAttribute("data-user-name");
      wavFileToDelete = this.getAttribute("data-wav-file") || null;
      confirmDelete(userId, userName);
    });
  });

  function confirmDelete(userId, userName) {
    userToDelete = userId;
    document.getElementById("deleteUserName").textContent = userName;
    modal.classList.remove("hidden");
  }

  document
    .getElementById("confirmDeleteBtn")
    .addEventListener("click", async () => {
      if (!userToDelete) return;

      try {
        const formData = new FormData();
        formData.append("user_id", userToDelete);
        if (wavFileToDelete) {
          formData.append("wav_file", wavFileToDelete);
        }

        const response = await fetch("/delete_user", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();

        if (data.success) {
          const row = document.getElementById(`user-row-${userToDelete}`);
          if (row) row.remove();

          statusMsg.textContent = `User deleted successfully`;
          statusMsg.classList.remove("hidden");
          statusMsg.classList.remove(
            "bg-red-100",
            "border-red-500",
            "text-red-700"
          );
          statusMsg.classList.add(
            "bg-green-100",
            "border",
            "border-green-500",
            "text-green-700"
          );

          const tbody = document.querySelector("table tbody");
          if (tbody && tbody.children.length === 0) {
            const table = document.querySelector("table");
            const parent = table.parentNode;
            table.remove();
            const emptyMsg = document.createElement("p");
            emptyMsg.className = "text-center py-4";
            emptyMsg.textContent = "No users enrolled yet.";
            parent.appendChild(emptyMsg);
          }
        } else {
          throw new Error("Failed to delete user");
        }
      } catch (err) {
        console.error("Error deleting user:", err);
        statusMsg.textContent = `Error: ${err.message}`;
        statusMsg.classList.remove("hidden");
        statusMsg.classList.remove(
          "bg-green-100",
          "border-green-500",
          "text-green-700"
        );
        statusMsg.classList.add(
          "bg-red-100",
          "border",
          "border-red-500",
          "text-red-700"
        );
      }

      modal.classList.add("hidden");
      userToDelete = null;
      wavFileToDelete = null;

      setTimeout(() => {
        statusMsg.classList.add("hidden");
      }, 5000);
    });

  document.getElementById("cancelDeleteBtn").addEventListener("click", () => {
    modal.classList.add("hidden");
    userToDelete = null;
    wavFileToDelete = null;
  });

  window.onclick = function (event) {
    if (event.target === modal) {
      modal.classList.add("hidden");
      userToDelete = null;
      wavFileToDelete = null;
    }
  };
}

export function verify() {
  const toggleBtn = document.getElementById("toggleVerification");
  const verificationStatus = document.getElementById("verificationStatus");
  const timer = document.getElementById("timer");
  const status = document.getElementById("recordingStatus");
  const attemptsContainer = document.getElementById("attempts-container");
  const commandSection = document.getElementById("command-section");
  const recognizedText = document.getElementById("recognized-text");
  const returnToVerifyBtn = document.getElementById("returnToVerify");
  const instructions = document.getElementById("instructions");

  let mediaRecorder = null;
  let audioChunks = [];
  let attemptCount = 0;
  let isRecording = false;
  let continuousMode = true;
  let stream = null;
  let verificationCooldown = false;
  let commandMode = false;
  let recognitionActive = false;

  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;

  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = "en-US";
    recognition.interimResults = false;

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      recognizedText.textContent = transcript;
      recognitionActive = false;
    };

    recognition.onend = () => {
      if (commandMode && recognitionActive) {
        setTimeout(() => {
          if (commandMode) {
            startCommandRecognition();
          }
        }, 1000);
      }
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error", event.error);
      if (commandMode) {
        recognizedText.textContent = `[Error: ${event.error}. Try again...]`;
        recognitionActive = false;
        setTimeout(() => {
          if (commandMode) {
            startCommandRecognition();
          }
        }, 2000);
      }
    };
  } else {
    console.error("Speech does not work in this browser");
  }

  function startCommandRecognition() {
    if (!recognition) return;

    try {
      recognition.start();
      recognitionActive = true;
      recognizedText.textContent = "Listening for command...";
    } catch (err) {
      console.error("Failed to start recognition:", err);
      setTimeout(startCommandRecognition, 1000);
    }
  }

  function enterCommandMode(userName) {
    commandMode = true;
    continuousMode = false;
    commandSection.classList.remove("hidden");

    verificationStatus.parentNode.classList.add("hidden");
    timer.classList.add("hidden");
    status.textContent = "";

    instructions.textContent = `Welcome, ${userName}! speak commands.`;

    startCommandRecognition();
  }

  returnToVerifyBtn.addEventListener("click", () => {
    commandMode = false;
    commandSection.classList.add("hidden");
    verificationStatus.parentNode.classList.remove("hidden");

    if (recognition && recognitionActive) {
      recognition.stop();
      recognitionActive = false;
    }

    instructions.textContent = "Voice verification is active.";

    continuousMode = true;
    if (!isRecording && !verificationCooldown) {
      startVerificationProcess();
    }
  });

  function resetRecordingUI() {
    timer.textContent = "3";
    timer.classList.add("hidden");
    status.textContent = "";
    isRecording = false;
  }

  function addAttemptToHistory(
    attemptNumber,
    success,
    user = null,
    authScore = null
  ) {
    const attemptDiv = document.createElement("div");

    let statusClass = "";
    let statusMessage = "";

    if (success) {
      statusClass = "bg-green-50 border-l-4 border-green-500";
      statusMessage =
        '<p class="text-green-600 font-bold">Authentication Successful!</p>';
    } else {
      statusClass = "bg-red-50 border-l-4 border-red-500";
      statusMessage =
        '<p class="text-red-600 font-bold">Authentication Failed!</p>';
    }

    attemptDiv.className = `p-4 mb-4 rounded-md shadow-sm ${statusClass}`;

    const timestamp = new Date().toLocaleTimeString();

    let scoreDetails = "";
    if (authScore !== null) {
      scoreDetails = `<p class="mt-2 text-sm">Authentication Score: <span class="font-bold">${authScore.toFixed(
        2
      )}</span></p>`;
    }

    if (success) {
      attemptDiv.innerHTML = `
                        <div class="flex justify-between text-sm mb-2 text-gray-600">
                            <span class="font-bold">Attempt #${attemptNumber}</span>
                            <span class="italic">${timestamp}</span>
                        </div>
                        ${statusMessage}
                        <p>Matched User: ${user}</p>
                        ${scoreDetails}
                    `;

      enterCommandMode(user);
    } else {
      attemptDiv.innerHTML = `
                        <div class="flex justify-between text-sm mb-2 text-gray-600">
                            <span class="font-bold">Attempt #${attemptNumber}</span>
                            <span class="italic">${timestamp}</span>
                        </div>
                        ${statusMessage}
                        <p>Voice not recognized</p>
                        ${scoreDetails}
                    `;
    }

    attemptsContainer.insertBefore(attemptDiv, attemptsContainer.firstChild);

    if (attemptsContainer.children.length > 8) {
      attemptsContainer.removeChild(attemptsContainer.lastChild);
    }
  }

  toggleBtn.addEventListener("click", () => {
    continuousMode = !continuousMode;

    if (continuousMode) {
      toggleBtn.textContent = "Pause Verification";
      verificationStatus.textContent = "Verification Active";
      verificationStatus.classList.add("active");
      if (!isRecording && !verificationCooldown) {
        startVerificationProcess();
      }
    } else {
      toggleBtn.textContent = "Resume Verification";
      verificationStatus.textContent = "Verification Paused";
      verificationStatus.classList.remove("active");
      if (isRecording && mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        if (stream) {
          stream.getTracks().forEach((track) => track.stop());
          stream = null;
        }
      }
    }
  });

  async function startVerificationProcess() {
    if (!continuousMode || isRecording || verificationCooldown || commandMode)
      return;

    try {
      isRecording = true;
      attemptCount++;
      audioChunks = [];

      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      timer.classList.remove("hidden");
      status.textContent = "Listening...";

      mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
        const reader = new FileReader();

        reader.onload = async (e) => {
          const base64data = e.target.result;

          try {
            status.textContent = "Verifying...";
            const response = await fetch("/verify", {
              method: "POST",
              headers: {
                "Content-Type": "application/x-www-form-urlencoded",
              },
              body: `audio=${encodeURIComponent(base64data)}`,
            });

            const data = await response.json();

            addAttemptToHistory(
              attemptCount,
              data.authenticated,
              data.user,
              data.auth_score
            );

            resetRecordingUI();

            if (!data.authenticated && !commandMode) {
              verificationCooldown = true;
              setTimeout(() => {
                verificationCooldown = false;
                if (continuousMode && !commandMode) {
                  startVerificationProcess();
                }
              }, 1000);
            }
          } catch (err) {
            status.textContent = "Error: " + err.message;
            console.error("Verification error:", err);
            resetRecordingUI();

            verificationCooldown = true;
            setTimeout(() => {
              verificationCooldown = false;
              if (continuousMode && !commandMode) {
                startVerificationProcess();
              }
            }, 2000);
          }
        };

        reader.readAsDataURL(audioBlob);
      };

      mediaRecorder.start();

      let seconds = 3;
      timer.textContent = seconds;
      const countInterval = setInterval(() => {
        seconds--;
        timer.textContent = seconds;

        if (seconds <= 0) {
          clearInterval(countInterval);
          if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
          }
          status.textContent = "Processing...";

          if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
          }
        }
      }, 1000);
    } catch (err) {
      console.error("Microphone error:", err);
      status.textContent = "Error: Could not access microphone";
      resetRecordingUI();

      verificationCooldown = true;
      setTimeout(() => {
        verificationCooldown = false;
        if (continuousMode && !commandMode) {
          startVerificationProcess();
        }
      }, 3000);
    }
  }

  setTimeout(() => {
    startVerificationProcess();
  }, 1000);
}
