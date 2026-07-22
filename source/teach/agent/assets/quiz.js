document.querySelectorAll("[data-quiz]").forEach((quiz) => {
  const feedback = quiz.querySelector("[data-feedback]");
  const buttons = quiz.querySelectorAll("button[data-answer]");

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const isCorrect = button.dataset.answer === "correct";

      buttons.forEach((candidate) => {
        candidate.classList.remove("correct", "wrong");
        candidate.disabled = isCorrect;
      });

      button.classList.add(isCorrect ? "correct" : "wrong");
      feedback.textContent = isCorrect
        ? quiz.dataset.correctFeedback
        : quiz.dataset.wrongFeedback;
    });
  });
});
