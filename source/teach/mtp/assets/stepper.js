document.querySelectorAll("[data-stepper]").forEach((stepper) => {
  const panels = [...stepper.querySelectorAll("[data-stepper-panel]")];
  const progress = stepper.querySelector("[data-stepper-progress]");
  const next = stepper.querySelector("[data-stepper-next]");
  const reset = stepper.querySelector("[data-stepper-reset]");
  let current = 0;

  const render = () => {
    panels.forEach((panel, index) => {
      panel.hidden = index !== current;
    });
    progress.textContent = `步骤 ${current + 1} / ${panels.length}`;
    next.disabled = current === panels.length - 1;
  };

  next.addEventListener("click", () => {
    current = Math.min(current + 1, panels.length - 1);
    render();
  });

  reset.addEventListener("click", () => {
    current = 0;
    render();
  });

  render();
});
