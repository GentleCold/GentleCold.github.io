const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

document.querySelectorAll("[data-speed-tool]").forEach((tool) => {
  const alphaInput = tool.querySelector("[data-alpha]");
  const gammaInput = tool.querySelector("[data-gamma]");
  const costInput = tool.querySelector("[data-cost]");
  const alphaLabel = tool.querySelector("[data-alpha-label]");
  const gammaLabel = tool.querySelector("[data-gamma-label]");
  const costLabel = tool.querySelector("[data-cost-label]");
  const expectedOutput = tool.querySelector("[data-expected]");
  const speedupOutput = tool.querySelector("[data-speedup]");
  const efficiencyOutput = tool.querySelector("[data-efficiency]");
  const bar = tool.querySelector("[data-speed-bar]");

  const update = () => {
    const alpha = clamp(Number(alphaInput.value), 0, 1);
    const gamma = Math.round(clamp(Number(gammaInput.value), 1, 16));
    const cost = clamp(Number(costInput.value), 0, 1);
    const expected = alpha === 1
      ? gamma + 1
      : (1 - alpha ** (gamma + 1)) / (1 - alpha);
    const speedup = expected / (1 + gamma * cost);
    const efficiency = expected / (gamma + 1);

    alphaLabel.textContent = alpha.toFixed(2);
    gammaLabel.textContent = String(gamma);
    costLabel.textContent = cost.toFixed(2);
    expectedOutput.textContent = expected.toFixed(2);
    speedupOutput.textContent = `${speedup.toFixed(2)}x`;
    efficiencyOutput.textContent = `${(efficiency * 100).toFixed(0)}%`;
    bar.style.width = `${Math.min(speedup / 5, 1) * 100}%`;
  };

  [alphaInput, gammaInput, costInput].forEach((input) => {
    input.addEventListener("input", update);
  });
  update();
});

document.querySelectorAll("[data-residual-tool]").forEach((tool) => {
  const rows = [...tool.querySelectorAll("[data-token-row]")];
  const betaOutput = tool.querySelector("[data-beta]");
  const tvOutput = tool.querySelector("[data-tv]");
  const status = tool.querySelector("[data-status]");

  const update = () => {
    const entries = rows.map((row) => ({
      row,
      p: Math.max(0, Number(row.querySelector("[data-p]").value) || 0),
      q: Math.max(0, Number(row.querySelector("[data-q]").value) || 0),
    }));
    const pSum = entries.reduce((sum, entry) => sum + entry.p, 0);
    const qSum = entries.reduce((sum, entry) => sum + entry.q, 0);

    if (Math.abs(pSum - 1) > 0.0001 || Math.abs(qSum - 1) > 0.0001) {
      status.textContent = `p 合计 ${pSum.toFixed(2)}，q 合计 ${qSum.toFixed(2)}；两者都必须为 1。`;
      return;
    }

    const beta = entries.reduce((sum, entry) => sum + Math.min(entry.p, entry.q), 0);
    const rejectMass = 1 - beta;

    entries.forEach(({ row, p, q }) => {
      const acceptance = q === 0 ? null : Math.min(1, p / q);
      const residual = rejectMass === 0 ? 0 : Math.max(0, p - q) / rejectMass;
      row.querySelector("[data-acceptance]").textContent = acceptance === null
        ? "不会被 q 提出"
        : acceptance.toFixed(3);
      row.querySelector("[data-residual]").textContent = residual.toFixed(3);
      row.querySelector("[data-final]").textContent = p.toFixed(3);
    });

    betaOutput.textContent = beta.toFixed(3);
    tvOutput.textContent = (1 - beta).toFixed(3);
    status.textContent = rejectMass === 0
      ? "p 与 q 完全相同，不会进入修正采样。"
      : "分布有效：接受路径与拒绝后的修正路径相加，逐 token 恢复 p。";
  };

  rows.forEach((row) => {
    row.querySelectorAll("input").forEach((input) => input.addEventListener("input", update));
  });
  update();
});
