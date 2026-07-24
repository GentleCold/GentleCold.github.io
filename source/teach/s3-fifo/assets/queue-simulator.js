const TRACE = ["A", "A", "A", "B", "C", "D", "E", "B", "F"];
const CACHE_CAPACITY = 4;
const SMALL_THRESHOLD = 1;
const GHOST_CAPACITY = 3;

const state = {
  cursor: 0,
  small: [],
  main: [],
  ghost: [],
  message: "点击“下一次请求”，观察对象怎样流动。",
};

function find(queue, key) {
  return queue.find((item) => item.key === key);
}

function dataSize() {
  return state.small.length + state.main.length;
}

function pushHead(queue, item) {
  queue.unshift(item);
}

function addGhost(key) {
  state.ghost = state.ghost.filter((item) => item.key !== key);
  pushHead(state.ghost, { key });
  if (state.ghost.length > GHOST_CAPACITY) {
    state.ghost.pop();
  }
}

function evictMain(events) {
  while (state.main.length > 0) {
    const candidate = state.main.pop();
    if (candidate.freq > 0) {
      candidate.freq -= 1;
      pushHead(state.main, candidate);
      events.push(`${candidate.key} 在 M 中还有热度，减 1 后回到队头`);
    } else {
      events.push(`${candidate.key} 从 M 真正淘汰`);
      return;
    }
  }
}

function evictSmall(events) {
  while (state.small.length > 0) {
    const candidate = state.small.pop();
    if (candidate.freq > 1) {
      pushHead(state.main, candidate);
      events.push(`${candidate.key} 在 S 中被重复访问，晋升 M`);
      if (dataSize() > CACHE_CAPACITY) {
        evictMain(events);
      }
    } else {
      addGhost(candidate.key);
      events.push(`${candidate.key} 复用不足：数据丢弃，只在 G 留名字`);
      return;
    }
  }
}

function makeRoom(events) {
  while (dataSize() >= CACHE_CAPACITY) {
    if (state.small.length >= SMALL_THRESHOLD) {
      evictSmall(events);
    } else {
      evictMain(events);
    }
  }
}

function request(key) {
  const events = [];
  const cached = find(state.small, key) || find(state.main, key);
  if (cached) {
    cached.freq = Math.min(cached.freq + 1, 3);
    events.push(`${key} 命中：只把热度计数加到 ${cached.freq}，不移动位置`);
    return events.join("；");
  }

  makeRoom(events);

  const ghostIndex = state.ghost.findIndex((item) => item.key === key);
  if (ghostIndex >= 0) {
    state.ghost.splice(ghostIndex, 1);
    pushHead(state.main, { key, freq: 0 });
    events.push(`${key} 未命中，但 G 记得它：直接进入 M`);
  } else {
    pushHead(state.small, { key, freq: 0 });
    events.push(`${key} 是新对象：进入 S 接受短期观察`);
  }
  return events.join("；");
}

function objectMarkup(item, ghost = false) {
  const className = ghost ? "object ghost-object" : "object";
  const frequency = ghost ? "仅名字" : `f=${item.freq}`;
  return `<span class="${className}">${item.key}<small>${frequency}</small></span>`;
}

function renderQueue(id, queue, ghost = false) {
  document.getElementById(id).innerHTML = queue
    .map((item) => objectMarkup(item, ghost))
    .join("");
}

function render() {
  renderQueue("small-queue", state.small);
  renderQueue("main-queue", state.main);
  renderQueue("ghost-queue", state.ghost, true);
  document.getElementById("event-log").textContent = state.message;

  document.querySelectorAll("#trace span").forEach((node, index) => {
    node.classList.toggle("done", index < state.cursor);
    node.classList.toggle("current", index === state.cursor);
  });

  const nextButton = document.getElementById("next-request");
  nextButton.disabled = state.cursor >= TRACE.length;
  nextButton.textContent = state.cursor >= TRACE.length ? "推演完成" : "下一次请求";
}

function reset() {
  state.cursor = 0;
  state.small = [];
  state.main = [];
  state.ghost = [];
  state.message = "已重置。先猜一猜：第三次 A 会不会改变它在队列中的位置？";
  render();
}

function step() {
  if (state.cursor >= TRACE.length) return;
  const key = TRACE[state.cursor];
  state.message = `请求 ${state.cursor + 1}：${request(key)}`;
  state.cursor += 1;
  render();
}

document.getElementById("trace").innerHTML = TRACE
  .map((key) => `<span>${key}</span>`)
  .join("");
document.getElementById("next-request").addEventListener("click", step);
document.getElementById("reset-simulator").addEventListener("click", reset);

document.querySelectorAll("[data-quiz-answer]").forEach((button) => {
  button.addEventListener("click", () => {
    const correct = button.dataset.quizAnswer === "correct";
    document.querySelectorAll("[data-quiz-answer]").forEach((candidate) => {
      candidate.classList.remove("correct", "wrong");
    });
    button.classList.add(correct ? "correct" : "wrong");
    document.getElementById("quiz-result").textContent = correct
      ? "答对了。S3-FIFO 的第一性洞察就是 quick demotion。"
      : "再想想：论文先观察到了哪一类对象大量占用缓存却不产生收益？";
  });
});

render();
