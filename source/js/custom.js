!(function (e, t, a) {
  function n() {
    c(
      ".heart{width: 10px;height: 10px;position: fixed;background: #f00;transform: rotate(45deg);-webkit-transform: rotate(45deg);-moz-transform: rotate(45deg);}.heart:after,.heart:before{content: '';width: inherit;height: inherit;background: inherit;border-radius: 50%;-webkit-border-radius: 500%;-moz-border-radius: 50%;position: fixed;}.heart:after{top: -5px;}.heart:before{left: -5px;}",
    ),
      o(),
      r();
  }
  function r() {
    for (var e = 0; e < d.length; e++)
      d[e].alpha <= 0
        ? (t.body.removeChild(d[e].el), d.splice(e, 1))
        : (d[e].y--,
          (d[e].scale += 0.004),
          (d[e].alpha -= 0.013),
          (d[e].el.style.cssText =
            "left:" +
            d[e].x +
            "px;top:" +
            d[e].y +
            "px;opacity:" +
            d[e].alpha +
            ";transform:scale(" +
            d[e].scale +
            "," +
            d[e].scale +
            ") rotate(45deg);background:" +
            d[e].color +
            ";z-index:99999"));
    requestAnimationFrame(r);
  }
  function o() {
    var t = "function" == typeof e.onclick && e.onclick;
    e.onclick = function (e) {
      t && t(), i(e);
    };
  }
  function i(e) {
    var a = t.createElement("div");
    (a.className = "heart"),
      d.push({
        el: a,
        x: e.clientX + 20,
        y: e.clientY - 20,
        scale: 1,
        alpha: 1,
        color: s(),
      }),
      t.body.appendChild(a);
  }
  function c(e) {
    var a = t.createElement("style");
    a.type = "text/css";
    try {
      a.appendChild(t.createTextNode(e));
    } catch (t) {
      a.styleSheet.cssText = e;
    }
    t.getElementsByTagName("head")[0].appendChild(a);
  }
  function s() {
    return "#e74c3c";
  }
  var d = [];
  (e.requestAnimationFrame = (function () {
    return (
      e.requestAnimationFrame ||
      e.webkitRequestAnimationFrame ||
      e.mozRequestAnimationFrame ||
      e.oRequestAnimationFrame ||
      e.msRequestAnimationFrame ||
      function (e) {
        setTimeout(e, 1e3 / 60);
      }
    );
  })()),
    n();
})(window, document);

!(function () {
  var start = new Date("2024/02/01 19:19:00");

  function update() {
    var now = new Date();
    now.setTime(now.getTime() + 250);
    days = (now - start) / 1000 / 60 / 60 / 24;
    dnum = Math.floor(days);
    hours = (now - start) / 1000 / 60 / 60 - 24 * dnum;
    hnum = Math.floor(hours);
    if (String(hnum).length === 1) {
      hnum = "0" + hnum;
    }
    minutes = (now - start) / 1000 / 60 - 24 * 60 * dnum - 60 * hnum;
    mnum = Math.floor(minutes);
    if (String(mnum).length === 1) {
      mnum = "0" + mnum;
    }
    seconds =
      (now - start) / 1000 - 24 * 60 * 60 * dnum - 60 * 60 * hnum - 60 * mnum;
    snum = Math.round(seconds);
    if (String(snum).length === 1) {
      snum = "0" + snum;
    }
    document.getElementById("timeDate").innerHTML =
      "本站已运行 " +
      dnum +
      " 天 " +
      hnum +
      " 小时 " +
      mnum +
      " 分 " +
      snum +
      " 秒";
  }

  update();
  setInterval(update, 1000);
})();

!(function () {
  var MIN_SCALE = 0.5;
  var MAX_SCALE = 8;
  var WHEEL_FACTOR = 1.12;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function applyTransform(content, state) {
    content.style.transform =
      "translate(" +
      state.x +
      "px, " +
      state.y +
      "px) scale(" +
      state.scale +
      ")";
  }

  function getBaseSize(svg) {
    var box = null;
    try {
      box = svg.getBBox ? svg.getBBox() : null;
    } catch (e) {
      box = null;
    }

    var width = svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width;
    var height = svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.height;

    width = width || Number(svg.getAttribute("width")) || (box && box.width) || svg.clientWidth;
    height = height || Number(svg.getAttribute("height")) || (box && box.height) || svg.clientHeight;

    return {
      width: Math.ceil(width || 1),
      height: Math.ceil(height || 1),
    };
  }

  function setViewerSize(svg, size) {
    svg.setAttribute("width", size.width);
    svg.setAttribute("height", size.height);
    svg.style.width = size.width + "px";
    svg.style.height = size.height + "px";
    svg.style.maxWidth = "none";
    svg.style.maxHeight = "none";
  }

  function createCloseButton(onClick) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "mermaid-viewer-close";
    button.textContent = "x";
    button.title = "关闭 Mermaid 图";
    button.setAttribute("aria-label", "关闭 Mermaid 图");
    button.addEventListener("click", onClick);
    return button;
  }

  function openMermaidViewer(sourceSvg) {
    var overlay = document.createElement("div");
    var stage = document.createElement("div");
    var content = document.createElement("div");
    var clone = sourceSvg.cloneNode(true);
    var size = getBaseSize(sourceSvg);
    var state = {
      scale: 1,
      x: window.innerWidth / 2 - size.width / 2,
      y: window.innerHeight / 2 - size.height / 2,
      dragging: false,
      startX: 0,
      startY: 0,
      originX: 0,
      originY: 0,
    };

    overlay.className = "mermaid-viewer-overlay";
    stage.className = "mermaid-viewer-stage";
    content.className = "mermaid-viewer-content";
    clone.removeAttribute("id");
    setViewerSize(clone, size);
    content.style.width = size.width + "px";
    content.style.height = size.height + "px";

    function closeViewer() {
      document.removeEventListener("keydown", onKeyDown);
      document.body.classList.remove("mermaid-viewer-open");
      overlay.remove();
    }

    function onKeyDown(event) {
      if (event.key === "Escape") {
        closeViewer();
      }
    }

    function zoomAt(clientX, clientY, nextScale) {
      var oldScale = state.scale;
      var scale = clamp(nextScale, MIN_SCALE, MAX_SCALE);

      state.x = clientX - ((clientX - state.x) / oldScale) * scale;
      state.y = clientY - ((clientY - state.y) / oldScale) * scale;
      state.scale = scale;
      applyTransform(content, state);
    }

    stage.addEventListener("wheel", function (event) {
      event.preventDefault();
      zoomAt(
        event.clientX,
        event.clientY,
        state.scale * (event.deltaY < 0 ? WHEEL_FACTOR : 1 / WHEEL_FACTOR),
      );
    });

    stage.addEventListener("pointerdown", function (event) {
      state.dragging = true;
      state.startX = event.clientX;
      state.startY = event.clientY;
      state.originX = state.x;
      state.originY = state.y;
      stage.setPointerCapture(event.pointerId);
    });

    stage.addEventListener("pointermove", function (event) {
      if (!state.dragging) {
        return;
      }

      state.x = state.originX + event.clientX - state.startX;
      state.y = state.originY + event.clientY - state.startY;
      applyTransform(content, state);
    });

    stage.addEventListener("pointerup", function (event) {
      state.dragging = false;
      stage.releasePointerCapture(event.pointerId);
    });

    stage.addEventListener("pointercancel", function () {
      state.dragging = false;
    });

    overlay.addEventListener("click", function (event) {
      if (event.target === overlay) {
        closeViewer();
      }
    });

    document.addEventListener("keydown", onKeyDown);
    content.appendChild(clone);
    stage.appendChild(content);
    overlay.appendChild(createCloseButton(closeViewer));
    overlay.appendChild(stage);
    document.body.appendChild(overlay);
    document.body.classList.add("mermaid-viewer-open");

    var fitScale = Math.min(
      1,
      (window.innerWidth * 0.86) / size.width,
      (window.innerHeight * 0.82) / size.height,
    );
    state.scale = clamp(fitScale, MIN_SCALE, MAX_SCALE);
    state.x = (window.innerWidth - size.width * state.scale) / 2;
    state.y = (window.innerHeight - size.height * state.scale) / 2;
    applyTransform(content, state);
  }

  function enhanceMermaidViewer(svg) {
    if (svg.dataset.mermaidViewerReady === "true") {
      return;
    }

    svg.dataset.mermaidViewerReady = "true";
    svg.tabIndex = 0;
    svg.setAttribute("role", "button");
    svg.setAttribute("aria-label", "打开 Mermaid 图查看器");
    svg.addEventListener("click", function () {
      openMermaidViewer(svg);
    });
    svg.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openMermaidViewer(svg);
      }
    });
  }

  function enhanceAllMermaidViewers() {
    document.querySelectorAll(".mermaid svg").forEach(enhanceMermaidViewer);
  }

  var enhanceScheduled = false;

  function requestEnhanceAllMermaidViewers() {
    if (enhanceScheduled) {
      return;
    }

    enhanceScheduled = true;
    window.requestAnimationFrame(function () {
      enhanceScheduled = false;
      enhanceAllMermaidViewers();
    });
  }

  function scheduleEnhance() {
    setTimeout(enhanceAllMermaidViewers, 0);
    setTimeout(enhanceAllMermaidViewers, 500);
    setTimeout(enhanceAllMermaidViewers, 1500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleEnhance);
  } else {
    scheduleEnhance();
  }

  if ("MutationObserver" in window) {
    new MutationObserver(function () {
      requestEnhanceAllMermaidViewers();
    }).observe(document.documentElement, {
      childList: true,
      subtree: true,
    });
  }
})();
