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
  var MAX_SCALE = 3;
  var STEP = 0.25;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function setScale(svg, scale) {
    var nextScale = clamp(scale, MIN_SCALE, MAX_SCALE);
    svg.dataset.mermaidScale = String(nextScale);
    svg.style.transform = "scale(" + nextScale + ")";
    svg.style.transformOrigin = "top left";

    var frame = svg.closest(".mermaid-zoom-frame");
    if (frame) {
      frame.style.width = svg.dataset.mermaidBaseWidth * nextScale + "px";
      frame.style.height = svg.dataset.mermaidBaseHeight * nextScale + "px";
    }
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

  function createButton(label, title, onClick) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "mermaid-zoom-btn";
    button.textContent = label;
    button.title = title;
    button.setAttribute("aria-label", title);
    button.addEventListener("click", onClick);
    return button;
  }

  function enhanceMermaidZoom(svg) {
    if (svg.dataset.mermaidZoomReady === "true") {
      return;
    }

    var mermaid = svg.closest(".mermaid");
    if (!mermaid) {
      return;
    }

    svg.dataset.mermaidZoomReady = "true";
    var size = getBaseSize(svg);
    svg.dataset.mermaidBaseWidth = String(size.width);
    svg.dataset.mermaidBaseHeight = String(size.height);
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    svg.style.width = size.width + "px";
    svg.style.height = size.height + "px";
    svg.style.maxWidth = "none";

    var wrap = document.createElement("div");
    wrap.className = "mermaid-zoom-wrap";

    var toolbar = document.createElement("div");
    toolbar.className = "mermaid-zoom-toolbar";

    var frame = document.createElement("div");
    frame.className = "mermaid-zoom-frame";

    toolbar.appendChild(createButton("-", "缩小 Mermaid 图", function () {
      setScale(svg, Number(svg.dataset.mermaidScale || 1) - STEP);
    }));
    toolbar.appendChild(createButton("+", "放大 Mermaid 图", function () {
      setScale(svg, Number(svg.dataset.mermaidScale || 1) + STEP);
    }));
    toolbar.appendChild(createButton("1:1", "按原始尺寸显示 Mermaid 图", function () {
      setScale(svg, 1);
    }));
    toolbar.appendChild(createButton("Fit", "适应文章宽度显示 Mermaid 图", function () {
      var availableWidth = wrap.clientWidth || size.width;
      setScale(svg, Math.min(1, availableWidth / size.width));
      wrap.scrollLeft = 0;
    }));

    mermaid.insertBefore(wrap, svg);
    wrap.appendChild(toolbar);
    wrap.appendChild(frame);
    frame.appendChild(svg);
    setScale(svg, 1);
  }

  function enhanceAllMermaidZoom() {
    document.querySelectorAll(".mermaid svg").forEach(enhanceMermaidZoom);
  }

  var enhanceScheduled = false;

  function requestEnhanceAllMermaidZoom() {
    if (enhanceScheduled) {
      return;
    }

    enhanceScheduled = true;
    window.requestAnimationFrame(function () {
      enhanceScheduled = false;
      enhanceAllMermaidZoom();
    });
  }

  function scheduleEnhance() {
    setTimeout(enhanceAllMermaidZoom, 0);
    setTimeout(enhanceAllMermaidZoom, 500);
    setTimeout(enhanceAllMermaidZoom, 1500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleEnhance);
  } else {
    scheduleEnhance();
  }

  if ("MutationObserver" in window) {
    new MutationObserver(function () {
      requestEnhanceAllMermaidZoom();
    }).observe(document.documentElement, {
      childList: true,
      subtree: true,
    });
  }
})();
