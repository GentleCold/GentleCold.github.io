let body = document.querySelector("body");
let htmlCode = '<div id="particles"></div><div id="particles_2"></div>';
body.insertAdjacentHTML("afterbegin", htmlCode);

particlesJS.load("particles", "/js/particlesjs-config.json", function () {
  console.log("callback - particles.js config loaded");
});

let subtitle = document
  .querySelector("#subtitle")
  .getAttribute("data-typed-text");

function generate(value) {
  if (value === 0) {
    particlesJS.load(
      "particles_2",
      "/js/particlesjs-config-2-green.json",
      function () {
        console.log("callback - particles.js config loaded");
      },
    );
  } else if (value === 1) {
    particlesJS.load(
      "particles_2",
      "/js/particlesjs-config-2-orange.json",
      function () {
        console.log("callback - particles.js config loaded");
      },
    );
  } else if (value === 2) {
    particlesJS.load(
      "particles_2",
      "/js/particlesjs-config-2-purple.json",
      function () {
        console.log("callback - particles.js config loaded");
      },
    );
  } else if (value === 3) {
    particlesJS.load(
      "particles_2",
      "/js/particlesjs-config-2-yellow.json",
      function () {
        console.log("callback - particles.js config loaded");
      },
    );
  } else if (value === 4) {
    particlesJS.load(
      "particles_2",
      "/js/particlesjs-config-2.json",
      function () {
        console.log("callback - particles.js config loaded");
      },
    );
  } else {
    particlesJS.load(
      "particles_2",
      "/js/particlesjs-config-2-blue.json",
      function () {
        console.log("callback - particles.js config loaded");
      },
    );
  }
}

function randomBackground(value) {
  if (value === 0) {
    document.querySelector("#banner").style.backgroundImage =
      "url(/imgs/default_1.jpg)";
  } else if (value === 1) {
    document.querySelector("#banner").style.backgroundImage = "url(/imgs/default_2.jpg)";
  } else if (value === 2) {
    document.querySelector("#banner").style.backgroundImage = "url(/imgs/default_3.jpg)";
  } else if (value === 3) {
    document.querySelector("#banner").style.backgroundImage = "url(/imgs/default_5.jpg)";
  } else if (value === 4) {
    document.querySelector("#banner").style.backgroundImage = "url(/imgs/default_4.jpg)";
  } else {
    document.querySelector("#banner").style.backgroundImage = "url(/imgs/default.jpg)";
  }
}

if (subtitle === "归档") {
  generate(0);
} else if (subtitle === "分类") {
  generate(1);
} else if (subtitle === "标签") {
  generate(2);
} else if (subtitle === "友情连接") {
  generate(3);
} else if (subtitle === "关于") {
  generate(4);
} else {
  let random = Math.floor(Math.random() * 6);
  generate(random);
  randomBackground(random);
}
