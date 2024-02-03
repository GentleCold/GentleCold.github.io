let body = document.querySelector("body");
let htmlCode = '<div id="particles"></div>'
body.insertAdjacentHTML("afterbegin", htmlCode);

particlesJS.load('particles', '/js/particlesjs-config.json', function() {
  console.log('callback - particles.js config loaded');
});
