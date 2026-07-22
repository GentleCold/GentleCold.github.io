#!/usr/bin/env node

const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "../..");
const publicRoot = path.join(root, "public");
const catalogFile = path.join(root, "source/_data/teach-catalog.json");

function contentType(file) {
  return {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
  }[path.extname(file).toLowerCase()] || "application/octet-stream";
}

function startServer() {
  const server = http.createServer((request, response) => {
    const requestPath = decodeURIComponent((request.url || "/").split("?", 1)[0]);
    const relative = requestPath === "/" ? "index.html" : requestPath.replace(/^\/+/, "");
    const candidate = path.resolve(publicRoot, relative);
    if (!candidate.startsWith(`${publicRoot}${path.sep}`) && candidate !== publicRoot) {
      response.writeHead(400);
      response.end("bad path");
      return;
    }
    let file = candidate;
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
    if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
      response.writeHead(404);
      response.end("not found");
      return;
    }
    response.writeHead(200, { "content-type": contentType(file) });
    fs.createReadStream(file).pipe(response);
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

async function main() {
  if (!fs.existsSync(publicRoot) || !fs.existsSync(catalogFile)) {
    throw new Error("Run npm run build and publish a course before e2e validation");
  }
  const catalog = JSON.parse(fs.readFileSync(catalogFile, "utf8"));
  if (!catalog.courses.length) throw new Error("Teach catalog is empty");
  const server = await startServer();
  const address = server.address();
  const baseUrl = `http://127.0.0.1:${address.port}`;
  const browser = await chromium.launch({ headless: true });
  const screenshotDir = fs.mkdtempSync(path.join(os.tmpdir(), "teach-blog-e2e-"));
  try {
    for (const viewport of [
      { name: "phone", width: 390, height: 844 },
      { name: "desktop", width: 1280, height: 900 },
    ]) {
      const context = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height } });
      const page = await context.newPage();
      const errors = [];
      const failedResources = [];
      page.on("pageerror", (error) => {
        // Fluid's existing placeholder LeanCloud config emits this harmless
        // browser error during local checks; local page failures remain fatal.
        if (error.message !== "Failed to fetch") errors.push(error.message);
      });
      page.on("requestfailed", (request) => {
        if (request.url().startsWith(baseUrl)) {
          failedResources.push(`${request.url()}: ${request.failure()?.errorText || "failed"}`);
        }
      });
      async function checkRoute(route) {
        const response = await page.goto(`${baseUrl}${route}`, { waitUntil: "networkidle" });
        if (!response || response.status() >= 400) throw new Error(`${route} returned ${response?.status()}`);
        await page.screenshot({ path: path.join(screenshotDir, `${viewport.name}-${route.replace(/[^a-z0-9]+/gi, "-")}.png`) });
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
        if (overflow) throw new Error(`${route} overflows viewport ${viewport.width}px`);
        if (!(await page.locator("body").innerText()).trim()) throw new Error(`${route} has no visible content`);
      }

      await checkRoute("/teach/");
      for (const course of catalog.courses) {
        await checkRoute(`/teach/${course.course_id}/`);
        const references = new Set();
        for (const lesson of course.lessons) {
          const lessonRoute = `/teach/${course.course_id}/${lesson.id}/`;
          await checkRoute(lessonRoute);
          for (const href of await page.locator('a[href*="/reference/"]').evaluateAll((links) => links.map((link) => link.href))) {
            const url = new URL(href);
            if (url.origin === baseUrl) references.add(url.pathname);
          }
          const quizButton = page.locator(".quiz button").first();
          if (await quizButton.count()) {
            await quizButton.click();
            if (!(await page.locator(".quiz .answer").first().innerText()).trim()) {
              throw new Error(`${lessonRoute} quiz click produced no feedback`);
            }
          }
        }
        for (const reference of references) await checkRoute(reference);
      }
      if (errors.length || failedResources.length) throw new Error(JSON.stringify({ errors, failedResources }));
      await context.close();
    }
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
  console.log(`Teach e2e validation passed; screenshots: ${screenshotDir}`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
