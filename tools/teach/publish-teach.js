#!/usr/bin/env node

const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");
const process = require("process");
const cheerio = require("cheerio");

const PUBLISH_SCHEMA_VERSION = 1;
const CATALOG_SCHEMA_VERSION = 1;
const PUBLISH_FILE = "PUBLISHING.json";
const COURSE_ID_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const LESSON_FILE_RE = /^(\d{4})-([a-z0-9][a-z0-9-]*)\.html$/;
const SECRET_PATTERNS = [
  /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/,
  /\bAKIA[0-9A-Z]{16}\b/,
  /\b(?:ghp|gho|github_pat)_[A-Za-z0-9_]{20,}\b/,
  /\bxox[baprs]-[A-Za-z0-9-]{20,}\b/,
  /\bsk-[A-Za-z0-9]{20,}\b/,
];

class PublishError extends Error {
  constructor(message, details = []) {
    super(message);
    this.name = "PublishError";
    this.details = details;
  }
}

function fail(message, details = []) {
  throw new PublishError(message, details);
}

function readText(file) {
  return fs.readFileSync(file, "utf8");
}

function writeText(file, content) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content, "utf8");
}

function isWithin(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative === "" || (relative && !relative.startsWith("..") && !path.isAbsolute(relative));
}

function ensureWithin(root, candidate, label) {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(candidate);
  if (!isWithin(resolvedRoot, resolved)) {
    fail(`${label} escapes its workspace: ${candidate}`);
  }
  return resolved;
}

function ensureNoEscapingSymlink(root, candidate) {
  let current = path.resolve(candidate);
  while (isWithin(root, current) && current !== root) {
    if (fs.existsSync(current) && fs.lstatSync(current).isSymbolicLink()) {
      const target = fs.realpathSync(current);
      if (!isWithin(root, target)) {
        fail(`Refusing symlink outside teaching workspace: ${candidate}`);
      }
    }
    current = path.dirname(current);
  }
}

function slugify(value) {
  const slug = value
    .normalize("NFKD")
    .replace(/[^\x00-\x7F]/g, " ")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
  if (slug) return slug;
  const suffix = crypto.createHash("sha256").update(value).digest("hex").slice(0, 10);
  return `teach-course-${suffix}`;
}

function missionTitle(workspaceRoot) {
  const mission = path.join(workspaceRoot, "MISSION.md");
  if (!fs.existsSync(mission)) {
    fail(`Teaching workspace is missing MISSION.md: ${workspaceRoot}`);
  }
  const match = readText(mission).match(/^#\s+Mission:\s*(.+?)\s*$/m);
  if (!match || !match[1]) {
    fail("MISSION.md must contain a '# Mission: ...' heading");
  }
  return match[1].trim();
}

function loadOrCreatePublishingMetadata(workspaceRoot) {
  const file = path.join(workspaceRoot, PUBLISH_FILE);
  let metadata;
  if (fs.existsSync(file)) {
    try {
      metadata = JSON.parse(readText(file));
    } catch (error) {
      fail(`Invalid ${PUBLISH_FILE}: ${error.message}`);
    }
  } else {
    const title = missionTitle(workspaceRoot);
    metadata = {
      schema_version: PUBLISH_SCHEMA_VERSION,
      workspace_id: crypto.randomUUID(),
      course_id: slugify(title),
      title,
    };
    writeText(file, `${JSON.stringify(metadata, null, 2)}\n`);
  }

  if (metadata.schema_version !== PUBLISH_SCHEMA_VERSION) {
    fail(`Unsupported ${PUBLISH_FILE} schema version: ${metadata.schema_version}`);
  }
  if (typeof metadata.workspace_id !== "string" || metadata.workspace_id.length < 16) {
    fail(`${PUBLISH_FILE} requires a stable workspace_id`);
  }
  if (typeof metadata.course_id !== "string" || !COURSE_ID_RE.test(metadata.course_id)) {
    fail(`${PUBLISH_FILE} course_id must be lowercase ASCII slug text`);
  }
  if (typeof metadata.title !== "string" || !metadata.title.trim()) {
    fail(`${PUBLISH_FILE} title must be non-empty`);
  }
  return metadata;
}

function listLessonFiles(workspaceRoot) {
  const lessonsRoot = path.join(workspaceRoot, "lessons");
  if (!fs.existsSync(lessonsRoot)) {
    fail(`Teaching workspace is missing lessons/: ${workspaceRoot}`);
  }
  const files = fs
    .readdirSync(lessonsRoot, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".html"))
    .map((entry) => entry.name)
    .sort();
  if (!files.length) {
    fail(`No lesson HTML files found in ${lessonsRoot}`);
  }

  let previousNumber = 0;
  return files.map((file) => {
    const match = file.match(LESSON_FILE_RE);
    if (!match) {
      fail(`Lesson filename must match NNNN-dash-case.html: ${file}`);
    }
    const number = Number(match[1]);
    if (number !== previousNumber + 1) {
      fail(`Lesson numbering must be contiguous; expected ${String(previousNumber + 1).padStart(4, "0")}, got ${file}`);
    }
    previousNumber = number;
    const source = path.join(lessonsRoot, file);
    ensureNoEscapingSymlink(workspaceRoot, source);
    const html = readText(source);
    const $ = cheerio.load(html, { decodeEntities: false });
    const title = $("title").first().text().trim() || $("h1").first().text().trim();
    if (!title) {
      fail(`Lesson has no <title> or <h1>: ${file}`);
    }
    return {
      file,
      id: file.slice(0, -5),
      number,
      title,
      source,
    };
  });
}

function copyDirectory(source, target, workspaceRoot) {
  if (!fs.existsSync(source)) return;
  ensureWithin(workspaceRoot, source, "Source directory");
  ensureNoEscapingSymlink(workspaceRoot, source);
  fs.mkdirSync(target, { recursive: true });
  for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
    const from = path.join(source, entry.name);
    const to = path.join(target, entry.name);
    ensureNoEscapingSymlink(workspaceRoot, from);
    if (entry.isDirectory()) {
      copyDirectory(from, to, workspaceRoot);
    } else if (entry.isFile()) {
      fs.mkdirSync(path.dirname(to), { recursive: true });
      fs.copyFileSync(from, to);
    } else {
      fail(`Unsupported asset entry (only files/directories allowed): ${from}`);
    }
  }
}

function scanSecrets(root) {
  const findings = [];
  function walk(current) {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const file = path.join(current, entry.name);
      if (entry.isDirectory()) walk(file);
      else if (entry.isFile() && /\.(?:html?|css|js|json|md|txt)$/i.test(entry.name)) {
        const text = readText(file);
        for (const pattern of SECRET_PATTERNS) {
          if (pattern.test(text)) findings.push(`${file}: ${pattern}`);
        }
      }
    }
  }
  walk(root);
  if (findings.length) fail("Likely secret material found in publishable teaching files", findings);
}

function assertSafeUrl(value, sourceFile) {
  const trimmed = value.trim();
  if (!trimmed || trimmed.startsWith("#") || /^(?:https?:|mailto:|data:|javascript:)/i.test(trimmed)) {
    return;
  }
  if (/^(?:file:|\/\/|\/home\/|\/Users\/|[A-Za-z]:[\\/])/.test(trimmed)) {
    fail(`Server-local or absolute resource URL in ${sourceFile}: ${trimmed}`);
  }
}

function validateHtmlReferences(html, sourceFile, outputRoot) {
  const $ = cheerio.load(html, { decodeEntities: false });
  for (const element of $("[src], [href]").toArray()) {
    const attribute = element.attribs.src !== undefined ? "src" : "href";
    const value = element.attribs[attribute];
    assertSafeUrl(value, sourceFile);
    const trimmed = value.trim();
    if (!trimmed || trimmed.startsWith("#") || /^(?:https?:|mailto:|data:|javascript:)/i.test(trimmed)) continue;
    const withoutFragment = trimmed.split("#", 1)[0].split("?", 1)[0];
    const candidate = trimmed.startsWith("/")
      ? path.join(outputRoot, withoutFragment.replace(/^\/+/, ""))
      : path.resolve(path.dirname(sourceFile), withoutFragment);
    if (!isWithin(outputRoot, candidate) || !fs.existsSync(candidate)) {
      fail(`Broken local resource reference in ${sourceFile}: ${value}`);
    }
  }
}

function injectNavigation(html, lesson, course, lessons) {
  const $ = cheerio.load(html, { decodeEntities: false });
  const index = lessons.findIndex((entry) => entry.id === lesson.id);
  const previous = index > 0 ? lessons[index - 1] : null;
  const next = index + 1 < lessons.length ? lessons[index + 1] : null;
  const link = (href, text, className = "") =>
    href ? `<a class="teach-publish-link ${className}" href="${href}">${text}</a>` : '<span class="teach-publish-link teach-publish-link-disabled"></span>';
  const nav = `<nav class="teach-publish-nav" aria-label="Teach lesson navigation">
  <a class="teach-publish-link" href="/teach/">Teach</a>
  <span class="teach-publish-separator" aria-hidden="true">/</span>
  <a class="teach-publish-link" href="/teach/${course.course_id}/">${escapeHtml(course.title)}</a>
  <span class="teach-publish-spacer"></span>
  ${link(previous && `/teach/${course.course_id}/${previous.id}/`, previous ? `上一课：${escapeHtml(previous.title)}` : "", "teach-publish-previous")}
  ${link(next && `/teach/${course.course_id}/${next.id}/`, next ? `下一课：${escapeHtml(next.title)}` : "", "teach-publish-next")}
</nav>`;
  $("head").append('<link rel="stylesheet" href="/teach/teach-shell.css">');
  $("body").prepend(nav);
  const serialized = $.html();
  return /^<!doctype\s+html/i.test(serialized) ? serialized : `<!doctype html>\n${serialized}`;
}

function injectReferenceNavigation(html, course) {
  const $ = cheerio.load(html, { decodeEntities: false });
  const nav = `<nav class="teach-publish-nav" aria-label="Teach reference navigation">
  <a class="teach-publish-link" href="/teach/">Teach</a>
  <span class="teach-publish-separator" aria-hidden="true">/</span>
  <a class="teach-publish-link" href="/teach/${course.course_id}/">${escapeHtml(course.title)}</a>
</nav>`;
  $("head").append('<link rel="stylesheet" href="/teach/teach-shell.css">');
  $("body").prepend(nav);
  const serialized = $.html();
  return /^<!doctype\s+html/i.test(serialized) ? serialized : `<!doctype html>\n${serialized}`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function extractMissionWhy(workspaceRoot) {
  const text = readText(path.join(workspaceRoot, "MISSION.md"));
  const section = text.match(/^##\s+Why\s*\r?\n([\s\S]*?)(?=^##\s|\s*$)/m);
  if (!section) return "";
  return section[1].replace(/\s+/g, " ").trim();
}

function readCatalog(blogRoot) {
  const file = path.join(blogRoot, "source/_data/teach-catalog.json");
  if (!fs.existsSync(file)) return { schema_version: CATALOG_SCHEMA_VERSION, courses: [] };
  let catalog;
  try {
    catalog = JSON.parse(readText(file));
  } catch (error) {
    fail(`Invalid teach catalog: ${error.message}`);
  }
  if (catalog.schema_version !== CATALOG_SCHEMA_VERSION || !Array.isArray(catalog.courses)) {
    fail("Teach catalog has an unsupported shape");
  }
  return catalog;
}

function renderCoursePage(course) {
  const lessonList = course.lessons
    .map((lesson) => `- [${escapeMarkdown(lesson.title)}](/teach/${course.course_id}/${lesson.id}/)`)
    .join("\n");
  return `---\nlayout: page\ntitle: ${yamlQuote(course.title)}\npermalink: /teach/${course.course_id}/\n---\n\n${course.why ? `${course.why}\n\n` : ""}## 课程目录\n\n${lessonList}\n`;
}

function renderTeachPage(catalog) {
  const courses = [...catalog.courses].sort((a, b) => a.title.localeCompare(b.title, "zh-CN"));
  const body = courses.length
    ? courses.map((course) => `## [${escapeMarkdown(course.title)}](/teach/${course.course_id}/)\n\n${course.why || ""}\n\n共 ${course.lessons.length} 节课。\n`).join("\n")
    : "还没有已发布的课程。\n";
  return `---\nlayout: page\ntitle: Teach\npermalink: /teach/\n---\n\n${body}`;
}

function escapeMarkdown(value) {
  return String(value).replace(/[\\\[\]_*`]/g, "\\$&");
}

function yamlQuote(value) {
  return JSON.stringify(String(value));
}

function prepareCourse(workspaceRoot, blogRoot, metadata, lessons, catalog) {
  const existing = catalog.courses.find((course) => course.course_id === metadata.course_id);
  const fingerprint = crypto.createHash("sha256").update(metadata.workspace_id).digest("hex");
  if (existing && existing.source_fingerprint !== fingerprint) {
    fail(`Course ID collision: ${metadata.course_id} already belongs to another teaching workspace`);
  }

  const course = {
    course_id: metadata.course_id,
    title: metadata.title.trim(),
    source_fingerprint: fingerprint,
    why: extractMissionWhy(workspaceRoot),
    lessons: lessons.map(({ id, number, title }) => ({ id, number, title })),
  };
  const catalogCourses = catalog.courses.filter((entry) => entry.course_id !== course.course_id);
  catalogCourses.push(course);
  const nextCatalog = {
    schema_version: CATALOG_SCHEMA_VERSION,
    courses: catalogCourses.sort((a, b) => a.course_id.localeCompare(b.course_id)),
  };

  const staging = fs.mkdtempSync(path.join(os.tmpdir(), "teach-publish-"));
  const stagedSource = path.join(staging, "source");
  const courseRoot = path.join(stagedSource, "teach", metadata.course_id);
  fs.mkdirSync(courseRoot, { recursive: true });
  const shellCss = path.join(blogRoot, "source/teach/teach-shell.css");
  if (!fs.existsSync(shellCss)) {
    fail("Blog is missing source/teach/teach-shell.css");
  }
  fs.mkdirSync(path.join(stagedSource, "teach"), { recursive: true });
  fs.copyFileSync(shellCss, path.join(stagedSource, "teach/teach-shell.css"));
  copyDirectory(path.join(workspaceRoot, "assets"), path.join(courseRoot, "assets"), workspaceRoot);
  copyDirectory(path.join(workspaceRoot, "reference"), path.join(courseRoot, "reference"), workspaceRoot);

  function decorateReferences(current) {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const file = path.join(current, entry.name);
      if (entry.isDirectory()) decorateReferences(file);
      else if (entry.isFile() && file.endsWith(".html")) {
        writeText(file, injectReferenceNavigation(readText(file), metadata));
      }
    }
  }
  if (fs.existsSync(path.join(courseRoot, "reference"))) decorateReferences(path.join(courseRoot, "reference"));

  for (const lesson of lessons) {
    const outputFile = path.join(courseRoot, lesson.id, "index.html");
    writeText(outputFile, injectNavigation(readText(lesson.source), lesson, metadata, lessons));
  }
  writeText(path.join(courseRoot, "index.md"), renderCoursePage(course));
  writeText(path.join(stagedSource, "_data/teach-catalog.json"), `${JSON.stringify(nextCatalog, null, 2)}\n`);
  writeText(path.join(stagedSource, "teach/index.md"), renderTeachPage(nextCatalog));
  scanSecrets(stagedSource);

  const generatedFiles = [];
  function collect(current) {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const file = path.join(current, entry.name);
      if (entry.isDirectory()) collect(file);
      else generatedFiles.push(file);
    }
  }
  collect(stagedSource);
  for (const file of generatedFiles) {
    if (file.endsWith(".html")) validateHtmlReferences(readText(file), file, stagedSource);
  }

  return { staging, stagedSource, catalog: nextCatalog, course };
}

function applyPrepared(blogRoot, prepared) {
  const paths = [
    [path.join(prepared.stagedSource, "_data/teach-catalog.json"), path.join(blogRoot, "source/_data/teach-catalog.json")],
    [path.join(prepared.stagedSource, "teach/index.md"), path.join(blogRoot, "source/teach/index.md")],
    [path.join(prepared.stagedSource, "teach", prepared.course.course_id), path.join(blogRoot, "source/teach", prepared.course.course_id)],
  ];
  for (const [from, to] of paths) {
    if (fs.statSync(from).isDirectory()) {
      fs.mkdirSync(to, { recursive: true });
      copyDirectory(from, to, prepared.stagedSource);
    } else {
      fs.mkdirSync(path.dirname(to), { recursive: true });
      fs.copyFileSync(from, to);
    }
  }
  fs.rmSync(prepared.staging, { recursive: true, force: true });
}

function publishWorkspace({ workspaceRoot, blogRoot }) {
  const workspace = path.resolve(workspaceRoot);
  const blog = path.resolve(blogRoot);
  if (!fs.existsSync(path.join(blog, "_config.yml")) || !fs.existsSync(path.join(blog, "package.json"))) {
    fail(`Not a Hexo blog root: ${blog}`);
  }
  const metadata = loadOrCreatePublishingMetadata(workspace);
  const lessons = listLessonFiles(workspace);
  const catalog = readCatalog(blog);
  const prepared = prepareCourse(workspace, blog, metadata, lessons, catalog);
  applyPrepared(blog, prepared);
  return {
    course: prepared.course,
    files: [
      "source/_data/teach-catalog.json",
      "source/teach/index.md",
      `source/teach/${metadata.course_id}/`,
      path.join(workspace, PUBLISH_FILE),
    ],
  };
}

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--workspace") args.workspaceRoot = argv[++i];
    else if (argv[i] === "--blog-root") args.blogRoot = argv[++i];
    else if (argv[i] === "--help" || argv[i] === "-h") args.help = true;
    else fail(`Unknown argument: ${argv[i]}`);
  }
  return args;
}

if (require.main === module) {
  try {
    const args = parseArgs(process.argv.slice(2));
    if (args.help || !args.workspaceRoot || !args.blogRoot) {
      console.log("Usage: node scripts/publish-teach.js --workspace <teach-dir> --blog-root <hexo-dir>");
      process.exit(args.help ? 0 : 2);
    }
    const result = publishWorkspace(args);
    console.log(`Published ${result.course.title}: ${result.course.lessons.length} lesson(s)`);
    console.log(result.files.join("\n"));
  } catch (error) {
    console.error(`${error.name || "Error"}: ${error.message}`);
    for (const detail of error.details || []) console.error(`  ${detail}`);
    process.exit(1);
  }
}

module.exports = {
  PublishError,
  PUBLISH_FILE,
  applyPrepared,
  listLessonFiles,
  loadOrCreatePublishingMetadata,
  missionTitle,
  publishWorkspace,
  prepareCourse,
  renderCoursePage,
  renderTeachPage,
  scanSecrets,
  validateHtmlReferences,
};
