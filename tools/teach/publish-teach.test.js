const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const {
  PUBLISH_FILE,
  PublishError,
  publishWorkspace,
} = require("./publish-teach");

function write(file, content) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content, "utf8");
}

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "teach-publisher-test-"));
  const workspace = path.join(root, "teach");
  const blog = path.join(root, "blog");
  write(path.join(workspace, "MISSION.md"), "# Mission: Test Course\n\n## Why\nLearn this on a phone.\n");
  write(path.join(workspace, "lessons/0001-first-lesson.html"), `<!doctype html>
<html><head><title>First lesson</title><link rel="stylesheet" href="../assets/lesson.css"></head>
<body><h1>First lesson</h1><img src="../assets/diagram.png"><p><a href="../reference/0001-reference.html">Reference</a></p>
<section class="quiz"><button onclick="document.querySelector('.answer').textContent = 'ok'">Answer</button><div class="answer"></div></section></body></html>`);
  write(path.join(workspace, "reference/0001-reference.html"), `<!doctype html><html><head><title>Reference</title></head><body><h1>Reference</h1></body></html>`);
  write(path.join(workspace, "assets/lesson.css"), "body { color: black; }\n");
  write(path.join(workspace, "assets/diagram.png"), "not really a png\n");
  write(path.join(blog, "_config.yml"), "title: Test\n");
  write(path.join(blog, "package.json"), "{}\n");
  write(path.join(blog, "source/teach/teach-shell.css"), ".teach-publish-nav {}\n");
  return { root, workspace, blog };
}

test("publishes a course and preserves relative lesson resources", () => {
  const { root, workspace, blog } = fixture();
  const result = publishWorkspace({ workspaceRoot: workspace, blogRoot: blog });
  assert.equal(result.course.course_id, "test-course");
  assert.equal(result.course.lessons.length, 1);
  assert.ok(fs.existsSync(path.join(workspace, PUBLISH_FILE)));
  const lesson = path.join(blog, "source/teach/test-course/0001-first-lesson/index.html");
  assert.ok(fs.existsSync(lesson));
  assert.match(fs.readFileSync(lesson, "utf8"), /teach-publish-nav/);
  assert.ok(fs.existsSync(path.join(blog, "source/teach/test-course/assets/lesson.css")));
  assert.ok(fs.existsSync(path.join(blog, "source/teach/test-course/reference/0001-reference.html")));
  assert.match(fs.readFileSync(path.join(blog, "source/teach/index.md"), "utf8"), /Test Course/);
  fs.rmSync(root, { recursive: true, force: true });
});

test("republishing updates the same lesson without duplicate catalog entries", () => {
  const { root, workspace, blog } = fixture();
  publishWorkspace({ workspaceRoot: workspace, blogRoot: blog });
  const source = path.join(workspace, "lessons/0001-first-lesson.html");
  fs.writeFileSync(source, fs.readFileSync(source, "utf8").replace("First lesson", "Updated lesson"));
  publishWorkspace({ workspaceRoot: workspace, blogRoot: blog });
  const catalog = JSON.parse(fs.readFileSync(path.join(blog, "source/_data/teach-catalog.json"), "utf8"));
  assert.equal(catalog.courses.length, 1);
  assert.equal(catalog.courses[0].lessons.length, 1);
  assert.match(fs.readFileSync(path.join(blog, "source/teach/test-course/0001-first-lesson/index.html"), "utf8"), /Updated lesson/);
  fs.rmSync(root, { recursive: true, force: true });
});

test("rejects a broken local resource before applying output", () => {
  const { root, workspace, blog } = fixture();
  const source = path.join(workspace, "lessons/0001-first-lesson.html");
  fs.writeFileSync(source, fs.readFileSync(source, "utf8").replace("diagram.png", "missing.png"));
  assert.throws(() => publishWorkspace({ workspaceRoot: workspace, blogRoot: blog }), PublishError);
  assert.equal(fs.existsSync(path.join(blog, "source/teach/test-course")), false);
  fs.rmSync(root, { recursive: true, force: true });
});

test("rejects course ID collision across workspaces", () => {
  const first = fixture();
  publishWorkspace({ workspaceRoot: first.workspace, blogRoot: first.blog });
  const secondRoot = fs.mkdtempSync(path.join(os.tmpdir(), "teach-publisher-collision-"));
  const second = path.join(secondRoot, "teach");
  fs.cpSync(first.workspace, second, { recursive: true });
  const metadata = JSON.parse(fs.readFileSync(path.join(second, PUBLISH_FILE), "utf8"));
  metadata.workspace_id = "different-workspace-id-123456";
  fs.writeFileSync(path.join(second, PUBLISH_FILE), JSON.stringify(metadata));
  assert.throws(() => publishWorkspace({ workspaceRoot: second, blogRoot: first.blog }), /Course ID collision/);
  fs.rmSync(first.root, { recursive: true, force: true });
  fs.rmSync(secondRoot, { recursive: true, force: true });
});

test("blocks high confidence credential material", () => {
  const { root, workspace, blog } = fixture();
  fs.appendFileSync(path.join(workspace, "lessons/0001-first-lesson.html"), "\n-----BEGIN PRIVATE KEY-----\n");
  assert.throws(() => publishWorkspace({ workspaceRoot: workspace, blogRoot: blog }), /Likely secret material/);
  fs.rmSync(root, { recursive: true, force: true });
});

test("creates a stable collision-resistant slug for a non-ASCII mission", () => {
  const { root, workspace, blog } = fixture();
  fs.writeFileSync(path.join(workspace, "MISSION.md"), "# Mission: 分布式系统\n\n## Why\n学习系统设计。\n");
  const result = publishWorkspace({ workspaceRoot: workspace, blogRoot: blog });
  assert.match(result.course.course_id, /^teach-course-[a-f0-9]{10}$/);
  fs.rmSync(root, { recursive: true, force: true });
});
