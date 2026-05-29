const fs = require('fs');
const path = require('path');
const assert = require('assert');

const root = path.resolve(__dirname, '..');
const customJs = fs.readFileSync(path.join(root, 'source/js/custom.js'), 'utf8');
const customCss = fs.readFileSync(path.join(root, 'source/css/custom.css'), 'utf8');

assert.match(customJs, /enhanceMermaidZoom/, 'custom.js should register Mermaid zoom enhancement');
assert.match(customJs, /querySelectorAll\(['"]\.mermaid svg['"]\)/, 'custom.js should find rendered Mermaid SVGs');
assert.match(customJs, /mermaid-zoom-btn/, 'custom.js should create Mermaid zoom controls');
assert.match(customJs, /transform = ['"]scale\(/, 'custom.js should scale Mermaid SVGs');

assert.match(customCss, /\.mermaid-zoom-wrap/, 'custom.css should style Mermaid zoom wrapper');
assert.match(customCss, /overflow:\s*auto/, 'Mermaid wrapper should allow scrolling large diagrams');
assert.match(customCss, /\.mermaid-zoom-toolbar/, 'custom.css should style Mermaid zoom toolbar');
assert.match(customCss, /\.mermaid-zoom-btn/, 'custom.css should style Mermaid zoom buttons');
