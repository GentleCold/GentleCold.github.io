const fs = require('fs');
const path = require('path');
const assert = require('assert');

const root = path.resolve(__dirname, '..');
const customJs = fs.readFileSync(path.join(root, 'source/js/custom.js'), 'utf8');
const customCss = fs.readFileSync(path.join(root, 'source/css/custom.css'), 'utf8');

assert.match(customJs, /enhanceMermaidViewer/, 'custom.js should register Mermaid viewer enhancement');
assert.match(customJs, /querySelectorAll\(['"]\.mermaid svg['"]\)/, 'custom.js should find rendered Mermaid SVGs');
assert.match(customJs, /openMermaidViewer/, 'custom.js should open a fullscreen Mermaid viewer');
assert.match(customJs, /wheel/, 'custom.js should support wheel zoom in the Mermaid viewer');
assert.match(customJs, /pointermove/, 'custom.js should support dragging in the Mermaid viewer');
assert.match(customJs, /scale\("/, 'custom.js should scale Mermaid SVGs in the viewer');
assert.doesNotMatch(customJs, /mermaid-zoom-wrap/, 'custom.js should not wrap inline Mermaid diagrams');
assert.doesNotMatch(customJs, /mermaid-zoom-toolbar/, 'custom.js should not add inline Mermaid toolbars');

assert.match(customCss, /\.mermaid-viewer-overlay/, 'custom.css should style Mermaid fullscreen overlay');
assert.match(customCss, /\.mermaid-viewer-content/, 'custom.css should style Mermaid fullscreen content');
assert.match(customCss, /\.mermaid-viewer-close/, 'custom.css should style Mermaid fullscreen close button');
assert.match(customCss, /\.markdown-body \.mermaid svg/, 'custom.css should only make inline Mermaid diagrams clickable');
assert.doesNotMatch(customCss, /\.mermaid-zoom-wrap/, 'custom.css should not style inline Mermaid wrappers');
assert.doesNotMatch(customCss, /\.mermaid-zoom-toolbar/, 'custom.css should not style inline Mermaid toolbars');
