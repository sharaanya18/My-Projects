// Bundles index.html, the stylesheet and every script into one self-contained
// HTML file (handy for sharing or opening anywhere).
// usage: node eris-lab/tools/build-single.cjs <out.html> [--fragment]
//   --fragment  omit <!doctype>/<html>/<head>/<body> wrappers (for hosts that add their own)
const fs = require("node:fs");
const path = require("node:path");

const root = path.join(__dirname, "..");
const out = process.argv[2];
if (!out) { console.error("usage: build-single.cjs <out.html> [--fragment]"); process.exit(1); }
const fragment = process.argv.includes("--fragment");

let html = fs.readFileSync(path.join(root, "index.html"), "utf8");
html = html.replace(/<link rel="stylesheet" href="assets\/styles\.css">/, () =>
  `<style>\n${fs.readFileSync(path.join(root, "assets/styles.css"), "utf8")}\n</style>`);
html = html.replace(/<script src="(src\/[^"]+)"><\/script>/g, (_, src) =>
  `<script>\n${fs.readFileSync(path.join(root, src), "utf8").replace(/<\/script/gi, "<\\/script")}\n</script>`);

if (fragment) {
  const head = html.match(/<head>([\s\S]*?)<\/head>/)[1]
    .replace(/<meta charset[^>]*>\s*/, "")
    .replace(/<meta name="viewport"[^>]*>\s*/, "");
  const body = html.match(/<body>([\s\S]*?)<\/body>/)[1];
  html = head.trim() + "\n" + body.trim() + "\n";
}
fs.writeFileSync(out, html);
console.log(`wrote ${out} (${(fs.statSync(out).size / 1024).toFixed(0)} KB)`);
