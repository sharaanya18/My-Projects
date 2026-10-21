// Bundles index.html, the stylesheet and every script into one self-contained
// HTML file (handy for sharing or opening anywhere).
// usage: node eris-lab/tools/build-single.cjs <out.html> [--fragment] [--pack <corpus-pack.json>]
//   --pack      bake a private corpus pack in (the output then contains private data: keep it private)
//   --fragment  omit <!doctype>/<html>/<head>/<body> wrappers (for hosts that add their own)
const fs = require("node:fs");
const path = require("node:path");

const root = path.join(__dirname, "..");
const out = process.argv[2];
if (!out) { console.error("usage: build-single.cjs <out.html> [--fragment]"); process.exit(1); }
const fragment = process.argv.includes("--fragment");
const packArg = process.argv.indexOf("--pack");
const packPath = packArg > 0 ? process.argv[packArg + 1] : null;

let packJs = "";
if (packPath) {
  const pack = JSON.parse(fs.readFileSync(packPath, "utf8"));
  if (pack.format !== "eris-solver-lab-pack") { console.error("not an Eris Solver Lab pack"); process.exit(1); }
  packJs = `<script>\nwindow.Eris = window.Eris || {}; window.Eris.PACK = ${JSON.stringify(pack).replace(/</g, "\\u003c")};\n</script>`;
}

let html = fs.readFileSync(path.join(root, "index.html"), "utf8");
html = html.replace(/<link rel="stylesheet" href="assets\/styles\.css">/, () =>
  `<style>\n${fs.readFileSync(path.join(root, "assets/styles.css"), "utf8")}\n</style>`);
html = html.replace(/<script src="(src\/[^"]+)"><\/script>/g, (_, src) =>
  `<script>\n${fs.readFileSync(path.join(root, src), "utf8").replace(/<\/script/gi, "<\\/script")}\n</script>` +
  // the pack goes in right after the last data file, before any engine or view reads it
  (src === "src/data/learn.js" && packJs ? "\n" + packJs : ""));
if (packPath && !html.includes("Eris.PACK =")) { console.error("could not place the pack"); process.exit(1); }

if (fragment) {
  const head = html.match(/<head>([\s\S]*?)<\/head>/)[1]
    .replace(/<meta charset[^>]*>\s*/, "")
    .replace(/<meta name="viewport"[^>]*>\s*/, "");
  const body = html.match(/<body>([\s\S]*?)<\/body>/)[1];
  html = head.trim() + "\n" + body.trim() + "\n";
}
fs.writeFileSync(out, html);
console.log(`wrote ${out} (${(fs.statSync(out).size / 1024).toFixed(0)} KB)`);
