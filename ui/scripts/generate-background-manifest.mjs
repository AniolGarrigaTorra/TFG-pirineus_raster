import { readdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const uiDir = dirname(scriptDir);
const backgroundsDir = join(uiDir, "public", "backgrounds");
const manifestPath = join(backgroundsDir, "manifest.json");
const imageExtensions = new Set([".avif", ".jpeg", ".jpg", ".png", ".webp"]);

function extensionOf(filename) {
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

const entries = await readdir(backgroundsDir, { withFileTypes: true });
const images = entries
  .filter((entry) => entry.isFile())
  .map((entry) => entry.name)
  .filter((filename) => imageExtensions.has(extensionOf(filename)))
  .sort((a, b) => a.localeCompare(b))
  .map((filename) => `/backgrounds/${filename}`);

await writeFile(
  manifestPath,
  `${JSON.stringify({ images }, null, 2)}\n`,
  "utf-8"
);

console.log(`Generated ${manifestPath} with ${images.length} backgrounds.`);
