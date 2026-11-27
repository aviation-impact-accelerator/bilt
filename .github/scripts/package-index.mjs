// Generate a PEP 503 package index from GitHub release assets.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const escapeHtml = (value) =>
  value.replaceAll("&", "&amp;").replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;").replaceAll(">", "&gt;");

const page = (title, links) =>
  `<!doctype html>\n<html><head><meta charset="utf-8"><title>${title}</title></head>\n<body>\n${links.join("\n")}\n</body></html>\n`;

export function generateIndex(pages, output, repository) {
  const wheels = new Map();
  for (const release of pages.flat()) {
    if (release.draft || release.prerelease) continue;
    for (const asset of release.assets) {
      if (asset.state !== "uploaded" || !/^bilt-[A-Za-z0-9_.+-]+\.whl$/.test(asset.name)) continue;
      const url = new URL(asset.browser_download_url);
      if (url.origin !== "https://github.com" ||
          !url.pathname.startsWith(`/${repository}/releases/download/`) ||
          decodeURIComponent(url.pathname.split("/").at(-1)) !== asset.name ||
          url.search || url.hash) {
        throw new Error(`Unexpected asset URL for ${asset.name}`);
      }
      if (!/^sha256:[a-f0-9]{64}$/.test(asset.digest ?? "")) {
        throw new Error(`Missing SHA256 digest for ${asset.name}`);
      }
      const href = `${url.href}#sha256=${asset.digest.slice(7)}`;
      if (wheels.has(asset.name) && wheels.get(asset.name) !== href) {
        throw new Error(`Duplicate wheel filename: ${asset.name}`);
      }
      wheels.set(asset.name, href);
    }
  }
  const links = [...wheels.entries()].sort(([a], [b]) => a.localeCompare(b))
    .map(([name, href]) => `<a href="${escapeHtml(href)}">${escapeHtml(name)}</a><br>`);
  mkdirSync(join(output, "bilt"), { recursive: true });
  writeFileSync(join(output, "index.html"), page("Package index", ['<a href="bilt/">bilt</a>']));
  writeFileSync(join(output, "bilt", "index.html"), page("Links for bilt", links));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  generateIndex(JSON.parse(readFileSync(0, "utf8")), process.argv[2], process.env.GITHUB_REPOSITORY);
}
