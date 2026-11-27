import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { generateIndex } from "./package-index.mjs";

const repo = "bilt-framework/bilt";
const hash = "a".repeat(64);
const wheel = (version) => ({
  name: `bilt-${version}-cp315-abi3-manylinux_2_28_x86_64.whl`,
  browser_download_url: `https://github.com/${repo}/releases/download/${version}/bilt-${version}-cp315-abi3-manylinux_2_28_x86_64.whl`,
  digest: `sha256:${hash}`,
  state: "uploaded",
});
const release = (assets, extra = {}) => ({ draft: false, prerelease: false, assets, ...extra });

test("index contains stable wheels across all API pages, with hashes", (t) => {
  const dir = mkdtempSync(join(tmpdir(), "bilt-index-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  generateIndex([
    [release([wheel("0.4.0"), { name: "source.tar.gz", state: "uploaded" }])],
    [release([wheel("0.5.0")]), release([wheel("0.6.0")], { draft: true }),
      release([wheel("0.7.0")], { prerelease: true })],
  ], dir, repo);
  assert.match(readFileSync(join(dir, "index.html"), "utf8"), /href="bilt\/"/);
  const html = readFileSync(join(dir, "bilt", "index.html"), "utf8");
  assert.ok(html.includes(wheel("0.4.0").browser_download_url + "#sha256=" + hash));
  assert.ok(html.includes(wheel("0.5.0").name));
  for (const excluded of ["0.6.0", "0.7.0", "source.tar.gz"]) assert.ok(!html.includes(excluded));
});

test("empty index is valid; missing hashes and foreign URLs fail", (t) => {
  const dir = mkdtempSync(join(tmpdir(), "bilt-index-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  generateIndex([[]], dir, repo);
  assert.ok(!readFileSync(join(dir, "bilt", "index.html"), "utf8").includes(".whl"));
  assert.throws(() => generateIndex([[release([{ ...wheel("0.4.0"), digest: null }])]], dir, repo), /digest/);
  assert.throws(() => generateIndex([[release([{ ...wheel("0.4.0"), browser_download_url: "https://example.org/x" }])]], dir, repo), /URL/);
});
