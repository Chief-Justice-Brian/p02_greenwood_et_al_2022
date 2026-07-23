import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const sourceDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(sourceDir, "../..");
const outputDir = resolve(projectRoot, "dist");
const serverDir = resolve(outputDir, "server");
const hostingDir = resolve(outputDir, ".openai");

await rm(outputDir, { recursive: true, force: true });
await mkdir(serverDir, { recursive: true });
await mkdir(hostingDir, { recursive: true });

const files = {};
for (const [route, filename, contentType] of [
  ["/", "index.html", "text/html; charset=utf-8"],
  ["/index.html", "index.html", "text/html; charset=utf-8"],
  ["/styles.css", "styles.css", "text/css; charset=utf-8"],
  ["/app.js", "app.js", "text/javascript; charset=utf-8"],
]) {
  files[route] = {
    body: await readFile(resolve(sourceDir, filename), "utf8"),
    contentType,
  };
}

const serverSource = `const files = ${JSON.stringify(files)};

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const file = files[url.pathname];

    if (!file) {
      return new Response("Not found", {
        status: 404,
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    }

    return new Response(file.body, {
      headers: {
        "content-type": file.contentType,
        "cache-control": url.pathname === "/" || url.pathname === "/index.html"
          ? "no-cache"
          : "public, max-age=3600",
        "x-content-type-options": "nosniff",
      },
    });
  },
};
`;

await writeFile(resolve(serverDir, "index.js"), serverSource);
await cp(
  resolve(projectRoot, ".openai/hosting.json"),
  resolve(hostingDir, "hosting.json"),
);

console.log("Built framework-free site in dist/");
