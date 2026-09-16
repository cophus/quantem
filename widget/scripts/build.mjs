// Bundle each widget as a self-contained ESM file.
// anywidget loads bundles via Blob URL; relative imports break in that context.
// esbuild flattens everything into one file per widget.

import { build, context } from "esbuild";
import { rmSync, copyFileSync, mkdirSync, existsSync } from "fs";

const watch = process.argv.includes("--watch");
const widgets = [
  { name: "show2d" },
  { name: "show4dstem" },
  { name: "diffsim" },
  // framework-free build of the diffraction simulator for web pages (MyST anywidget directive)
  { name: "diffsim-web", entry: "js/diffsim-web/index.ts", outfile: "dist/diffraction-sim.js" },
];

rmSync("src/quantem/widget/static", { recursive: true, force: true });
mkdirSync("src/quantem/widget/static", { recursive: true });

const baseOpts = {
  bundle: true,
  format: "esm",
  jsx: "automatic",
  target: "es2022",
  define: { "process.env.NODE_ENV": '"production"' },
  loader: { ".css": "text" },
  minify: true,
  sourcemap: false,
  legalComments: "none",
};

for (const w of widgets) {
  const opts = {
    ...baseOpts,
    entryPoints: [w.entry || `js/${w.name}/index.tsx`],
    outfile: w.outfile || `src/quantem/widget/static/${w.name}.js`,
  };
  if (watch) {
    const ctx = await context(opts);
    await ctx.watch();
    console.log(`watching ${w.name}...`);
  } else {
    const start = Date.now();
    await build(opts);
    console.log(`built ${w.name}.js (${Date.now() - start}ms)`);
  }
}

if (!watch) console.log("done.");
