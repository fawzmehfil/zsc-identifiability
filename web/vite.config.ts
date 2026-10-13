import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: process.env.GITHUB_ACTIONS ? "/zsc-identifiability/" : "/",
  build: {
    target: "es2022",
    sourcemap: true,
  },
  test: {
    environment: "node",
  },
});
