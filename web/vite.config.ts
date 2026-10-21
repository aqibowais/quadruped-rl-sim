import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "./",
  server: { host: "127.0.0.1", port: 5175 },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
