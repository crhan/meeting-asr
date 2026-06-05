import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built assets land in the Python package's static dir so the FastAPI server serves the
// SPA directly from the wheel -- no node toolchain needed at install time. During dev,
// `npm run dev` serves on :5173 and proxies /api to the running `meeting-asr web` server.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src/app/web/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
});
