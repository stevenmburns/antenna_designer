import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8000", ws: true, changeOrigin: true },
      "/healthz": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/sweep": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/converge": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/pattern": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/geometry": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/examples": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/export_nec": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
