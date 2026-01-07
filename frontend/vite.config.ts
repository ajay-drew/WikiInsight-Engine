import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:9000",
        changeOrigin: true,
        secure: false,
        timeout: 10000,
        configure: (proxy, _options) => {
          proxy.on("error", (err, _req, res) => {
            console.error("Proxy error:", err);
            if (res && !res.headersSent) {
              res.writeHead(500, {
                "Content-Type": "application/json",
              });
              res.end(
                JSON.stringify({
                  error: "Backend server is not running. Please start the API server on port 9000.",
                  details: "Run 'run_app.cmd' or start the backend manually with: uvicorn src.api.main:app --host 127.0.0.1 --port 9000",
                })
              );
            }
          });
          proxy.on("proxyReq", (proxyReq, req, _res) => {
            console.log(`[Proxy] ${req.method} ${req.url} -> http://127.0.0.1:9000${req.url}`);
          });
        },
      },
    },
  },
});


