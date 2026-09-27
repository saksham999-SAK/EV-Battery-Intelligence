import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/predict": "http://127.0.0.1:8001",
      "/explain": "http://127.0.0.1:8001",
      "/batteries": "http://127.0.0.1:8001",
      "/battery-history": "http://127.0.0.1:8001",
      "/battery-forecast": "http://127.0.0.1:8001",
      "/upload-battery": "http://127.0.0.1:8001",
    },
  },
})
