import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Using a relative base ('./') means the build works whether it's served
// from a domain root (Vercel) or a repo subpath (https://<user>.github.io/<repo>/).
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: 'dist',
  },
});
