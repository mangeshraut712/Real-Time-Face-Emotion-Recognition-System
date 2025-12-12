import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
    plugins: [
        react()
    ],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
        },
    },
    build: {
        outDir: "../static",
        emptyOutDir: true,
        // Performance optimizations with esbuild (faster than terser)
        minify: 'esbuild',
        target: 'es2015',
        rollupOptions: {
            output: {
                manualChunks: {
                    'react-vendor': ['react', 'react-dom'],
                    'motion': ['framer-motion'],
                    'charts': ['recharts'],
                    'ui-core': ['@radix-ui/react-slot', 'class-variance-authority', 'clsx', 'tailwind-merge'],
                    'icons': ['lucide-react']
                },
                // Optimize chunk names for better caching
                chunkFileNames: 'assets/[name]-[hash].js',
                entryFileNames: 'assets/[name]-[hash].js',
                assetFileNames: 'assets/[name]-[hash].[ext]'
            }
        },
        chunkSizeWarningLimit: 1000,
        // Disable source maps for production
        sourcemap: false,
        // Optimize CSS
        cssCodeSplit: true,
        // Report compressed size
        reportCompressedSize: true
    },
    // Optimize dependencies
    optimizeDeps: {
        include: ['react', 'react-dom', 'framer-motion', 'recharts'],
        exclude: []
    },
    // Performance settings
    server: {
        hmr: {
            overlay: false  // Disable error overlay for better performance
        }
    }
})
