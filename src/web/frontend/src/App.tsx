/**
 * App v2.0 - 2025 Enhanced Version
 * Features:
 * - React Query Provider
 * - Toast Notifications (Sonner)
 * - Theme Provider
 * - Accessibility
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Toaster } from "sonner"
import { Header } from "./components/Header"
import { Dashboard } from "./components/Dashboard"

// Create React Query client with optimized settings
const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: 100, // 100ms
            refetchOnWindowFocus: false,
            retry: 1,
        },
    },
})

function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <div className="min-h-screen bg-background text-foreground antialiased">
                {/* Skip to main content - Accessibility */}
                <a
                    href="#main-content"
                    className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-primary-foreground focus:rounded"
                >
                    Skip to main content
                </a>

                <Header />

                <main id="main-content" role="main" aria-label="Emotion Detection Dashboard">
                    <Dashboard />
                </main>

                {/* Toast Notifications */}
                <Toaster
                    position="bottom-right"
                    toastOptions={{
                        style: {
                            background: 'hsl(var(--background))',
                            color: 'hsl(var(--foreground))',
                            border: '1px solid hsl(var(--border))',
                        },
                    }}
                    richColors
                    closeButton
                />
            </div>
        </QueryClientProvider>
    )
}

export default App
