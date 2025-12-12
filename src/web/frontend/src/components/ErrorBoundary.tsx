import { Component, ErrorInfo, ReactNode } from "react"
import { AlertCircle } from "lucide-react"

interface Props {
    children: ReactNode
}

interface State {
    hasError: boolean
    error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null,
    }

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error }
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error("Uncaught error:", error, errorInfo)
    }

    public render() {
        if (this.state.hasError) {
            return (
                <div className="flex min-h-screen flex-col items-center justify-center bg-zinc-950 p-4 text-center text-white">
                    <div className="mb-4 rounded-full bg-red-500/10 p-4 text-red-500">
                        <AlertCircle className="h-10 w-10" />
                    </div>
                    <h1 className="mb-2 text-2xl font-bold">Something went wrong</h1>
                    <p className="mb-6 max-w-md text-zinc-400">
                        {this.state.error?.message || "An unexpected error occurred."}
                    </p>
                    <button
                        onClick={() => window.location.reload()}
                        className="rounded-lg bg-white px-4 py-2 text-sm font-medium text-black hover:bg-zinc-200"
                    >
                        Reload Page
                    </button>
                </div>
            )
        }

        return this.props.children
    }
}
