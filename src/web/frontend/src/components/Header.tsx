import { motion } from "framer-motion"
import {
    Sparkles,
    Moon,
    Sun,
    Github,
    Volume2,
    VolumeX,
    Square,
    Circle,
    Trash2,
    Keyboard,
    FileJson,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useDashboardStore } from "../store"
import { format } from "date-fns"
import { toast } from "sonner"

export function Header() {
    const {
        isStreaming,
        soundEnabled,
        toggleSound,
        isRecording,
        toggleRecording,
        emotionsHistory,
        clearEmotions
    } = useDashboardStore()

    const toggleTheme = () => {
        document.documentElement.classList.toggle("dark")
    }

    const exportSession = () => {
        if (emotionsHistory.length === 0) {
            toast.error('No data to export')
            return
        }

        const data = {
            exportedAt: new Date().toISOString(),
            history: emotionsHistory,
            stats: {
                totalEmotions: emotionsHistory.length,
            }
        }

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `emotion - session - ${format(new Date(), 'yyyy-MM-dd-HH-mm')}.json`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
        toast.success('Session exported')
    }

    return (
        <motion.header
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
        >
            <div className="container flex h-12 md:h-14 max-w-screen-2xl items-center">
                <div className="mr-4 flex items-center gap-3">
                    <div className="h-8 w-8 rounded-xl bg-gradient-to-br from-primary to-purple-600 text-white flex items-center justify-center shadow-lg shadow-primary/20">
                        <Sparkles className="h-5 w-5" />
                    </div>
                    <span className="hidden font-bold sm:inline-block text-xl bg-clip-text text-transparent bg-gradient-to-r from-foreground to-foreground/70">
                        Emotion AI
                    </span>
                    <Badge
                        variant={isStreaming ? "default" : "secondary"}
                        className="hidden sm:flex items-center gap-1 px-2 py-0.5"
                    >
                        <span
                            className={`h-1.5 w-1.5 rounded-full ${
                                isStreaming ? "bg-emerald-500" : "bg-muted-foreground/40"
                            }`}
                        />
                        {isStreaming ? "LIVE" : "OFFLINE"}
                    </Badge>
                    <span className="hidden md:inline text-xs text-muted-foreground">
                        {emotionsHistory.length} detections
                    </span>
                </div>

                <div className="flex flex-1 items-center justify-end space-x-2">
                    {/* Dashboard Controls */}
                    <div className="flex items-center space-x-1 mr-2 md:mr-4 border-r pr-2 md:pr-4 border-border/40 overflow-x-auto">
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={toggleSound}
                            title={soundEnabled ? "Mute Sound (s)" : "Enable Sound (s)"}
                        >
                            {soundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4 text-muted-foreground" />}
                        </Button>

                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={toggleRecording}
                            className={isRecording ? "text-red-500 hover:text-red-600 bg-red-500/10" : ""}
                            title={isRecording ? "Stop Recording" : "Start Recording"}
                        >
                            {isRecording ? <Square className="h-4 w-4 fill-current" /> : <Circle className="h-4 w-4" />}
                        </Button>

                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={exportSession}
                            disabled={emotionsHistory.length === 0}
                            title="Export Session"
                            className="hidden sm:inline-flex"
                        >
                            <FileJson className="h-4 w-4" />
                        </Button>

                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={clearEmotions}
                            disabled={emotionsHistory.length === 0}
                            title="Clear History (x)"
                            className="hidden sm:inline-flex"
                        >
                            <Trash2 className="h-4 w-4" />
                        </Button>

                        <Button
                            variant="ghost"
                            size="icon"
                            title="Keyboard Shortcuts"
                            onClick={() => toast.info('Shortcuts', { description: 'Ctrl+C: Camera | Ctrl+S: Sound | Ctrl+E: Export | Ctrl+X: Clear' })}
                            className="hidden md:inline-flex"
                        >
                            <Keyboard className="h-4 w-4" />
                        </Button>
                    </div>

                    <nav className="flex items-center space-x-1">
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-9 w-9"
                            onClick={toggleTheme}
                        >
                            <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                            <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                            <span className="sr-only">Toggle theme</span>
                        </Button>
                        <Button variant="ghost" size="icon" className="h-9 w-9" asChild>
                            <a href="https://github.com/mangeshraut712" target="_blank" rel="noreferrer">
                                <Github className="h-4 w-4" />
                                <span className="sr-only">GitHub</span>
                            </a>
                        </Button>
                    </nav>
                </div>
            </div>
        </motion.header>
    )
}
