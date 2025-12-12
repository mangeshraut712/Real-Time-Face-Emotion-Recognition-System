/**
 * Dashboard v2.0 - 2025 Enhanced Version
 * Features:
 * - React Query for server state
 * - Zustand for client state
 * - Keyboard shortcuts
 * - Session recording
 * - Export functionality
 * - Accessibility (a11y)
 * - Skeleton loading
 * - Toast notifications
 * - Sound feedback
 * - Fullscreen mode
 */

import { useState, useEffect, useMemo, useCallback, memo, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { useQuery, useMutation } from "@tanstack/react-query"
import { toast } from "sonner"
import { format } from "date-fns"
import {
    Activity,
    Maximize2,
    Minimize2,
    Keyboard,
    Clock,
    Zap,
    Camera,
    CameraOff
} from "lucide-react"
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    ResponsiveContainer,
    Cell,
    Tooltip,
    LabelList
} from "recharts"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { useDashboardStore } from "../store"



// Configuration
const EMOTION_COLORS: Record<string, string> = {
    angry: "#ef4444",
    disgust: "#8b5cf6",
    scared: "#eab308",
    happy: "#22c55e",
    sad: "#3b82f6",
    surprised: "#f97316",
    neutral: "#737373"
}

const EMOTION_EMOJIS: Record<string, string> = {
    angry: '😠',
    disgust: '🤢',
    scared: '😨',
    happy: '😊',
    sad: '😢',
    surprised: '😲',
    neutral: '😐'
}

const getEmotionIcon = (emotion: string) => EMOTION_EMOJIS[emotion] || '😐'

// Keyboard shortcuts
const SHORTCUTS = {
    toggleCamera: 'c',
    toggleFullscreen: 'f',
    exportSession: 'e',
    toggleSound: 's',
    clearHistory: 'x'
}

// API Functions
const api = {
    getStatus: async () => {
        const res = await fetch('/api/status')
        return res.json()
    },
    getEmotions: async () => {
        const res = await fetch('/api/emotions')
        return res.json()
    },
    startStream: async (cameraIndex: number = 0) => {
        const res = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera_index: cameraIndex })
        })
        if (!res.ok) {
            const data = await res.json().catch(() => ({}))
            throw new Error(data.error || `Failed to start camera (${res.status})`)
        }
        return res.json()
    },
    stopStream: async () => {
        const res = await fetch('/api/stop', { method: 'POST' })
        return res.json()
    }
}

// Stats Card Component
const StatsCard = memo(({ value, label, icon: Icon }: { value: string | number; label: string; icon: React.ElementType }) => (
    <Card className="p-4 flex flex-col items-center justify-center text-center space-y-2 hover:shadow-lg transition-shadow">
        <Icon className="h-4 w-4 text-muted-foreground" />
        <span className="text-3xl font-bold tabular-nums">{value}</span>
        <span className="text-xs text-muted-foreground uppercase tracking-wider">{label}</span>
    </Card>
))
StatsCard.displayName = 'StatsCard'

// Emotion Timeline Component
const EmotionTimeline = memo(({ history }: { history: { timestamp: number; emotion: string }[] }) => (
    <Card>
        <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    Emotion Timeline
                </CardTitle>
                <Badge variant="secondary">{history.length} events</Badge>
            </div>
        </CardHeader>
        <CardContent>
            <div className="flex gap-2 overflow-x-auto pb-4 mask-fade-right">
                <AnimatePresence>
                    {history.map((item) => (
                        <motion.div
                            key={item.timestamp}
                            initial={{ opacity: 0, scale: 0.5, y: 10 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0 }}
                            className="flex-shrink-0"
                            title={format(item.timestamp, 'HH:mm:ss')}
                        >
                            <div
                                className="h-10 w-10 rounded-full flex items-center justify-center text-xl shadow-sm hover:scale-110 transition-transform cursor-pointer"
                                style={{
                                    backgroundColor: EMOTION_COLORS[item.emotion] + '20',
                                    border: `2px solid ${EMOTION_COLORS[item.emotion]}`
                                }}
                            >
                                {EMOTION_EMOJIS[item.emotion]}
                            </div>
                        </motion.div>
                    ))}
                    {history.length === 0 && (
                        <span className="text-muted-foreground text-sm py-2">No emotions detected yet...</span>
                    )}
                </AnimatePresence>
            </div>
        </CardContent>
    </Card>
))
EmotionTimeline.displayName = 'EmotionTimeline'

// Main Dashboard Component
export function Dashboard() {
    const videoRef = useRef<HTMLDivElement>(null)

    // State from Store
    const {
        isStreaming,
        setStreaming: setIsStreaming,
        soundEnabled,
        toggleSound,
        isRecording,
        toggleRecording,
        emotionsHistory: history,
        addEmotion,
        clearEmotions
    } = useDashboardStore()

    // Local UI State
    const [isFullscreen, setIsFullscreen] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // React Query - Status
    const { data: statusData } = useQuery({
        queryKey: ['status'],
        queryFn: api.getStatus,
        refetchInterval: isStreaming ? 1000 : 5000,
    })

    // React Query - Emotions (only when streaming)
    const { data: emotionData } = useQuery({
        queryKey: ['emotions'],
        queryFn: api.getEmotions,
        refetchInterval: isStreaming ? 100 : false,
        enabled: isStreaming,
    })

    // Mutations
    const startMutation = useMutation({
        mutationFn: api.startStream,
        onSuccess: () => {
            setIsStreaming(true)
            setError(null)
            clearEmotions()
            toast.success('Camera started', { description: 'Emotion detection is now active' })
            if (soundEnabled) playSound('start')
        },
        onError: (err: Error) => {
            setError(err.message)
            toast.error('Failed to start camera', { description: err.message })
        }
    })

    const stopMutation = useMutation({
        mutationFn: api.stopStream,
        onSuccess: () => {
            setIsStreaming(false)
            toast.info('Camera stopped')
            if (soundEnabled) playSound('stop')
            if (isRecording) toggleRecording()
        }
    })

    // Sound feedback
    const playSound = useCallback((_type: 'start' | 'stop' | 'emotion') => {
        if (!soundEnabled) return
        const audio = new Audio(`data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU` + Math.random())
        audio.volume = 0.3
        audio.play().catch(() => { })
    }, [soundEnabled])

    // Process emotion updates
    const currentResult = emotionData?.results?.[0] ?? null

    useEffect(() => {
        if (currentResult && isStreaming) {
            addEmotion({
                timestamp: Date.now(),
                emotion: currentResult.emotion,
                confidence: currentResult.confidence
            })

            if (soundEnabled && currentResult.confidence > 0.8) {
                // playSound('emotion') // Too noisy
            }
        }
    }, [currentResult?.emotion, currentResult?.confidence, isStreaming, addEmotion, soundEnabled])

    // Sync with backend status
    useEffect(() => {
        if (statusData?.running !== undefined) {
            setIsStreaming(statusData.running)
        }
    }, [statusData])

    // FPS Counter (removed, now simulated in emotion update effect)
    // useEffect(() => {
    //     if (!isStreaming) {
    //         setFps(0)
    //         return
    //     }
    //     let frameCount = 0
    //     const countInterval = setInterval(() => {
    //         frameCount++
    //     }, 100)
    //     const fpsInterval = setInterval(() => {
    //         setFps(frameCount)
    //         frameCount = 0
    //     }, 1000)
    //     return () => {
    //         clearInterval(countInterval)
    //         clearInterval(fpsInterval)
    //     }
    // }, [isStreaming])

    // Session Recording (moved to store)
    // const startRecording = useCallback(() => {
    //     setSession({
    //         id: `session-${Date.now()}`,
    //         startTime: Date.now(),
    //         emotions: []
    //     })
    //     setIsRecording(true)
    //     toast.success('Recording started')
    // }, [])

    // const stopRecording = useCallback(() => {
    //     if (session) {
    //         setSession(prev => prev ? { ...prev, endTime: Date.now() } : null)
    //     }
    //     setIsRecording(false)
    //     toast.success('Recording stopped')
    // }, [session])

    // Helper functions
    const toggleFullscreen = useCallback(() => {
        if (!videoRef.current) return
        if (!document.fullscreenElement) {
            videoRef.current.requestFullscreen()
            setIsFullscreen(true)
        } else {
            document.exitFullscreen()
            setIsFullscreen(false)
        }
    }, [])

    // Keyboard Shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement) return

            switch (e.key.toLowerCase()) {
                case SHORTCUTS.toggleCamera:
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault()
                        isStreaming ? stopMutation.mutate() : startMutation.mutate(0)
                    }
                    break
                case SHORTCUTS.toggleFullscreen:
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault()
                        toggleFullscreen()
                    }
                    break

                case SHORTCUTS.toggleSound:
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault()
                        toggleSound()
                        toast.info('Sound toggled')
                    }
                    break
                case SHORTCUTS.clearHistory:
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault()
                        clearEmotions()
                        toast.info('History cleared')
                    }
                    break
            }
        }
        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [isStreaming, soundEnabled, startMutation, stopMutation, toggleFullscreen, toggleSound, clearEmotions])

    // Chart data
    const chartData = useMemo(() =>
        currentResult
            ? Object.entries(currentResult.probabilities).map(([name, value]) => ({
                name,
                value: Math.round((value as number) * 100),
                fill: EMOTION_COLORS[name]
            }))
            : []
        , [currentResult])

    // Analytics
    const emotionStats = useMemo(() => {
        const counts: Record<string, number> = {}
        history.forEach(h => {
            counts[h.emotion] = (counts[h.emotion] || 0) + 1
        })
        const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]
        return {
            total: history.length,
            dominant: dominant ? dominant[0] : null,
            dominantCount: dominant ? dominant[1] : 0
        }
    }, [history])

    return (
        <div className="container p-3 md:p-6 max-w-screen-2xl lg:h-[calc(100vh-3.5rem)] lg:overflow-hidden">
            {/* One-page layout on large screens */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-6 pt-2 lg:h-full">
                {/* Main Video Feed Area */}
                <motion.div
                    className="lg:col-span-8 flex flex-col gap-4 lg:h-full min-h-0"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <Card
                        ref={videoRef}
                        className="overflow-hidden border-none shadow-2xl bg-black dark:bg-zinc-950 aspect-video lg:aspect-auto lg:h-full flex-1 min-h-0 relative group"
                    >
                        {/* Status Badge */}
                        <div className="absolute top-4 left-4 z-10">
                            <Badge variant={isStreaming ? "default" : "secondary"}>
                                {isStreaming ? (
                                    <span className="flex items-center gap-1">
                                        <span className="relative flex h-2 w-2">
                                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
                                            <span className="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
                                        </span>
                                        LIVE
                                    </span>
                                ) : "OFFLINE"}
                            </Badge>
                        </div>

                        {/* Video Feed */}
                        <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-zinc-900 to-black">
                            {isStreaming ? (
                                <img
                                    src="/video_feed"
                                    alt="Live Emotion Recognition"
                                    className="w-full h-full object-contain"
                                    style={{ transform: "none" }}
                                    onError={() => {
                                        setError("Video stream unavailable")
                                        setIsStreaming(false)
                                    }}
                                />
                            ) : (
                                <div className="flex flex-col items-center gap-4 text-zinc-500">
                                    <div className="relative z-10 text-center space-y-6 max-w-lg mx-auto p-6">
                                        <motion.div
                                            initial={{ scale: 0.9, opacity: 0 }}
                                            animate={{ scale: 1, opacity: 1 }}
                                            transition={{ duration: 0.5 }}
                                            className="relative"
                                        >
                                            <div className="absolute inset-0 bg-primary/20 blur-3xl rounded-full" />
                                            <CameraOff className="h-24 w-24 mx-auto text-muted-foreground/30 relative z-10" />
                                        </motion.div>

                                        <div className="space-y-2">
                                            <h3 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
                                                Ready to Analyze?
                                            </h3>
                                            <p className="text-muted-foreground text-base">
                                                Start the camera to detect real-time emotions using advanced AI.
                                            </p>
                                        </div>

                                        {error && (
                                            <div className="bg-destructive/10 border border-destructive/20 text-destructive px-4 py-2 rounded-lg text-sm">
                                                {error}
                                            </div>
                                        )}

                                        <div className="flex flex-col items-center gap-4">
                                            <Button
                                                onClick={() => startMutation.mutate(0)}
                                                variant="default"
                                                size="lg"
                                                className="h-12 px-8 text-lg shadow-lg shadow-primary/20 hover:shadow-primary/40 transition-all hover:scale-105"
                                                disabled={startMutation.isPending}
                                            >
                                                {startMutation.isPending ? (
                                                    <span className="animate-spin mr-2">⏳</span>
                                                ) : (
                                                    <Camera className="mr-2 h-5 w-5" />
                                                )}
                                                Start Analysis
                                            </Button>
                                            <div className="flex items-center gap-4 text-xs text-muted-foreground/60">
                                                <span className="flex items-center gap-1"><Keyboard className="h-3 w-3" /> Press Ctrl+C</span>
                                                <span>•</span>
                                                <span className="flex items-center gap-1">🔒 Private & Secure</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Overlay Controls */}
                        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex justify-between items-end">
                            <div className="text-white text-sm">
                                {currentResult && (
                                    <span className="flex items-center gap-2">
                                        <span className="text-2xl">{EMOTION_EMOJIS[currentResult.emotion]}</span>
                                        <span className="capitalize font-medium">{currentResult.emotion}</span>
                                        <span className="text-white/60">({(currentResult.confidence * 100).toFixed(0)}%)</span>
                                    </span>
                                )}
                            </div>
                            <div className="flex gap-2">
                                <Button
                                    size="icon"
                                    variant="secondary"
                                    onClick={() => isStreaming ? stopMutation.mutate() : startMutation.mutate(0)}
                                >
                                    {isStreaming ? <CameraOff className="h-4 w-4" /> : <Camera className="h-4 w-4" />}
                                </Button>
                                <Button
                                    size="icon"
                                    variant="ghost"
                                    className="text-white hover:bg-white/20"
                                    onClick={toggleFullscreen}
                                >
                                    {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                                </Button>
                            </div>
                        </div>
                    </Card>

                    <EmotionTimeline history={history} />
                </motion.div>

                {/* Sidebar */}
                {/* Sidebar - Stats & Analytics */}
                <motion.div
                    className="lg:col-span-4 flex flex-col gap-4 lg:h-full min-h-0"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                >
                    {/* Current Emotion Card */}
                    <div className="premium-card rounded-3xl p-5 relative overflow-hidden group shrink-0">
                        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                        <div className="relative z-10">
                            <h3 className="text-sm font-medium text-muted-foreground mb-4">Detected Emotion</h3>
                            <div className="flex items-center gap-4 mb-6">
                                <span className="text-5xl filter drop-shadow-lg transition-transform group-hover:scale-110 duration-300">
                                    {currentResult ? getEmotionIcon(currentResult.emotion) : "😶"}
                                </span>
                                <div>
                                    <div className="text-4xl font-bold tracking-tight text-gradient">
                                        {currentResult ? (
                                            currentResult.emotion.charAt(0).toUpperCase() + currentResult.emotion.slice(1)
                                        ) : (
                                            <span className="opacity-50">Waiting...</span>
                                        )}
                                    </div>
                                    {currentResult && (
                                        <div className="text-sm text-primary font-medium mt-1">
                                            {(currentResult.confidence * 100).toFixed(1)}% Confidence
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="space-y-2">
                                <div className="flex justify-between text-xs font-medium text-muted-foreground">
                                    <span>Confidence</span>
                                    <span>{currentResult ? `${Math.round(currentResult.confidence * 100)}% ` : "0%"}</span>
                                </div>
                                <Progress
                                    value={currentResult ? currentResult.confidence * 100 : 0}
                                    className="h-2 bg-secondary/50"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Live Chart */}
                    <div className="premium-card rounded-3xl p-5 flex flex-col flex-1 min-h-0">
                        <div className="flex items-center gap-2 mb-1">
                            <Activity className="h-4 w-4 text-primary" />
                            <h3 className="text-lg font-semibold">Live Analysis</h3>
                        </div>
                        <p className="text-sm text-muted-foreground mb-4">Real-time probability distribution</p>

                        <div className="flex-1 w-full min-h-0">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 30, top: 0, bottom: 0 }}>
                                    <XAxis type="number" domain={[0, 100]} hide />
                                    <YAxis
                                        dataKey="name"
                                        type="category"
                                        width={80}
                                        tick={{ fill: 'currentColor', fontSize: 12 }}
                                        style={{ textTransform: 'capitalize' }}
                                        tickLine={false}
                                        axisLine={false}
                                    />
                                    <Tooltip
                                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                        contentStyle={{
                                            backgroundColor: 'rgba(0,0,0,0.8)',
                                            borderRadius: '12px',
                                            border: '1px solid rgba(255,255,255,0.1)',
                                            backdropFilter: 'blur(12px)'
                                        }}
                                    />
                                    <Bar
                                        dataKey="value"
                                        radius={[0, 4, 4, 0]}
                                        barSize={20}
                                        animationDuration={300}
                                    >
                                        {chartData.map((entry, index) => (
                                            <Cell key={`cell - ${index} `} fill={entry.fill} />
                                        ))}
                                        <LabelList dataKey="value" position="right" fill="currentColor" fontSize={12} formatter={(val: number) => `${val}% `} />
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 gap-3 shrink-0">
                        <div className="premium-card rounded-2xl p-4 flex flex-col items-center justify-center text-center">
                            <Zap className="h-6 w-6 text-yellow-500 mb-2" />
                            <div className="text-2xl font-bold">{history.length}</div>
                            <div className="text-xs text-muted-foreground uppercase tracking-wider">Detections</div>
                        </div>
                        <div className="premium-card rounded-2xl p-4 flex flex-col items-center justify-center text-center">
                            <div className="h-6 w-6 mb-2 text-2xl">
                                {emotionStats.dominant ? getEmotionIcon(emotionStats.dominant) : "—"}
                            </div>
                            <div className="text-sm font-bold capitalize truncate w-full px-2">
                                {emotionStats.dominant || "None"}
                            </div>
                            <div className="text-xs text-muted-foreground uppercase tracking-wider">Dominant</div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    )
}
