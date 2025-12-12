import { create } from 'zustand'

interface Emotion {
    timestamp: number
    emotion: string
    confidence: number
}

interface DashboardState {
    // Media & Hardware
    isStreaming: boolean
    soundEnabled: boolean

    // Session Recording
    isRecording: boolean
    recordingStartTime: number | null
    emotionsHistory: Emotion[]

    // Actions
    setStreaming: (streaming: boolean) => void
    toggleSound: () => void
    toggleRecording: () => void
    addEmotion: (emotion: Emotion) => void
    clearEmotions: () => void
}

export const useDashboardStore = create<DashboardState>((set) => ({
    isStreaming: false,
    soundEnabled: true,
    isRecording: false,
    recordingStartTime: null,
    emotionsHistory: [],

    setStreaming: (streaming) => set({ isStreaming: streaming }),
    toggleSound: () => set((state) => ({ soundEnabled: !state.soundEnabled })),
    toggleRecording: () => set((state) => ({
        isRecording: !state.isRecording,
        recordingStartTime: !state.isRecording ? Date.now() : null
    })),
    addEmotion: (emotion) => set((state) => ({ emotionsHistory: [...state.emotionsHistory, emotion] })),
    clearEmotions: () => set({ emotionsHistory: [] })
}))
