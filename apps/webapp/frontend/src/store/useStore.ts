/**
 * Zustand store dla DogFACS Dataset Generator.
 *
 * Zarządza stanem aplikacji:
 * - Uploaded video file
 * - Processing status
 * - SessionData (frames z anotacjami, keypoints, AU, emocje)
 * - Wybrany frame do edycji
 */

import { create } from 'zustand';
import type {
  DeltaActionUnit,
  FrameAnnotation,
  ProcessingSettings,
  ProcessVideoResponse,
  SessionData,
} from '../types';
import {
  exportCOCO,
  exportSessionCOCO,
  getSession,
  patchAUs,
  patchBreed,
  patchEmotion,
  patchKeypoints,
  processVideo,
  recomputeAUs,
  recomputeEmotion,
} from '../utils/api';

interface AppState {
  // Video data
  video: File | null;
  videoUrl: string | null;
  videoDuration: number;
  videoData: ProcessVideoResponse | null;

  // Session data (anotacje edytowalne)
  sessionData: SessionData | null;
  selectedFrameIdx: number | null;

  // Processing settings
  settings: ProcessingSettings;

  // UI state
  processing: boolean;
  saving: boolean;
  error: string | null;

  // Actions — video
  uploadVideo: (file: File) => void;
  setVideoMetadata: (duration: number) => void;
  updateSettings: (settings: Partial<ProcessingSettings>) => void;
  processVideo: () => Promise<void>;
  setNeutralFrame: (idx: number) => void;
  exportDataset: () => Promise<void>;
  reset: () => void;

  // Actions — session
  loadSession: (sessionId: string) => Promise<void>;
  selectFrame: (frameIdx: number) => void;
  updateFrameKeypoints: (frameIdx: number, keypoints: number[]) => Promise<void>;
  updateFrameAUs: (frameIdx: number, aus: Record<string, DeltaActionUnit>) => Promise<void>;
  updateFrameEmotion: (frameIdx: number, emotion: string) => Promise<void>;
  updateFrameBreed: (frameIdx: number, breed: string) => Promise<void>;
  recomputeFrameAUs: (frameIdx: number) => Promise<void>;
  recomputeFrameEmotion: (frameIdx: number) => Promise<void>;
  exportSession: () => Promise<void>;

  // Helpers
  getSelectedFrame: () => FrameAnnotation | null;
}

const defaultSettings: ProcessingSettings = {
  fps_sample: 1,
  num_peaks: 10,
  min_separation_frames: 30,
  neutral_mode: 'auto',
  manual_neutral_idx: null,
};

/** Aktualizuje klatkę w lokalnej kopii sessionData. */
function updateFrameInSession(
  sessionData: SessionData,
  frameIdx: number,
  updates: Partial<FrameAnnotation>
): SessionData {
  const frames = sessionData.frames.map((f) =>
    f.frame_idx === frameIdx ? { ...f, ...updates } : f
  );
  return { ...sessionData, frames };
}

const useStore = create<AppState>((set, get) => ({
  // Initial state
  video: null,
  videoUrl: null,
  videoDuration: 0,
  videoData: null,
  sessionData: null,
  selectedFrameIdx: null,
  settings: { ...defaultSettings },
  processing: false,
  saving: false,
  error: null,

  // ---------------------------------------------------------------------------
  // Video actions
  // ---------------------------------------------------------------------------

  uploadVideo: (file: File) => {
    const prevUrl = get().videoUrl;
    if (prevUrl) URL.revokeObjectURL(prevUrl);
    const url = URL.createObjectURL(file);
    set({ video: file, videoUrl: url, error: null, videoData: null, sessionData: null });
  },

  setVideoMetadata: (duration: number) => set({ videoDuration: duration }),

  updateSettings: (newSettings: Partial<ProcessingSettings>) => {
    set((state) => ({ settings: { ...state.settings, ...newSettings } }));
  },

  processVideo: async () => {
    const { video, settings } = get();
    if (!video) { set({ error: 'Nie wybrano wideo' }); return; }

    set({ processing: true, error: null });
    try {
      const data = await processVideo(video, {
        fps_sample: settings.fps_sample,
        num_peaks: settings.num_peaks,
        min_separation_frames: settings.min_separation_frames,
        neutral_idx: settings.neutral_mode === 'manual' ? settings.manual_neutral_idx : null,
      });
      set({ videoData: data, processing: false });

      // Po przetworzeniu — załaduj dane sesji z pełnymi anotacjami
      await get().loadSession(data.session_id);
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Błąd przetwarzania wideo',
        processing: false,
      });
    }
  },

  setNeutralFrame: async (idx: number) => {
    const { video, settings } = get();
    if (!video) return;
    set({ processing: true, error: null });
    try {
      const data = await processVideo(video, {
        fps_sample: settings.fps_sample,
        num_peaks: settings.num_peaks,
        neutral_idx: idx,
      });
      set({ videoData: data, processing: false });
      await get().loadSession(data.session_id);
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Błąd przetwarzania',
        processing: false,
      });
    }
  },

  exportDataset: async () => {
    const { videoData } = get();
    if (!videoData) { set({ error: 'Brak danych do eksportu' }); return; }
    try {
      await exportCOCO({
        peak_frames: videoData.peak_frames,
        neutral_frame_idx: videoData.neutral_frame_idx,
        video_filename: videoData.video_filename,
      });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Błąd eksportu' });
    }
  },

  reset: () => {
    const prevUrl = get().videoUrl;
    if (prevUrl) URL.revokeObjectURL(prevUrl);
    set({
      video: null,
      videoUrl: null,
      videoDuration: 0,
      videoData: null,
      sessionData: null,
      selectedFrameIdx: null,
      settings: { ...defaultSettings },
      processing: false,
      saving: false,
      error: null,
    });
  },

  // ---------------------------------------------------------------------------
  // Session actions
  // ---------------------------------------------------------------------------

  loadSession: async (sessionId: string) => {
    try {
      const data = await getSession(sessionId);
      set({ sessionData: data });
      // Domyślnie wybierz pierwszy frame
      if (data.frames.length > 0) {
        set({ selectedFrameIdx: data.frames[0].frame_idx });
      }
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Błąd ładowania sesji' });
    }
  },

  selectFrame: (frameIdx: number) => set({ selectedFrameIdx: frameIdx }),

  updateFrameKeypoints: async (frameIdx: number, keypoints: number[]) => {
    const { sessionData } = get();
    if (!sessionData) return;

    set({ saving: true });
    try {
      await patchKeypoints(sessionData.session_id, frameIdx, keypoints);
      set({
        saving: false,
        sessionData: updateFrameInSession(sessionData, frameIdx, {
          keypoints,
          annotation_status: 'reviewed',
        }),
      });
    } catch (err) {
      set({
        saving: false,
        error: err instanceof Error ? err.message : 'Błąd zapisu keypoints',
      });
    }
  },

  updateFrameAUs: async (frameIdx: number, aus: Record<string, DeltaActionUnit>) => {
    const { sessionData } = get();
    if (!sessionData) return;

    set({ saving: true });
    try {
      await patchAUs(sessionData.session_id, frameIdx, aus);
      set({
        saving: false,
        sessionData: updateFrameInSession(sessionData, frameIdx, {
          aus,
          annotation_status: 'reviewed',
        }),
      });
    } catch (err) {
      set({
        saving: false,
        error: err instanceof Error ? err.message : 'Błąd zapisu AU',
      });
    }
  },

  updateFrameEmotion: async (frameIdx: number, emotion: string) => {
    const { sessionData } = get();
    if (!sessionData) return;

    set({ saving: true });
    try {
      await patchEmotion(sessionData.session_id, frameIdx, emotion);
      set({
        saving: false,
        sessionData: updateFrameInSession(sessionData, frameIdx, {
          emotion,
          emotion_confidence: 1.0,
          emotion_rule_applied: 'manual',
          annotation_status: 'reviewed',
        }),
      });
    } catch (err) {
      set({
        saving: false,
        error: err instanceof Error ? err.message : 'Błąd zapisu emocji',
      });
    }
  },

  updateFrameBreed: async (frameIdx: number, breed: string) => {
    const { sessionData } = get();
    if (!sessionData) return;

    set({ saving: true });
    try {
      await patchBreed(sessionData.session_id, frameIdx, breed);
      set({
        saving: false,
        sessionData: updateFrameInSession(sessionData, frameIdx, {
          breed,
          breed_confidence: 1.0,
          annotation_status: 'reviewed',
        }),
      });
    } catch (err) {
      set({
        saving: false,
        error: err instanceof Error ? err.message : 'Błąd zapisu rasy',
      });
    }
  },

  recomputeFrameAUs: async (frameIdx: number) => {
    const { sessionData } = get();
    if (!sessionData) return;

    set({ saving: true });
    try {
      const aus = await recomputeAUs(sessionData.session_id, frameIdx);
      set({
        saving: false,
        sessionData: updateFrameInSession(sessionData, frameIdx, { aus }),
      });
    } catch (err) {
      set({
        saving: false,
        error: err instanceof Error ? err.message : 'Błąd przeliczania AU',
      });
    }
  },

  recomputeFrameEmotion: async (frameIdx: number) => {
    const { sessionData } = get();
    if (!sessionData) return;

    set({ saving: true });
    try {
      const result = await recomputeEmotion(sessionData.session_id, frameIdx);
      set({
        saving: false,
        sessionData: updateFrameInSession(sessionData, frameIdx, {
          emotion: result.emotion,
          emotion_confidence: result.emotion_confidence,
          emotion_rule_applied: result.emotion_rule_applied,
        }),
      });
    } catch (err) {
      set({
        saving: false,
        error: err instanceof Error ? err.message : 'Błąd przeliczania emocji',
      });
    }
  },

  exportSession: async () => {
    const { sessionData } = get();
    if (!sessionData) { set({ error: 'Brak danych sesji do eksportu' }); return; }
    try {
      await exportSessionCOCO(sessionData.session_id, sessionData.video_filename);
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Błąd eksportu sesji' });
    }
  },

  getSelectedFrame: () => {
    const { sessionData, selectedFrameIdx } = get();
    if (!sessionData || selectedFrameIdx === null) return null;
    return sessionData.frames.find((f) => f.frame_idx === selectedFrameIdx) ?? null;
  },
}));

export default useStore;
