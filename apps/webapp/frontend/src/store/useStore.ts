/**
 * Zustand store dla DogFACS Dataset Generator.
 *
 * Zarządza stanem aplikacji:
 * - Uploaded video file
 * - Processing status
 * - Neutral frame selection
 * - Peak frames data
 * - AU toggles
 * - Processing settings (fps, mode, peaks)
 */

import { create } from 'zustand';
import type { ProcessVideoResponse, ProcessingSettings } from '../types';
import { processVideo, exportCOCO } from '../utils/api';

interface AppState {
  // Video data
  video: File | null;
  videoUrl: string | null;
  videoDuration: number;
  videoData: ProcessVideoResponse | null;

  // Processing settings
  settings: ProcessingSettings;

  // UI state
  processing: boolean;
  error: string | null;

  // Actions
  uploadVideo: (file: File) => void;
  setVideoMetadata: (duration: number) => void;
  updateSettings: (settings: Partial<ProcessingSettings>) => void;
  processVideo: () => Promise<void>;
  setNeutralFrame: (idx: number) => void;
  toggleAU: (peakIdx: number, auName: string) => void;
  exportDataset: () => Promise<void>;
  reset: () => void;
}

const defaultSettings: ProcessingSettings = {
  fps_sample: 1,
  num_peaks: 10,
  min_separation_frames: 30,
  neutral_mode: 'auto',
  manual_neutral_idx: null,
};

const useStore = create<AppState>((set, get) => ({
  // Initial state
  video: null,
  videoUrl: null,
  videoDuration: 0,
  videoData: null,
  settings: { ...defaultSettings },
  processing: false,
  error: null,

  // Upload video
  uploadVideo: (file: File) => {
    // Revoke previous URL if exists
    const prevUrl = get().videoUrl;
    if (prevUrl) {
      URL.revokeObjectURL(prevUrl);
    }
    const url = URL.createObjectURL(file);
    set({ video: file, videoUrl: url, error: null, videoData: null });
  },

  // Set video metadata after loading
  setVideoMetadata: (duration: number) => {
    set({ videoDuration: duration });
  },

  // Update processing settings
  updateSettings: (newSettings: Partial<ProcessingSettings>) => {
    const { settings } = get();
    set({ settings: { ...settings, ...newSettings } });
  },

  // Process video through backend
  processVideo: async () => {
    const { video, settings } = get();
    if (!video) {
      set({ error: 'No video selected' });
      return;
    }

    set({ processing: true, error: null });

    try {
      const options = {
        fps_sample: settings.fps_sample,
        num_peaks: settings.num_peaks,
        min_separation_frames: settings.min_separation_frames,
        neutral_idx: settings.neutral_mode === 'manual' ? settings.manual_neutral_idx : null,
      };
      const data = await processVideo(video, options);
      set({ videoData: data, processing: false });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Unknown error',
        processing: false,
      });
    }
  },

  // Set neutral frame (manual override)
  setNeutralFrame: async (idx: number) => {
    const { video, settings } = get();
    if (!video) return;

    set({ processing: true, error: null });

    try {
      // Re-process with manual neutral frame
      const options = {
        fps_sample: settings.fps_sample,
        num_peaks: settings.num_peaks,
        neutral_idx: idx,
      };
      const data = await processVideo(video, options);
      set({ videoData: data, processing: false });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Unknown error',
        processing: false,
      });
    }
  },

  // Toggle AU activation (manual override)
  toggleAU: (peakIdx: number, auName: string) => {
    const { videoData } = get();
    if (!videoData) return;

    // Clone videoData
    const newVideoData = { ...videoData };
    const updatedPeakFrames = [...newVideoData.peak_frames];

    // Toggle AU
    const peakFrame = { ...updatedPeakFrames[peakIdx] };
    const aus = { ...peakFrame.aus };
    const au = { ...aus[auName] };

    au.is_active = !au.is_active;
    aus[auName] = au;
    peakFrame.aus = aus;

    // Recompute emotion based on new AU state
    // TODO: Call backend API to recompute emotion
    // For now, just update the AU

    updatedPeakFrames[peakIdx] = peakFrame;
    newVideoData.peak_frames = updatedPeakFrames;

    set({ videoData: newVideoData });
  },

  // Export dataset to COCO format
  exportDataset: async () => {
    const { videoData } = get();
    if (!videoData) {
      set({ error: 'No data to export' });
      return;
    }

    try {
      await exportCOCO({
        peak_frames: videoData.peak_frames,
        neutral_frame_idx: videoData.neutral_frame_idx,
        video_filename: videoData.video_filename,
      });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Export failed',
      });
    }
  },

  // Reset state
  reset: () => {
    const prevUrl = get().videoUrl;
    if (prevUrl) {
      URL.revokeObjectURL(prevUrl);
    }
    set({
      video: null,
      videoUrl: null,
      videoDuration: 0,
      videoData: null,
      settings: { ...defaultSettings },
      processing: false,
      error: null,
    });
  },
}));

export default useStore;
