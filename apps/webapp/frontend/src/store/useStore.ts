/**
 * Zustand store dla DogFACS Dataset Generator.
 *
 * Zarządza stanem aplikacji:
 * - Uploaded video file
 * - Processing status
 * - Neutral frame selection
 * - Peak frames data
 * - AU toggles
 */

import { create } from 'zustand';
import type { PeakFrame, ProcessVideoResponse } from '../types';
import { processVideo, exportCOCO } from '../utils/api';

interface AppState {
  // Video data
  video: File | null;
  videoData: ProcessVideoResponse | null;

  // UI state
  processing: boolean;
  error: string | null;

  // Actions
  uploadVideo: (file: File) => void;
  processVideo: () => Promise<void>;
  setNeutralFrame: (idx: number) => void;
  toggleAU: (peakIdx: number, auName: string) => void;
  exportDataset: () => Promise<void>;
  reset: () => void;
}

const useStore = create<AppState>((set, get) => ({
  // Initial state
  video: null,
  videoData: null,
  processing: false,
  error: null,

  // Upload video
  uploadVideo: (file: File) => {
    set({ video: file, error: null });
  },

  // Process video through backend
  processVideo: async () => {
    const { video } = get();
    if (!video) {
      set({ error: 'No video selected' });
      return;
    }

    set({ processing: true, error: null });

    try {
      const data = await processVideo(video);
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
    const { video } = get();
    if (!video) return;

    set({ processing: true, error: null });

    try {
      // Re-process with manual neutral frame
      const data = await processVideo(video, { neutral_idx: idx });
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
    set({
      video: null,
      videoData: null,
      processing: false,
      error: null,
    });
  },
}));

export default useStore;
