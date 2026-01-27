/**
 * TypeScript types dla DogFACS Dataset Generator.
 */

export interface DeltaActionUnit {
  ratio: number;
  delta: number;
  is_active: boolean;
  confidence: number;
}

export interface PeakFrame {
  frame_idx: number;
  image_url: string;
  aus: Record<string, DeltaActionUnit>;
  emotion: string;
  emotion_confidence: number;
  emotion_rule_applied: string;
  tfm_score: number;
}

export interface ProcessVideoResponse {
  session_id: string;
  video_filename: string;
  neutral_frame_idx: number;
  neutral_frame_url: string;
  peak_frames: PeakFrame[];
  total_frames: number;
}

export interface ExportCOCORequest {
  peak_frames: PeakFrame[];
  neutral_frame_idx: number;
  video_filename: string;
}

/**
 * Налаштування обробки відео.
 */
export interface ProcessingSettings {
  fps_sample: number;          // 1-30 fps
  num_peaks: number;           // 5-20 peak frames
  min_separation_frames: number;
  neutral_mode: 'auto' | 'manual';
  manual_neutral_idx: number | null;
}

/**
 * Параметри API для обробки відео.
 */
export interface ProcessVideoOptions {
  fps_sample?: number;
  num_peaks?: number;
  neutral_idx?: number | null;
  min_separation_frames?: number;
}

export const EMOTION_EMOJI: Record<string, string> = {
  happy: '😊',
  sad: '😢',
  angry: '😠',
  fearful: '😨',
  relaxed: '😌',
  neutral: '😐',
};

export const AU_NAMES: Record<string, string> = {
  AU101: 'Inner Brow Raiser',
  AU102: 'Outer Brow Raiser',
  AU12: 'Lip Corner Puller',
  AU115: 'Upper Eyelid Raiser',
  AU116: 'Lower Eyelid Raiser',
  AU117: 'Eye Closure',
  AU121: 'Eye Widener',
  EAD102: 'Ears Forward',
  EAD103: 'Ears Flattener',
  AD19: 'Tongue Show',
  AD37: 'Nose Lick',
  AU26: 'Jaw Drop',
};
