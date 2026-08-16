/**
 * API utilities dla komunikacji z FastAPI backend.
 */

import axios from 'axios';
import { saveAs } from 'file-saver';
import type {
  AUVerdict,
  DeltaActionUnit,
  ExportCOCORequest,
  ProcessVideoOptions,
  ProcessVideoResponse,
  SessionData,
} from '../types';

const API_BASE = '/api';

// =============================================================================
// Video Processing
// =============================================================================

/** Przetwarza wideo przez backend pipeline. */
export async function processVideo(
  file: File,
  options?: ProcessVideoOptions
): Promise<ProcessVideoResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams();
  if (options?.fps_sample)          params.append('fps_sample', String(options.fps_sample));
  if (options?.num_peaks)           params.append('num_peaks', String(options.num_peaks));
  if (options?.min_separation_frames) params.append('min_separation_frames', String(options.min_separation_frames));
  if (options?.neutral_idx != null) params.append('neutral_idx', String(options.neutral_idx));

  const response = await axios.post<ProcessVideoResponse>(
    `${API_BASE}/process_video?${params}`,
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } }
  );
  return response.data;
}

/** Eksportuje dataset do formatu COCO (stary endpoint). */
export async function exportCOCO(request: ExportCOCORequest): Promise<void> {
  const response = await axios.post(`${API_BASE}/export_coco`, request, {
    responseType: 'blob',
  });
  saveAs(response.data, `dogfacs_dataset_${request.video_filename}.json`);
}

/** Health check. */
export async function healthCheck(): Promise<{ status: string; pipeline_loaded: boolean }> {
  const response = await axios.get(`${API_BASE}/health`);
  return response.data;
}

// =============================================================================
// Sessions API (Sprint 9 endpoints)
// =============================================================================

/** Pobiera pełne dane sesji z wszystkimi anotacjami. */
export async function getSession(sessionId: string): Promise<SessionData> {
  const response = await axios.get<SessionData>(`${API_BASE}/sessions/${sessionId}`);
  return response.data;
}

/** Odpowiedź importu zbioru COCO do sesji weryfikacji. */
export interface ImportCocoResponse {
  session_id: string;
  pairs: number;
  frames: number;
  source: string;
  verified: number;
  /** true = podjęto istniejącą sesję, false = założono nową */
  resumed: boolean;
}

/** Zbiór gotowy do weryfikacji, znaleziony przez backend w katalogu danych. */
export interface AvailableDataset {
  path: string;
  name: string;
  pairs: number;
  verified: number;
  session_id: string;
}

/**
 * Wylicza zbiory gotowe do weryfikacji.
 *
 * Anotator nie wpisuje ścieżek — narzędzie samo pokazuje, co jest do zrobienia.
 */
export async function listDatasets(): Promise<{ root: string; datasets: AvailableDataset[] }> {
  const response = await axios.get<{ root: string; datasets: AvailableDataset[] }>(
    `${API_BASE}/sessions/datasets/available`
  );
  return response.data;
}

/**
 * Tworzy sesję weryfikacji ze zbioru COCO po kuracji.
 *
 * @param path Ścieżka do pliku po `curate_for_review.py` (widziana przez backend)
 * @param limit Najwyżej tyle par; pominięty = wszystkie
 */
export async function importCoco(path: string, limit?: number): Promise<ImportCocoResponse> {
  const response = await axios.post<ImportCocoResponse>(
    `${API_BASE}/sessions/import_coco`,
    { path, limit: limit ?? null }
  );
  return response.data;
}

/** Komplet ocen człowieka zapisywany jednym żądaniem. */
export interface ReviewPayload {
  verdicts: Record<string, AUVerdict>;
  usable: boolean;
  keypoints_ok: boolean | null;
  breed: string | null;
  emotion: string | null;
}

/**
 * Zapisuje CAŁĄ weryfikację pary jednym żądaniem: AU, keypoints, rasę i emocję.
 *
 * Osobne zapisy znaczyłyby, że zbiór trzeba przejść tyle razy, ile jest pól —
 * a przy 518 parach jedno przejście jest warunkiem wykonalności.
 */
export async function patchReview(
  sessionId: string,
  frameIdx: number,
  trackId: number,
  payload: ReviewPayload
): Promise<void> {
  await axios.patch(
    `${API_BASE}/sessions/${sessionId}/frames/${frameIdx}/review?track_id=${trackId}`,
    { ...payload, mark_verified: true }
  );
}

/**
 * Zapisuje werdykty człowieka o AU danej klatki.
 *
 * Werdykty DOPISUJĄ się do już zapisanych — anotator ocenia AU po kolei
 * i nie powinien tracić wcześniejszych decyzji.
 */
export async function patchAUVerdicts(
  sessionId: string,
  frameIdx: number,
  trackId: number,
  verdicts: Record<string, AUVerdict>,
  markVerified: boolean,
  usable = true
): Promise<void> {
  await axios.patch(
    `${API_BASE}/sessions/${sessionId}/frames/${frameIdx}/au_verdicts?track_id=${trackId}`,
    { verdicts, mark_verified: markVerified, usable }
  );
}

/** Aktualizuje keypoints klatki. */
export async function patchKeypoints(
  sessionId: string,
  frameIdx: number,
  keypoints: number[]
): Promise<void> {
  await axios.patch(
    `${API_BASE}/sessions/${sessionId}/frames/${frameIdx}/keypoints`,
    { keypoints }
  );
}

/** Aktualizuje Action Units klatki. */
export async function patchAUs(
  sessionId: string,
  frameIdx: number,
  aus: Record<string, DeltaActionUnit>
): Promise<void> {
  await axios.patch(
    `${API_BASE}/sessions/${sessionId}/frames/${frameIdx}/aus`,
    { aus }
  );
}

/** Aktualizuje emocję klatki. */
export async function patchEmotion(
  sessionId: string,
  frameIdx: number,
  emotion: string,
  emotionConfidence = 1.0,
  emotionRuleApplied: string | null = 'manual'
): Promise<void> {
  await axios.patch(
    `${API_BASE}/sessions/${sessionId}/frames/${frameIdx}/emotion`,
    { emotion, emotion_confidence: emotionConfidence, emotion_rule_applied: emotionRuleApplied }
  );
}

/** Aktualizuje rasę psa na klatce. */
export async function patchBreed(
  sessionId: string,
  frameIdx: number,
  breed: string,
  breedConfidence = 1.0
): Promise<void> {
  await axios.patch(
    `${API_BASE}/sessions/${sessionId}/frames/${frameIdx}/breed`,
    { breed, breed_confidence: breedConfidence }
  );
}

/** Przelicza AU z keypoints klatki i klatki neutralnej. */
export async function recomputeAUs(
  sessionId: string,
  frameIdx: number
): Promise<Record<string, DeltaActionUnit>> {
  const response = await axios.post<{ ok: boolean; aus: Record<string, DeltaActionUnit> }>(
    `${API_BASE}/sessions/${sessionId}/frames/${frameIdx}/recompute_aus`
  );
  return response.data.aus;
}

/** Przelicza emocję z AU klatki. */
export async function recomputeEmotion(
  sessionId: string,
  frameIdx: number
): Promise<{ emotion: string; emotion_confidence: number; emotion_rule_applied: string }> {
  const response = await axios.post<{
    ok: boolean;
    emotion: string;
    emotion_confidence: number;
    emotion_rule_applied: string;
  }>(`${API_BASE}/sessions/${sessionId}/frames/${frameIdx}/recompute_emotion`);
  return response.data;
}

/** Eksportuje sesję do formatu COCO JSON. */
export async function exportSessionCOCO(sessionId: string, videoFilename: string): Promise<void> {
  const response = await axios.post(
    `${API_BASE}/sessions/${sessionId}/export_coco`,
    null,
    { responseType: 'blob' }
  );
  saveAs(response.data, `dogfacs_${sessionId}_${videoFilename}.json`);
}
