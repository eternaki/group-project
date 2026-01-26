/**
 * API utilities dla komunikacji z FastAPI backend.
 */

import axios from 'axios';
import { saveAs } from 'file-saver';
import type { ProcessVideoResponse, ExportCOCORequest } from '../types';

const API_BASE_URL = '/api';

/**
 * Przetwarza wideo przez backend.
 */
export async function processVideo(
  file: File,
  options?: {
    num_peaks?: number;
    neutral_idx?: number | null;
    min_separation_frames?: number;
  }
): Promise<ProcessVideoResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams();
  if (options?.num_peaks) {
    params.append('num_peaks', options.num_peaks.toString());
  }
  if (options?.neutral_idx !== undefined && options.neutral_idx !== null) {
    params.append('neutral_idx', options.neutral_idx.toString());
  }
  if (options?.min_separation_frames) {
    params.append('min_separation_frames', options.min_separation_frames.toString());
  }

  const response = await axios.post<ProcessVideoResponse>(
    `${API_BASE_URL}/process_video?${params.toString()}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
}

/**
 * Eksportuje dataset do formatu COCO.
 */
export async function exportCOCO(request: ExportCOCORequest): Promise<void> {
  const response = await axios.post(
    `${API_BASE_URL}/export_coco`,
    request,
    {
      responseType: 'blob',
    }
  );

  // Download file
  const filename = `dogfacs_dataset_${request.video_filename}.json`;
  saveAs(response.data, filename);
}

/**
 * Health check.
 */
export async function healthCheck(): Promise<{ status: string; pipeline_loaded: boolean }> {
  const response = await axios.get(`${API_BASE_URL}/health`);
  return response.data;
}
