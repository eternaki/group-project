/**
 * VideoUpload component - Upload video file with settings.
 *
 * Features:
 * - Video upload and preview
 * - FPS slider (1-30 fps)
 * - Auto/Manual neutral frame mode
 * - Frame selector for manual mode
 * - Number of peaks slider
 */

import { useRef, useEffect, useCallback } from 'react';
import useStore from '../store/useStore';

export default function VideoUpload() {
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const {
    video,
    videoUrl,
    videoDuration,
    settings,
    uploadVideo,
    setVideoMetadata,
    updateSettings,
    processVideo,
    processing,
    reset,
  } = useStore();

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadVideo(file);
    }
  };

  // Handle video metadata loaded
  const handleVideoLoaded = () => {
    if (videoRef.current) {
      setVideoMetadata(videoRef.current.duration);
    }
  };

  // Update canvas preview for manual frame selection
  const updateFramePreview = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || settings.neutral_mode !== 'manual') return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth || 320;
    canvas.height = video.videoHeight || 240;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }, [settings.neutral_mode]);

  // Seek video to frame when manual_neutral_idx changes
  useEffect(() => {
    if (videoRef.current && settings.neutral_mode === 'manual' && settings.manual_neutral_idx !== null) {
      const time = settings.manual_neutral_idx / settings.fps_sample;
      videoRef.current.currentTime = time;
    }
  }, [settings.manual_neutral_idx, settings.fps_sample, settings.neutral_mode]);

  // Update preview when video seeks
  const handleSeeked = () => {
    updateFramePreview();
  };

  // Handle process button
  const handleProcess = async () => {
    await processVideo();
  };

  // Handle reset
  const handleReset = () => {
    reset();
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  // Calculate estimated frames based on duration and fps
  const estimatedFrames = Math.floor(videoDuration * settings.fps_sample);
  const maxNeutralIdx = Math.max(0, estimatedFrames - 1);

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4">Upload Video</h2>

      <div className="space-y-6">
        {/* File input */}
        <div>
          <label
            htmlFor="video-upload"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Select dog video (max 60s recommended)
          </label>
          <input
            ref={inputRef}
            id="video-upload"
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-md file:border-0
              file:text-sm file:font-semibold
              file:bg-primary-50 file:text-primary-700
              hover:file:bg-primary-100"
          />
        </div>

        {/* Video Preview */}
        {videoUrl && (
          <div className="border rounded-lg p-4 bg-gray-50">
            <h3 className="font-semibold text-gray-700 mb-3">Video Preview</h3>
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              onLoadedMetadata={handleVideoLoaded}
              onSeeked={handleSeeked}
              className="w-full max-w-xl mx-auto rounded-lg"
            />
            {videoDuration > 0 && (
              <p className="text-sm text-gray-600 mt-2 text-center">
                Duration: {videoDuration.toFixed(1)}s | Estimated frames: ~{estimatedFrames}
              </p>
            )}
          </div>
        )}

        {/* Settings Panel */}
        {video && videoDuration > 0 && (
          <div className="border rounded-lg p-4 bg-gray-50 space-y-5">
            <h3 className="font-semibold text-gray-700">Processing Settings</h3>

            {/* FPS Slider */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                FPS for processing:{' '}
                <span className="text-primary-600 font-bold">{settings.fps_sample}</span> fps
                <span className="text-gray-500 text-xs ml-2">
                  (~{estimatedFrames} frames)
                </span>
              </label>
              <input
                type="range"
                min="1"
                max="30"
                value={settings.fps_sample}
                onChange={(e) => updateSettings({ fps_sample: parseInt(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>1 fps (faster)</span>
                <span>30 fps (more frames)</span>
              </div>
            </div>

            {/* Number of Peaks Slider */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Number of peak frames:{' '}
                <span className="text-primary-600 font-bold">{settings.num_peaks}</span>
              </label>
              <input
                type="range"
                min="5"
                max="20"
                value={settings.num_peaks}
                onChange={(e) => updateSettings({ num_peaks: parseInt(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>5 peaks</span>
                <span>20 peaks</span>
              </div>
            </div>

            {/* Neutral Frame Mode Toggle */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Neutral frame selection mode
              </label>
              <div className="flex gap-2">
                <button
                  onClick={() => updateSettings({ neutral_mode: 'auto', manual_neutral_idx: null })}
                  className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                    settings.neutral_mode === 'auto'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Auto
                </button>
                <button
                  onClick={() => updateSettings({ neutral_mode: 'manual', manual_neutral_idx: 0 })}
                  className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                    settings.neutral_mode === 'manual'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Manual
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {settings.neutral_mode === 'auto'
                  ? 'Pipeline will automatically detect the best neutral (relaxed) frame'
                  : 'Manually select a frame where the dog has a neutral expression'}
              </p>
            </div>

            {/* Manual Frame Selector */}
            {settings.neutral_mode === 'manual' && (
              <div className="border-t pt-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select neutral frame:{' '}
                  <span className="text-green-600 font-bold">
                    {settings.manual_neutral_idx ?? 0}
                  </span>
                  <span className="text-gray-500 text-xs ml-2">
                    (time: {((settings.manual_neutral_idx ?? 0) / settings.fps_sample).toFixed(2)}s)
                  </span>
                </label>
                <input
                  type="range"
                  min="0"
                  max={maxNeutralIdx}
                  value={settings.manual_neutral_idx ?? 0}
                  onChange={(e) => updateSettings({ manual_neutral_idx: parseInt(e.target.value) })}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
                />
                <div className="mt-3">
                  <p className="text-sm text-gray-600 mb-2">Frame preview:</p>
                  <canvas
                    ref={canvasRef}
                    className="max-w-md mx-auto border-2 border-green-500 rounded-lg"
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Video info */}
        {video && (
          <div className="text-sm text-gray-600 flex gap-4">
            <p>
              <span className="font-semibold">File:</span> {video.name}
            </p>
            <p>
              <span className="font-semibold">Size:</span>{' '}
              {(video.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        )}

        {/* Actions */}
        <div className="flex space-x-3">
          <button
            onClick={handleProcess}
            disabled={!video || processing}
            className="flex-1 bg-primary-600 text-white px-6 py-3 rounded-md
              font-semibold hover:bg-primary-700 disabled:bg-gray-300
              disabled:cursor-not-allowed transition-colors"
          >
            {processing ? (
              <span className="flex items-center justify-center">
                <svg
                  className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Processing...
              </span>
            ) : (
              'Process Video'
            )}
          </button>

          {video && (
            <button
              onClick={handleReset}
              disabled={processing}
              className="px-6 py-3 rounded-md font-semibold
                bg-gray-200 text-gray-700 hover:bg-gray-300
                disabled:bg-gray-100 disabled:cursor-not-allowed
                transition-colors"
            >
              Reset
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
