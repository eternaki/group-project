/**
 * VideoUpload component - Upload video file.
 */

import { useRef } from 'react';
import useStore from '../store/useStore';

export default function VideoUpload() {
  const inputRef = useRef<HTMLInputElement>(null);
  const { video, uploadVideo, processVideo, processing, reset } = useStore();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadVideo(file);
    }
  };

  const handleProcess = async () => {
    await processVideo();
  };

  const handleReset = () => {
    reset();
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4">Upload Video</h2>

      <div className="space-y-4">
        {/* File input */}
        <div>
          <label
            htmlFor="video-upload"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Select dog video (20s recommended)
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

        {/* Video info */}
        {video && (
          <div className="text-sm text-gray-600">
            <p>
              <span className="font-semibold">Selected:</span> {video.name}
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
