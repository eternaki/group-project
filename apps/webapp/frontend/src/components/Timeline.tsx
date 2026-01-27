/**
 * Timeline component - Visual representation of neutral and peak frames.
 */

import useStore from '../store/useStore';
import { EMOTION_EMOJI } from '../types';

export default function Timeline() {
  const { videoData } = useStore();

  if (!videoData) return null;

  const { neutral_frame_idx, peak_frames, total_frames } = videoData;

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mt-6">
      <h2 className="text-xl font-bold mb-4">Timeline</h2>

      {/* Timeline bar */}
      <div className="relative w-full h-16 bg-gray-200 rounded-lg overflow-visible">
        {/* Neutral frame marker */}
        <div
          className="absolute top-0 h-full w-1 bg-green-500 z-20"
          style={{ left: `${(neutral_frame_idx / total_frames) * 100}%` }}
          title={`Neutral frame: ${neutral_frame_idx}`}
        >
          <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs font-bold text-green-700 whitespace-nowrap">
            Neutral
          </div>
          <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-xs text-gray-600">
            {neutral_frame_idx}
          </div>
        </div>

        {/* Peak frame markers */}
        {peak_frames.map((peak, idx) => (
          <div
            key={peak.frame_idx}
            className="absolute top-0 h-full w-1 bg-red-500 z-10 cursor-pointer hover:w-2 transition-all"
            style={{ left: `${(peak.frame_idx / total_frames) * 100}%` }}
            title={`Peak ${idx + 1}: Frame ${peak.frame_idx} - ${peak.emotion}`}
          >
            <div className="absolute top-1/2 -translate-y-1/2 left-1/2 -translate-x-1/2">
              <span className="text-lg">{EMOTION_EMOJI[peak.emotion] || ''}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex gap-6 mt-8 text-sm">
        <span className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded"></div>
          Neutral frame ({neutral_frame_idx})
        </span>
        <span className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 rounded"></div>
          Peak frames ({peak_frames.length})
        </span>
      </div>

      {/* Stats */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="bg-gray-50 p-3 rounded">
          <div className="text-gray-500">Total frames</div>
          <div className="font-bold text-lg">{total_frames}</div>
        </div>
        <div className="bg-green-50 p-3 rounded">
          <div className="text-gray-500">Neutral frame</div>
          <div className="font-bold text-lg text-green-700">{neutral_frame_idx}</div>
        </div>
        <div className="bg-red-50 p-3 rounded">
          <div className="text-gray-500">Peak frames</div>
          <div className="font-bold text-lg text-red-700">{peak_frames.length}</div>
        </div>
        <div className="bg-blue-50 p-3 rounded">
          <div className="text-gray-500">Emotions detected</div>
          <div className="font-bold text-lg text-blue-700">
            {[...new Set(peak_frames.map((p) => p.emotion))].length}
          </div>
        </div>
      </div>
    </div>
  );
}
