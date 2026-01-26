/**
 * PeakFrameCard component - Display single peak frame with AU toggles.
 */

import { AU_NAMES, EMOTION_EMOJI, type PeakFrame } from '../types';
import useStore from '../store/useStore';

interface PeakFrameCardProps {
  frame: PeakFrame;
  frameIdx: number;
}

export default function PeakFrameCard({ frame, frameIdx }: PeakFrameCardProps) {
  const { toggleAU } = useStore();

  const handleToggleAU = (auName: string) => {
    toggleAU(frameIdx, auName);
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      {/* Frame Image */}
      <div className="relative">
        <img
          src={frame.image_url}
          alt={`Peak frame ${frame.frame_idx}`}
          className="w-full h-48 object-cover"
        />
        <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded text-xs">
          Frame {frame.frame_idx}
        </div>
      </div>

      <div className="p-4">
        {/* Emotion Display */}
        <div className="mb-4 p-3 bg-gradient-to-r from-primary-50 to-primary-100 rounded-lg text-center">
          <div className="text-4xl mb-1">{EMOTION_EMOJI[frame.emotion]}</div>
          <div className="font-bold text-lg text-gray-800">{frame.emotion.toUpperCase()}</div>
          <div className="text-sm text-gray-600">
            Confidence: {(frame.emotion_confidence * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {frame.emotion_rule_applied}
          </div>
        </div>

        {/* TFM Score */}
        <div className="mb-3 text-sm text-gray-600 text-center">
          <span className="font-semibold">TFM Score:</span> {frame.tfm_score.toFixed(3)}
        </div>

        {/* AU Toggles */}
        <div className="border-t pt-3">
          <h4 className="font-semibold text-sm mb-2 text-gray-700">Action Units</h4>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(frame.aus).map(([auName, au]) => (
              <label
                key={auName}
                className={`flex items-center space-x-2 p-2 rounded cursor-pointer transition-colors ${
                  au.is_active
                    ? 'bg-green-50 border border-green-300'
                    : 'bg-gray-50 border border-gray-200'
                }`}
              >
                <input
                  type="checkbox"
                  checked={au.is_active}
                  onChange={() => handleToggleAU(auName)}
                  className="rounded text-primary-600 focus:ring-primary-500"
                />
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-semibold text-gray-700">{auName}</div>
                  <div className="text-xs text-gray-500 truncate">
                    {AU_NAMES[auName] || auName}
                  </div>
                  <div className="text-xs text-gray-600">
                    {au.delta > 0 ? '+' : ''}
                    {(au.delta * 100).toFixed(0)}%
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
