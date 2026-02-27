/**
 * EmotionSelector — wybór emocji z 9 klas DogFACS.
 *
 * Wyświetla siatkę emocji z emoji i podpowiedziami AU.
 * Kliknięcie zapisuje emocję przez PATCH /emotion.
 */

import { EMOTION_AU_HINTS, EMOTION_CLASSES, EMOTION_EMOJI, type EmotionClass } from '../types';
import useStore from '../store/useStore';

interface EmotionSelectorProps {
  frameIdx: number;
  currentEmotion: string | null;
  confidence: number;
  ruleApplied: string | null;
}

export default function EmotionSelector({
  frameIdx,
  currentEmotion,
  confidence,
  ruleApplied,
}: EmotionSelectorProps) {
  const { updateFrameEmotion, saving } = useStore();

  const handleSelect = async (emotion: EmotionClass) => {
    if (emotion === currentEmotion) return;
    await updateFrameEmotion(frameIdx, emotion);
  };

  return (
    <div className="space-y-3">
      {/* Aktualnie ustawiona emocja */}
      {currentEmotion && (
        <div className="flex items-center gap-2 p-2 bg-gray-50 rounded-lg border border-gray-200">
          <span className="text-2xl">{EMOTION_EMOJI[currentEmotion]}</span>
          <div>
            <div className="text-sm font-bold capitalize text-gray-800">{currentEmotion}</div>
            <div className="text-xs text-gray-500">
              {confidence > 0 ? `Pewność: ${(confidence * 100).toFixed(0)}%` : ''}
              {ruleApplied && ruleApplied !== 'manual' ? ` · ${ruleApplied}` : ''}
            </div>
          </div>
        </div>
      )}

      {/* Siatka emocji */}
      <div className="grid grid-cols-3 gap-2">
        {EMOTION_CLASSES.map((emotion) => {
          const isSelected = emotion === currentEmotion;
          return (
            <button
              key={emotion}
              onClick={() => handleSelect(emotion)}
              disabled={saving}
              title={EMOTION_AU_HINTS[emotion]}
              className={`flex flex-col items-center p-2 rounded-lg border transition-all text-center ${
                isSelected
                  ? 'bg-purple-100 border-purple-400 ring-2 ring-purple-300'
                  : 'bg-white border-gray-200 hover:bg-purple-50 hover:border-purple-300'
              } disabled:opacity-50`}
            >
              <span className="text-xl mb-1">{EMOTION_EMOJI[emotion]}</span>
              <span className="text-xs font-medium capitalize leading-tight">{emotion}</span>
              <span className="text-[10px] text-gray-400 leading-tight mt-0.5">
                {EMOTION_AU_HINTS[emotion]}
              </span>
            </button>
          );
        })}
      </div>

      {saving && (
        <p className="text-xs text-center text-gray-400">Zapisuję…</p>
      )}
    </div>
  );
}
