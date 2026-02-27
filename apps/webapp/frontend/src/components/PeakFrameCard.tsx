/**
 * PeakFrameCard — karta anotacji jednej klatki.
 *
 * Zawiera 4 zakładki:
 *   [Keypoints] — canvas editor drag&drop 46 punktów
 *   [AU]        — panel slajderów 21 Action Units
 *   [Emocja]    — wybór emocji z 9 klas DogFACS
 *   [Rasa]      — wyszukiwarka rasy psa
 */

import { useState } from 'react';
import type { FrameAnnotation } from '../types';
import { EMOTION_EMOJI } from '../types';
import AnnotationStatusBadge from './AnnotationStatusBadge';
import AUPanel from './AUPanel';
import BreedPicker from './BreedPicker';
import DogSelector from './DogSelector';
import EmotionSelector from './EmotionSelector';
import KeypointEditor from './KeypointEditor';

interface PeakFrameCardProps {
  frame: FrameAnnotation;
}

type TabId = 'keypoints' | 'au' | 'emotion' | 'breed';

const TABS: { id: TabId; label: string }[] = [
  { id: 'keypoints', label: 'Keypoints' },
  { id: 'au',        label: 'AU' },
  { id: 'emotion',   label: 'Emocja' },
  { id: 'breed',     label: 'Rasa' },
];

export default function PeakFrameCard({ frame }: PeakFrameCardProps) {
  const [activeTab, setActiveTab] = useState<TabId>('keypoints');
  const [selectedDogIdx, setSelectedDogIdx] = useState(0);

  const activeAUCount = Object.values(frame.aus).filter((au) => au.is_active).length;

  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden flex flex-col">
      {/* Miniatura klatki */}
      <div className="relative bg-gray-900">
        <img
          src={frame.image_url}
          alt={`Frame ${frame.frame_idx}`}
          className="w-full h-36 object-cover"
        />
        {/* Badges na obrazku */}
        <div className="absolute top-2 left-2">
          <AnnotationStatusBadge status={frame.annotation_status} />
        </div>
        <div className="absolute top-2 right-2 bg-black bg-opacity-60 text-white px-2 py-0.5 rounded text-xs">
          #{frame.frame_idx}
        </div>

        {/* Podsumowanie — emocja + TFM */}
        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 to-transparent px-3 py-2 flex items-end justify-between">
          <div className="flex items-center gap-1.5">
            {frame.emotion && (
              <>
                <span className="text-lg">{EMOTION_EMOJI[frame.emotion] ?? '❓'}</span>
                <span className="text-white text-sm font-semibold capitalize">{frame.emotion}</span>
                {frame.emotion_confidence > 0 && (
                  <span className="text-gray-300 text-xs">
                    {(frame.emotion_confidence * 100).toFixed(0)}%
                  </span>
                )}
              </>
            )}
          </div>
          <div className="text-gray-300 text-xs">
            TFM {frame.tfm_score.toFixed(2)}
          </div>
        </div>
      </div>

      {/* DogSelector + AU summary */}
      <div className="px-3 py-2 border-b border-gray-100 flex items-center justify-between gap-2 flex-wrap">
        <DogSelector dogCount={1} selectedDogIdx={selectedDogIdx} onSelect={setSelectedDogIdx} />
        <span className="text-xs text-gray-400">
          AU aktywnych: <strong className="text-amber-600">{activeAUCount}</strong>/21
          {frame.breed && <> · <span className="text-indigo-600">{frame.breed}</span></>}
        </span>
      </div>

      {/* Zakładki */}
      <div className="flex border-b border-gray-200 bg-gray-50">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-2 text-xs font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-blue-700 border-b-2 border-blue-500 bg-white'
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Zawartość zakładki */}
      <div className="p-3 flex-1 overflow-y-auto max-h-80">
        {activeTab === 'keypoints' && (
          <KeypointEditor
            frameIdx={frame.frame_idx}
            imageUrl={frame.image_url}
            keypoints={frame.keypoints}
          />
        )}

        {activeTab === 'au' && (
          <AUPanel
            frameIdx={frame.frame_idx}
            aus={frame.aus}
          />
        )}

        {activeTab === 'emotion' && (
          <EmotionSelector
            frameIdx={frame.frame_idx}
            currentEmotion={frame.emotion}
            confidence={frame.emotion_confidence}
            ruleApplied={frame.emotion_rule_applied}
          />
        )}

        {activeTab === 'breed' && (
          <BreedPicker
            frameIdx={frame.frame_idx}
            currentBreed={frame.breed}
            confidence={frame.breed_confidence}
          />
        )}
      </div>
    </div>
  );
}
