/**
 * PeakFramesGrid — siatka kart anotacji klatek.
 *
 * Wyświetla klatki z sesji (SessionData.frames).
 * Każda karta zawiera zakładki: Keypoints / AU / Emocja / Rasa.
 */

import PeakFrameCard from './PeakFrameCard';
import useStore from '../store/useStore';

export default function PeakFramesGrid() {
  const { sessionData } = useStore();

  if (!sessionData || sessionData.frames.length === 0) return null;

  return (
    <div className="mt-8">
      <h2 className="text-2xl font-bold mb-1">
        Klatki szczytowe ({sessionData.frames.length})
      </h2>
      <p className="text-sm text-gray-500 mb-4">
        Sesja: <code className="bg-gray-100 px-1 rounded">{sessionData.session_id}</code>
        · {sessionData.video_filename}
        · Klatka neutralna: #{sessionData.neutral_frame_idx}
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {sessionData.frames.map((frame) => (
          <PeakFrameCard key={frame.frame_idx} frame={frame} />
        ))}
      </div>
    </div>
  );
}
