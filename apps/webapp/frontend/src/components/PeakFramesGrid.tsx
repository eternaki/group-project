/**
 * PeakFramesGrid component - Grid of peak frame cards.
 */

import PeakFrameCard from './PeakFrameCard';
import useStore from '../store/useStore';

export default function PeakFramesGrid() {
  const { videoData } = useStore();

  if (!videoData || videoData.peak_frames.length === 0) {
    return null;
  }

  return (
    <div className="mt-8">
      <h2 className="text-2xl font-bold mb-4">
        Peak Frames ({videoData.peak_frames.length})
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {videoData.peak_frames.map((frame, idx) => (
          <PeakFrameCard key={frame.frame_idx} frame={frame} frameIdx={idx} />
        ))}
      </div>
    </div>
  );
}
