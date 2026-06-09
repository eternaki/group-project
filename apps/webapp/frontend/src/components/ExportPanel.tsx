/**
 * ExportPanel component - Export dataset to COCO format.
 */

import useStore from '../store/useStore';

export default function ExportPanel() {
  const { videoData, sessionData, exportDataset, exportSession } = useStore();

  if (!videoData) {
    return null;
  }

  const handleExport = async () => {
    await exportDataset();
  };

  const handleExportSession = async () => {
    await exportSession();
  };

  return (
    <div className="mt-8 bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4">Eksport datasetu</h2>

      <div className="space-y-4">
        {/* Dataset Info */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-semibold">Wideo:</span> {videoData.video_filename}
          </div>
          <div>
            <span className="font-semibold">Liczba klatek:</span> {videoData.total_frames}
          </div>
          <div>
            <span className="font-semibold">Klatka neutralna:</span> {videoData.neutral_frame_idx}
          </div>
          <div>
            <span className="font-semibold">Klatki szczytowe:</span> {videoData.peak_frames.length}
          </div>
        </div>

        {/* Neutral Frame Preview */}
        <div className="border-t pt-4">
          <h3 className="font-semibold text-sm mb-2">Klatka neutralna</h3>
          <img
            src={videoData.neutral_frame_url}
            alt="Klatka neutralna"
            className="w-48 h-32 object-cover rounded border"
          />
        </div>

        {/* Export Buttons */}
        <div className="border-t pt-4 space-y-2">
          {sessionData && (
            <button
              onClick={handleExportSession}
              className="w-full bg-green-600 text-white px-6 py-3 rounded-md
                font-semibold hover:bg-green-700 transition-colors"
            >
              Eksport COCO JSON (z keypoints + AU)
            </button>
          )}
          <button
            onClick={handleExport}
            className="w-full bg-gray-500 text-white px-6 py-3 rounded-md
              font-semibold hover:bg-gray-600 transition-colors text-sm"
          >
            Eksport uproszczony (bez keypoints)
          </button>
        </div>
      </div>
    </div>
  );
}
