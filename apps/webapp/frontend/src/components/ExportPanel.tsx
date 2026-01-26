/**
 * ExportPanel component - Export dataset to COCO format.
 */

import useStore from '../store/useStore';

export default function ExportPanel() {
  const { videoData, exportDataset } = useStore();

  if (!videoData) {
    return null;
  }

  const handleExport = async () => {
    await exportDataset();
  };

  return (
    <div className="mt-8 bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4">Export Dataset</h2>

      <div className="space-y-4">
        {/* Dataset Info */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-semibold">Video:</span> {videoData.video_filename}
          </div>
          <div>
            <span className="font-semibold">Total Frames:</span> {videoData.total_frames}
          </div>
          <div>
            <span className="font-semibold">Neutral Frame:</span> {videoData.neutral_frame_idx}
          </div>
          <div>
            <span className="font-semibold">Peak Frames:</span> {videoData.peak_frames.length}
          </div>
        </div>

        {/* Neutral Frame Preview */}
        <div className="border-t pt-4">
          <h3 className="font-semibold text-sm mb-2">Neutral Frame</h3>
          <img
            src={videoData.neutral_frame_url}
            alt="Neutral frame"
            className="w-48 h-32 object-cover rounded border"
          />
        </div>

        {/* Export Button */}
        <div className="border-t pt-4">
          <button
            onClick={handleExport}
            className="w-full bg-green-600 text-white px-6 py-3 rounded-md
              font-semibold hover:bg-green-700 transition-colors"
          >
            Export COCO JSON
          </button>
        </div>
      </div>
    </div>
  );
}
