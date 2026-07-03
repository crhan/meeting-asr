/** Clickable clip progress bar: click position seeks to that fraction (the audio
 *  endpoints serve Range requests). Shared by every page that plays clips. */
export function SeekBar({
  progress,
  onSeek,
}: {
  progress: number;
  onSeek: (fraction: number) => void;
}) {
  return (
    <div
      className="seg-progress seekable"
      onClick={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        onSeek((e.clientX - rect.left) / rect.width);
      }}
    >
      <div className="seg-progress-bar" style={{ width: `${progress * 100}%` }} />
    </div>
  );
}
