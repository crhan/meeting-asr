import { useRef, useState } from "react";

/** Single-clip audio controller: only one clip plays at a time, with progress. */
export function useClipAudio() {
  const ref = useRef<HTMLAudioElement | null>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  function ensure(): HTMLAudioElement {
    if (!ref.current) {
      const audio = new Audio();
      audio.ontimeupdate = () => {
        if (audio.duration) setProgress(audio.currentTime / audio.duration);
      };
      audio.onended = () => setPlayingKey(null);
      ref.current = audio;
    }
    return ref.current;
  }

  function toggle(key: string, url: string): void {
    const audio = ensure();
    if (playingKey === key && !audio.paused) {
      audio.pause();
      setPlayingKey(null);
      return;
    }
    audio.src = url;
    void audio.play();
    setPlayingKey(key);
    setProgress(0);
  }

  return { playingKey, progress, toggle };
}
