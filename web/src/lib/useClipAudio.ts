import { useEffect, useRef, useState } from "react";
import { reportGlobalError } from "./globalError";
import { tr } from "./i18n";

/** Single-clip audio controller: only one clip plays at a time, with progress. */
export function useClipAudio() {
  const ref = useRef<HTMLAudioElement | null>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  // A detached `new Audio()` keeps playing after the owning component unmounts (page or
  // tab switch) with no UI left to stop it -- pause it on the way out.
  useEffect(
    () => () => {
      ref.current?.pause();
    },
    [],
  );

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
    // A failed load (404/401 clip) never fires onended; reset so the button isn't stuck
    // on ⏸, and say so -- a silent no-op reads as a dead button. AbortError is just a
    // rapid clip switch, not a failure.
    audio.play().catch((err: unknown) => {
      if ((err as DOMException)?.name !== "AbortError") {
        reportGlobalError(tr("Audio clip failed to load.", "音频片段加载失败。"));
      }
      setPlayingKey((prev) => (prev === key ? null : prev));
    });
    setPlayingKey(key);
    setProgress(0);
  }

  /** Seek the playing clip to a 0..1 fraction (progress-bar click). */
  function seek(fraction: number): void {
    const audio = ref.current;
    if (!audio || !audio.duration) return;
    audio.currentTime = Math.min(Math.max(fraction, 0), 1) * audio.duration;
  }

  return { playingKey, progress, toggle, seek };
}
