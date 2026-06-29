import { useQuery } from "@tanstack/react-query";
import { getHealth } from "../api/client";

/**
 * Small version label in the topbar. Reuses the cached ["health"] query (also
 * driven by AuthGate), so it adds no extra request; /api/health is
 * unauthenticated, so the version shows even on the token-entry screen.
 */
export function AppVersion() {
  const { data } = useQuery({ queryKey: ["health"], queryFn: getHealth });
  if (!data?.version) return null;
  return (
    <span className="app-version mono" title="meeting-asr">
      v{data.version}
    </span>
  );
}
