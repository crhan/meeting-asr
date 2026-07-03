import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { captureAccept, captureRollback, getPendingCapture } from "../api/client";
import { confirmDialog } from "../lib/confirm";
import { tr } from "../lib/i18n";

/**
 * App-wide recovery for a capture transaction whose originating page is gone -- e.g. the user
 * navigated away (or reloaded / closed the tab) while the capture job was still running, so no
 * page ever learned the transaction id. Without a way to resolve it, the pending transaction
 * blocks every later store write with 409 until the server-side stale sweep. Polling surfaces
 * it anywhere and offers accept/rollback.
 */
export function PendingCaptureBanner() {
  const queryClient = useQueryClient();
  const { data } = useQuery({
    queryKey: ["pending-capture"],
    queryFn: getPendingCapture,
    refetchInterval: 5000,
    refetchOnWindowFocus: true,
  });

  const resolve = useMutation({
    mutationFn: (action: "accept" | "rollback") =>
      action === "accept"
        ? captureAccept(data!.transaction_id)
        : captureRollback(data!.transaction_id),
    onSuccess: () => {
      // Accept/rollback changed the global store + possibly a project's matches. Invalidate
      // the keys the pages actually use (by prefix) so library/quality/sample and speaker
      // review panes reflect it -- "voiceprints" was a stale guess that matched nothing.
      // capture-plan is included because CapturePage caches its plan with staleTime: Infinity;
      // resolving a capture from here (the cross-page recovery path) changes voiceprint/person
      // bindings, so returning to capture must re-plan rather than show the pre-capture plan.
      for (const key of [
        ["pending-capture"],
        ["vp-library"],
        ["vp-person"],
        ["vp-quality"],
        ["speakers"],
        ["capture-plan"],
      ]) {
        queryClient.invalidateQueries({ queryKey: key });
      }
    },
  });

  if (!data) return null;

  // The banner has no room for the capture's score details, so both one-click
  // resolutions of somebody-else's-work get a plain-language confirm first.
  const askRollback = async () => {
    if (
      await confirmDialog({
        message: tr(
          "Roll back this capture? The voiceprint library returns to its pre-capture state; you can re-run the capture at any time.",
          "回滚这次采集？声纹库将恢复到采集前的状态，之后可以随时重新采集。",
        ),
        confirmLabel: tr("Rollback", "回滚"),
      })
    )
      resolve.mutate("rollback");
  };
  const askAccept = async () => {
    if (
      await confirmDialog({
        message: tr(
          "Accept this capture and merge its samples into the voiceprint library?",
          "接受这次采集，把采到的样本并入声纹库？",
        ),
        confirmLabel: tr("Accept", "接受"),
      })
    )
      resolve.mutate("accept");
  };

  return (
    <div className="pending-capture-banner">
      <span>
        ⚠️{" "}
        {tr(
          "A voiceprint capture is awaiting review",
          "有一次声纹采集待处理",
        )}
        {data.project_id ? <span className="mono"> · {data.project_id}</span> : null}
      </span>
      <span className="row gap">
        <button className="btn ghost" disabled={resolve.isPending} onClick={askRollback}>
          {tr("Rollback", "回滚")}
        </button>
        <button className="btn" disabled={resolve.isPending} onClick={askAccept}>
          {tr("Accept", "接受")}
        </button>
      </span>
    </div>
  );
}
