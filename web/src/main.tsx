import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { seedTokenFromUrl } from "./lib/auth";
import { reportGlobalError } from "./lib/globalError";
import { tr } from "./lib/i18n";
import "./styles.css";

// Capture ?token= from the entry URL before anything renders or fetches.
seedTokenFromUrl();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 5_000, refetchOnWindowFocus: false },
    // Default surface for mutation failures. Most CRUD mutations (voiceprints, lexicon,
    // settings, capture banner, correction accept) only set onSuccess, so a failure --
    // including the deliberate 409s while a capture transaction is pending -- was
    // previously invisible. Mutations that render their own error pass an explicit
    // onError, which replaces this default (per-option merge), so nothing double-reports.
    mutations: {
      onError: (error) =>
        reportGlobalError(
          tr("Operation failed: ", "操作失败：") + (error as Error).message,
        ),
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
