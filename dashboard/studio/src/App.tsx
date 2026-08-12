import { BrowserRouter, Route, Routes } from "react-router-dom";
import StudioLayout from "./StudioLayout";
import HomePage from "./pages/HomePage";
import HowItWorksPage from "./pages/HowItWorksPage";
import ResultsPage from "./pages/ResultsPage";
import NewAnalysisPage from "./pages/NewAnalysisPage";
import AnalyzingPage from "./pages/AnalyzingPage";
import ReportPage from "./pages/ReportPage";
import NotFoundPage from "./pages/NotFoundPage";
import { SAMPLE_ANALYSIS_ID } from "./fixtures/holidaySale";

/**
 * Neural Bridge Studio.
 *
 * This package builds alone. There is no evidence browser, no scorecard, no
 * story and no document viewer in the output — not merely unreachable, absent,
 * because `verify-boundary.mjs` fails the build if anything here imports them.
 * See ADR-0006.
 *
 * BrowserRouter, not MemoryRouter: this site gets sent to people as a link, so
 * every page needs a real URL that survives a refresh. That requires two things
 * in step — `public/_redirects` rewriting every path to index.html, and
 * `base: "/"` in the Vite config so a deep link resolves its assets from the
 * root rather than from its own path.
 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<StudioLayout />}>
          <Route index element={<HomePage />} />
          <Route path="how-it-works" element={<HowItWorksPage />} />
          <Route path="results" element={<ResultsPage />} />
          {/* A shareable URL for the finished example. It renders the real
              report rather than redirecting, so the link someone is sent stays
              readable instead of turning into an opaque analysis id. */}
          <Route path="sample" element={<ReportPage analysisId={SAMPLE_ANALYSIS_ID} />} />
          <Route path="analyze" element={<NewAnalysisPage />} />
          <Route path="analyses/:id" element={<AnalyzingPage />} />
          <Route path="analyses/:id/report" element={<ReportPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
