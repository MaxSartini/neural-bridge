import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import OverviewPage from "./pages/OverviewPage";
import DocPage from "./pages/DocPage";
import BrowsePage from "./pages/BrowsePage";
import FileViewerPage from "./pages/FileViewerPage";
import ScorecardPage from "./pages/ScorecardPage";
import ScrollyStory from "./scrolly/ScrollyStory";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        {/* The designed overview, not a render of README.md. The raw markdown
            stays one click away at /doc/README.md — same principle as ADR-0002
            keeping the results table beside the charts. */}
        <Route path="/" element={<OverviewPage />} />
        <Route path="/scorecard" element={<ScorecardPage />} />
        <Route path="/story" element={<ScrollyStory />} />
        <Route path="/doc/*" element={<DocPage />} />
        <Route path="/browse" element={<BrowsePage />} />
        <Route path="/browse/*" element={<BrowsePage />} />
        <Route path="/view/*" element={<FileViewerPage />} />
        <Route path="*" element={<p>Not found.</p>} />
      </Route>
      {/* The client product used to mount at /app here. It is now its own
          package and its own deployment — see ADR-0006. Nothing in this app
          links to it: a cross-deployment link needs a real URL, which C7 adds
          once the Studio is live. */}
    </Routes>
  );
}
