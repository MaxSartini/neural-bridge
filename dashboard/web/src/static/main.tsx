import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import StaticScorecard from "./StaticScorecard";
import "../styles/global.css";

const container = document.getElementById("root");
if (!container) {
  throw new Error("root element not found");
}

createRoot(container).render(
  <StrictMode>
    <StaticScorecard />
  </StrictMode>,
);
