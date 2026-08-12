import EvidenceLink from "../components/EvidenceLink";
import BridgeScrollyChart from "./BridgeScrollyChart";
import { storySteps, ACT_TWO_START } from "./steps";
import { useActiveStep, usePrefersReducedMotion } from "./useActiveStep";

/**
 * Scroll-driven telling of the bridge result.
 *
 * The chart pins while the prose scrolls past it; each step reconfigures the
 * same chart rather than swapping in a new one.
 *
 * Under `prefers-reduced-motion` this renders a completely different layout —
 * every step's chart stacked with its prose, no pinning and no observer — so
 * the argument survives without any scrolling at all. That is a separate
 * layout on purpose: a "reduced" version of the pinned one would still depend
 * on scroll position to make sense.
 */
export default function ScrollyStory() {
  const reduced = usePrefersReducedMotion();
  const { active, registerStep } = useActiveStep(storySteps.length, !reduced);

  if (reduced) {
    return (
      <div className="story">
        <StoryHeader />
        <div className="story-static">
          {storySteps.map((step, i) => (
            <section key={step.id} className="story-static-step">
              {i === ACT_TWO_START && <hr className="story-act-break" />}
              <h3>{step.heading}</h3>
              <p>{step.body}</p>
              <BridgeScrollyChart step={step} animate={false} height={260} />
            </section>
          ))}
        </div>
        <StoryFooter />
      </div>
    );
  }

  const current = storySteps[active];

  return (
    <div className="story">
      <StoryHeader />

      <div className="story-scrolly">
        <div className="story-sticky">
          <BridgeScrollyChart step={current} animate />
          <ol className="story-progress" aria-hidden="true">
            {storySteps.map((step, i) => (
              <li key={step.id} className={i === active ? "on" : undefined} />
            ))}
          </ol>
        </div>

        <div className="story-steps">
          {storySteps.map((step, i) => (
            <section
              key={step.id}
              ref={registerStep(i)}
              className={`story-step${i === active ? " active" : ""} tone-${step.tone}`}
            >
              <h3>{step.heading}</h3>
              <p>{step.body}</p>
              <EvidenceLink sourceDoc={step.sourceDoc} className="story-source">
                Evidence
              </EvidenceLink>
            </section>
          ))}
        </div>
      </div>

      <StoryFooter />
    </div>
  );
}

function StoryHeader() {
  return (
    <header className="story-header">
      <p className="section-num">The bridge</p>
      <h1>Rich features weren&apos;t automatically useful</h1>
      <p className="scorecard-intro">
        Adding predicted cortical features to a strong autoregressive baseline made it
        <em> worse</em>. What changed that was not better features — it was the way across.
        Scroll to follow the six stages.
      </p>
    </header>
  );
}

function StoryFooter() {
  return (
    <p className="story-footer">
      All figures transcribed from{" "}
      <EvidenceLink sourceDoc="results/README.md">results/README.md</EvidenceLink>. The two acts
      run on different benchmarks and their numbers are not comparable across the break.
    </p>
  );
}
