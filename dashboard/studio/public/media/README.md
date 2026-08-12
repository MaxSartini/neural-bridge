# Studio imagery

Drop image files here and they ship with the site. Nothing else is needed —
`public/` is copied to the build root, so a file placed at
`public/media/hero.jpg` is served at `/media/hero.jpg`.

To use one, pass its path to `ImageSlot`:

```tsx
<ImageSlot src="/media/hero.jpg" alt="A director reviewing a cut" fallback={<FrameStripArt />} />
```

Until a file exists, every slot renders generated artwork instead, so the site
is complete without photography rather than showing gaps.

## Constraints that are not negotiable

- **Vendored, never hotlinked.** The deployed policy is
  `img-src 'self' data:` — an image loaded from a CDN or a stock service is
  blocked outright, and `CONVENTIONS.md` forbids runtime third-party requests
  regardless.
- **Licensed for commercial use.** This site is shown to prospective customers
  and investors. Keep the licence or receipt somewhere findable.
- **No recognisable private individuals** without a release.

## Practical notes

- **Format:** `.jpg` for photographs, `.png` only when transparency is needed,
  `.webp` welcome. Vite fingerprints and cache-busts them automatically.
- **Size:** export at roughly 2× the display size and keep files under ~400 KB.
  A hero displays around 1000px wide, so 2000px is plenty.
- **Every photograph is duotoned.** `ImageSlot` applies the green screen-print
  the design system requires, so images arrive on-brand no matter their source
  palette. Composition and contrast survive that treatment; subtle colour does
  not — choose images that read from their shapes.

## Slots currently in the design

| Where | Suggested subject | Ratio |
|---|---|---|
| Home hero | The work being reviewed — an edit suite, a timeline, a screening | 16 / 9 |
| How it works | Something concrete per step, or leave as artwork | 4 / 3 |
| Results opener | Abstract is fine here; the numbers carry the page | 16 / 9 |
