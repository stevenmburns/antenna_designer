# antennaknobs site

The marketing landing (**antennaknobs.com**) and documentation
(**antennaknobs.dev**), built with [Astro](https://astro.build) +
[Starlight](https://starlight.astro.build). Lives in the monorepo alongside the
package so the docs can't drift from the code.

## Develop

```bash
cd site
npm install
npm run dev        # http://localhost:4321
```

## Build

```bash
npm run build      # type-checks (astro check) then builds to ./dist
npm run preview    # serve the production build locally
```

## Layout

```
src/
  content/docs/
    index.mdx          # home / splash (the .com-facing hero)
    start/             # welcome, quickstart
    concepts/          # the model; "many ways to express geometry" (centerpiece)
    reference/         # catalog, solver, web, cli
    project/           # contributing
  styles/custom.css    # brand accent
astro.config.mjs       # Starlight config + sidebar
```

## Notes

- The live simulator URL is defined once as `SIMULATOR_URL` in `astro.config.mjs` and
  referenced in the home hero / docs. **Swap it for the real domain** (e.g.
  `https://app.antennaknobs.dev`) once DNS is wired up.
- Pages contain `TODO` comments marking content that should be **generated from
  the package** (the catalog, API reference, benchmark plots) so docs stay in
  sync with the code rather than being hand-maintained.
- This is the initial scaffold: structure + the key pages (home, the five-ways
  authoring tour). The full `.com` marketing pages (`/gallery`, `/why`, `/learn`)
  from `docs/website-content-plan.md` are still to come.
```
