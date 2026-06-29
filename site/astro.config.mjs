// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// The live simulator (the FastAPI workbench), served on its custom domain.
// (The Fly app is still reachable at antennaknobs.fly.dev, but app.antennaknobs.dev
// is the canonical public URL.) Referenced from the home hero and the Web docs.
const SIMULATOR_URL = "https://app.antennaknobs.dev/";

// https://astro.build/config
export default defineConfig({
  site: "https://antennaknobs.dev",
  integrations: [
    starlight({
      title: "antennaknobs",
      tagline: "Turn a knob, watch the antenna.",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/stevenmburns/antennaknobs",
        },
      ],
      customCss: ["./src/styles/custom.css"],
      sidebar: [
        {
          label: "Start here",
          items: [
            { label: "What is antennaknobs?", slug: "start/welcome" },
            { label: "Quickstart", slug: "start/quickstart" },
            { label: "Open the live simulator", link: SIMULATOR_URL, attrs: { target: "_blank" } },
          ],
        },
        {
          label: "Concepts",
          items: [
            { label: "The model", slug: "concepts/model" },
            { label: "Many ways to express geometry", slug: "concepts/authoring" },
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "Design catalog", slug: "reference/catalog" },
            { label: "The solver & accuracy", slug: "reference/solver" },
            { label: "Web workbench", slug: "reference/web" },
            { label: "Command line", slug: "reference/cli" },
          ],
        },
        {
          label: "Project",
          items: [{ label: "Contributing", slug: "project/contributing" }],
        },
      ],
    }),
  ],
});
