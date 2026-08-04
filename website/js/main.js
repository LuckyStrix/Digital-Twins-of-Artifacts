const ICONS_BY_TYPE = {
  tablet: "assets/icons/ClayTablet1.svg",
  scroll: "assets/icons/PapyrusScroll1.svg",
  papyrus: "assets/icons/PapyrusScroll1.svg",
};
const FALLBACK_ICON = "assets/icons/Generic1.svg";

function iconForType(type) {
  return ICONS_BY_TYPE[type.trim().toLowerCase()] || FALLBACK_ICON;
}

function titleCase(str) {
  return str.replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase());
}

function truncate(text, max) {
  if (text.length <= max) return text;
  return text.slice(0, max - 1).trimEnd() + "…";
}

// Shared by both the "Artifacts" grid (our own digital twins) and the
// "MISHA-Imaged Artifacts" grid — only the link list differs between them.
function renderCard(artifact, links, extraClass) {
  const card = document.createElement("article");
  card.className = extraClass ? `artifact-card ${extraClass}` : "artifact-card";

  const icon = document.createElement("img");
  icon.className = "card-icon";
  icon.src = iconForType(artifact.type || "");
  icon.alt = "";
  icon.loading = "lazy";

  const title = document.createElement("h4");
  title.textContent = artifact.name;

  const badge = document.createElement("span");
  badge.className = "type-badge";
  badge.textContent = titleCase(artifact.type || "other");

  const desc = document.createElement("p");
  desc.className = "card-description";
  desc.textContent = truncate(artifact.description || "", 160);

  const linksWrap = document.createElement("div");
  linksWrap.className = "card-links";
  for (const { href, text } of links) {
    const link = document.createElement("a");
    link.className = "card-link";
    link.href = href;
    link.textContent = text;
    linksWrap.appendChild(link);
  }

  card.append(icon, title, badge, desc, linksWrap);
  return card;
}

function viewerLink(artifact) {
  return { href: `viewer.html?id=${encodeURIComponent(artifact.id)}`, text: "View in 3D →" };
}

function msiLink(artifact) {
  return { href: `msi.html?id=${encodeURIComponent(artifact.id)}`, text: "Explore Spectral Bands →" };
}

async function loadArtifacts() {
  const container = document.getElementById("artifact-groups");
  const status = document.getElementById("artifact-status");
  const mishaSection = document.getElementById("misha-artifacts");
  const mishaContainer = document.getElementById("misha-artifact-groups");
  const mishaStatus = document.getElementById("misha-artifact-status");

  let data;
  try {
    const res = await fetch("artifacts/manifest.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`manifest request failed: ${res.status}`);
    data = await res.json();
  } catch (err) {
    console.error(err);
    status.textContent = "Couldn't load artifacts. Make sure artifacts/manifest.json exists (run tools/build_manifest.py).";
    mishaSection.style.display = "none";
    return;
  }

  const allArtifacts = data.artifacts || [];
  // MISHA data comes from an external collaborator (see the section intro)
  // and gets its own section rather than being folded into our own
  // type-grouped digital-twin gallery below.
  const mishaArtifacts = allArtifacts.filter((a) => a.msi);
  const ownArtifacts = allArtifacts.filter((a) => !a.msi);

  renderOwnArtifacts(ownArtifacts, container, status);
  renderMishaArtifacts(mishaArtifacts, mishaContainer, mishaStatus, mishaSection);
}

function renderOwnArtifacts(artifacts, container, status) {
  if (artifacts.length === 0) {
    status.textContent = "No artifacts yet — add a folder to artifacts/ and rerun tools/build_manifest.py.";
    return;
  }
  status.remove();

  const groups = new Map();
  for (const artifact of artifacts) {
    const type = (artifact.type || "other").trim() || "other";
    if (!groups.has(type)) groups.set(type, []);
    groups.get(type).push(artifact);
  }

  const sortedTypes = [...groups.keys()].sort((a, b) => a.localeCompare(b));

  for (const type of sortedTypes) {
    const section = document.createElement("div");
    section.className = "artifact-type-group";

    const heading = document.createElement("h3");
    heading.className = "type-heading";
    heading.textContent = titleCase(type);

    const grid = document.createElement("div");
    grid.className = "card-grid";
    for (const artifact of groups.get(type)) {
      grid.appendChild(renderCard(artifact, [viewerLink(artifact)]));
    }

    section.append(heading, grid);
    container.appendChild(section);
  }
}

function renderMishaArtifacts(artifacts, container, status, section) {
  if (artifacts.length === 0) {
    // No MISHA-imaged artifacts yet (e.g. a fresh clone of this repo) —
    // hide the whole section rather than show an empty external-collaboration
    // callout with nothing under it.
    section.style.display = "none";
    return;
  }
  status.remove();

  const grid = document.createElement("div");
  grid.className = "card-grid";
  for (const artifact of artifacts) {
    const links = [msiLink(artifact)];
    if (artifact.model) links.push(viewerLink(artifact));
    grid.appendChild(renderCard(artifact, links, "misha-card"));
  }
  container.appendChild(grid);
}

loadArtifacts();
